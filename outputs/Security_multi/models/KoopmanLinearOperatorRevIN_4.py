import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanLinearOperatorRevIN(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'obs_dim': [64, 256, 1024],
            'n_blocks': [2, 4, 8],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.obs_dim = self.config.get('obs_dim', 256)
        self.n_blocks = self.config.get('n_blocks', 4)
        self.dropout_p = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(n_features, affine=True)
        
        # Encoder: nonlinear observation function
        self.encoder = nn.Sequential(
            nn.Linear(n_features, self.obs_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.obs_dim, self.obs_dim)
        )
        
        # Koopman operator: parametrized as stable matrix
        # Use real Schur form with 2x2 blocks for oscillatory modes
        self.koopman_diag = nn.Parameter(torch.randn(self.obs_dim // 2) * 0.01)
        self.koopman_off = nn.Parameter(torch.randn(self.obs_dim // 2) * 0.01)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.obs_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.obs_dim, n_features)
        )
        
        # Initial state processing
        self.init_processor = nn.LSTM(
            self.obs_dim, self.obs_dim, 
            num_layers=1, batch_first=True
        )

    def _build_koopman_matrix(self):
        # Build block diagonal matrix with 2x2 rotation-like blocks
        # Each block: [[a, -b], [b, a]] with constraint a^2 + b^2 < 1 for stability
        device = self.koopman_diag.device
        K = torch.zeros(self.obs_dim, self.obs_dim, device=device)
        
        for i in range(self.obs_dim // 2):
            a = torch.tanh(self.koopman_diag[i]) * 0.99  # ensure |a| < 1
            b = torch.tanh(self.koopman_off[i]) * 0.99
            # Normalize to ensure eigenvalue magnitude < 1
            scale = torch.sqrt(a**2 + b**2 + 1e-6)
            if scale > 0.99:
                a = a * 0.99 / scale
                b = b * 0.99 / scale
            
            K[2*i, 2*i] = a
            K[2*i, 2*i+1] = -b
            K[2*i+1, 2*i] = b
            K[2*i+1, 2*i+1] = a
        
        return K

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Encode to observables
        z = self.encoder(x)  # (batch, input_len, obs_dim)
        
        # Process initial sequence
        z_processed, (h, c) = self.init_processor(z)  # (batch, input_len, obs_dim)
        
        # Take last state as initial condition
        z_t = z_processed[:, -1, :]  # (batch, obs_dim)
        
        # Get Koopman operator
        K = self._build_koopman_matrix()  # (obs_dim, obs_dim)
        
        # Evolve in Koopman space
        predictions = []
        for t in range(self.output_len):
            z_t = torch.matmul(z_t, K.t())  # (batch, obs_dim)
            predictions.append(z_t)
        
        z_future = torch.stack(predictions, dim=1)  # (batch, output_len, obs_dim)
        
        # Decode
        out = self.decoder(z_future)  # (batch, output_len, n_features)
        
        # Add residual: repeat last known value
        last_val = x[:, -1:, :].expand(-1, self.output_len, -1)  # (batch, output_len, n_features)
        out = out + last_val  # (batch, output_len, n_features)
        
        x = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)

