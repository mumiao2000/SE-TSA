import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanAutoRevIN_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'obs_dim': [64, 256, 1024],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3],
            'koopman_reg': [0.0, 0.01, 0.1]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.obs_dim = self.config.get('obs_dim', 256)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        self.koopman_reg = self.config.get('koopman_reg', 0.01)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: lift to observables
        encoder_layers = []
        in_dim = n_features
        for i in range(self.n_layers):
            out_dim = self.obs_dim if i == self.n_layers - 1 else self.obs_dim // 2
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                encoder_layers.append(nn.LayerNorm(out_dim))
                encoder_layers.append(nn.GELU())
                encoder_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Koopman operator: linear evolution matrix
        # We learn it as a structured matrix for stability
        self.K_diag = nn.Parameter(torch.randn(self.obs_dim) * 0.01)
        self.K_off = nn.Parameter(torch.randn(self.obs_dim, self.obs_dim) * 0.001)
        
        # For periodic systems, add imaginary component
        self.K_freq = nn.Parameter(torch.randn(self.obs_dim // 2) * 0.1)
        
        # Decoder
        decoder_layers = []
        in_dim = self.obs_dim
        for i in range(self.n_layers):
            out_dim = n_features if i == self.n_layers - 1 else self.obs_dim // 2
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                decoder_layers.append(nn.LayerNorm(out_dim))
                decoder_layers.append(nn.GELU())
                decoder_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Also predict from full sequence using attention-like pooling
        self.time_pool = nn.Linear(n_features, 1)
        
    def get_koopman_matrix(self):
        # Construct stable Koopman matrix
        # K = diag(exp(K_diag)) + skew(K_off - K_off^T) for stability
        K = torch.diag(torch.exp(self.K_diag))  # (obs_dim, obs_dim)
        skew = self.K_off - self.K_off.T  # (obs_dim, obs_dim)
        K = K + skew * 0.1  # small skew component
        
        # Add rotation for periodicity (block diagonal 2x2 rotations)
        for i in range(0, self.obs_dim - 1, 2):
            if i // 2 < len(self.K_freq):
                freq = self.K_freq[i // 2]
                rot = torch.tensor([[torch.cos(freq), -torch.sin(freq)],
                                   [torch.sin(freq), torch.cos(freq)]], device=K.device)
                K[i:i+2, i:i+2] = rot
        
        return K
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Encode each time step to observables
        z = self.encoder(x)  # (batch, input_len, obs_dim)
        
        # Get Koopman matrix
        K = self.get_koopman_matrix()  # (obs_dim, obs_dim)
        
        # Initialize from last observed state
        z_t = z[:, -1, :]  # (batch, obs_dim)
        
        # Evolve with Koopman operator
        predictions = []
        for t in range(self.output_len):
            z_t = z_t @ K.T  # (batch, obs_dim)
            predictions.append(z_t)
        
        z_pred = torch.stack(predictions, dim=1)  # (batch, output_len, obs_dim)
        
        # Decode
        out = self.decoder(z_pred)  # (batch, output_len, n_features)
        
        # Add residual: use time-weighted average of input
        w = F.softmax(self.time_pool(x), dim=1)  # (batch, input_len, 1)
        x_res = (x.transpose(1, 2) @ w).squeeze(-1)  # (batch, n_features, input_len) @ (batch, input_len, 1) = (batch, n_features, 1) -> (batch, n_features)
        out = out + x_res.unsqueeze(1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

