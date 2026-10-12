import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanNBEATS(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'koopman_dim': [64, 256, 1024],
            'n_blocks': [1, 4, 8],
            'hidden_dim': [64, 256, 512],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.koopman_dim = self.config.get('koopman_dim', 256)
        self.n_blocks = self.config.get('n_blocks', 4)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to Koopman space
        self.encoder = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.koopman_dim)
        )
        
        # Koopman operator (linear dynamics)
        self.koopman_K = nn.Parameter(torch.randn(self.koopman_dim, self.koopman_dim) * 0.01)
        
        # N-BEATS-style trend/seasonality decomposition blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(nn.Sequential(
                nn.Linear(self.koopman_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.koopman_dim)
            ))
        
        # Output projection
        self.decoder = nn.Sequential(
            nn.Linear(self.koopman_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
        # Backcast/forecast weights for N-BEATS structure
        self.backcast_length = input_len
        self.forecast_length = output_len
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Encode last timestep to Koopman space
        x_last = x[:, -1, :]  # (batch, n_features)
        z = self.encoder(x_last)  # (batch, koopman_dim)
        
        # Apply Koopman operator for multi-step prediction
        forecasts = []
        z_t = z
        for t in range(self.output_len):
            # Linear evolution
            z_t = torch.matmul(z_t, self.koopman_K.t())  # (batch, koopman_dim)
            
            # Apply N-BEATS residual blocks
            residual = z_t
            for block in self.blocks:
                residual = residual + block(residual)  # (batch, koopman_dim)
            z_t = residual
            
            # Decode to feature space
            pred = self.decoder(z_t)  # (batch, n_features)
            forecasts.append(pred)
        
        out = torch.stack(forecasts, dim=1)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
