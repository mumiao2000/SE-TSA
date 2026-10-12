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
        
        # Koopman encoder: lift to observable space
        self.koopman_encoder = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.koopman_dim)
        )  # (batch, input_len, koopman_dim)
        
        # Koopman operator (learned linear dynamics)
        self.K = nn.Parameter(torch.randn(self.koopman_dim, self.koopman_dim) * 0.01)
        
        # N-BEATS decomposition blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(
                NBEATSBlock(self.koopman_dim, self.hidden_dim, self.output_len, self.dropout)
            )
        
        # Project back to feature space
        self.koopman_decoder = nn.Sequential(
            nn.Linear(self.koopman_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )  # (batch, output_len, n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Lift to Koopman observable space
        z = self.koopman_encoder(x)  # (batch, input_len, koopman_dim)
        
        # Take last timestep as initial condition
        z_last = z[:, -1, :]  # (batch, koopman_dim)
        
        # Propagate with Koopman operator
        outputs = []
        z_t = z_last
        for t in range(self.output_len):
            z_t = torch.matmul(z_t, self.K.t())  # (batch, koopman_dim)
            outputs.append(z_t)
        z_future = torch.stack(outputs, dim=1)  # (batch, output_len, koopman_dim)
        
        # N-BEATS decomposition on Koopman observables
        residual = z_future
        forecast = torch.zeros_like(residual)  # (batch, output_len, koopman_dim)
        for block in self.blocks:
            block_forecast, block_residual = block(residual)
            forecast = forecast + block_forecast  # (batch, output_len, koopman_dim)
            residual = block_residual  # (batch, output_len, koopman_dim)
        
        # Decode back to original feature space
        out = self.koopman_decoder(forecast)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class NBEATSBlock(nn.Module):
    def __init__(self, koopman_dim, hidden_dim, output_len, dropout):
        super().__init__()
        self.koopman_dim = koopman_dim
        self.output_len = output_len
        
        # Trend and seasonality projections
        self.trend_fc = nn.Sequential(
            nn.Linear(koopman_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len * koopman_dim)
        )
        self.seasonal_fc = nn.Sequential(
            nn.Linear(koopman_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len * koopman_dim)
        )
        self.backcast_fc = nn.Linear(koopman_dim, koopman_dim)
        
    def forward(self, x):
        # x: (batch, output_len, koopman_dim)
        batch_size = x.shape[0]
        
        # Pool across time for global features
        x_pooled = x.mean(dim=1)  # (batch, koopman_dim)
        
        # Trend component (linear growth)
        trend = self.trend_fc(x_pooled)  # (batch, output_len * koopman_dim)
        trend = trend.view(batch_size, self.output_len, self.koopman_dim)  # (batch, output_len, koopman_dim)
        
        # Seasonal component
        seasonal = self.seasonal_fc(x_pooled)  # (batch, output_len * koopman_dim)
        seasonal = seasonal.view(batch_size, self.output_len, self.koopman_dim)  # (batch, output_len, koopman_dim)
        
        forecast = trend + seasonal  # (batch, output_len, koopman_dim)
        
        # Backcast for residual connection
        backcast = self.backcast_fc(x_pooled)  # (batch, koopman_dim)
        backcast = backcast.unsqueeze(1).expand(-1, self.output_len, -1)  # (batch, output_len, koopman_dim)
        residual = x - backcast  # (batch, output_len, koopman_dim)
        
        return forecast, residual
