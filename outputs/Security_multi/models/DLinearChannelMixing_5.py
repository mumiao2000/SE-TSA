import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DLinearChannelMixing(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'kernel_size': [3, 13, 51],
            'hidden_dim': [64, 256, 512],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.kernel_size = self.config.get('kernel_size', 13)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Ensure kernel_size is odd
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Moving average for decomposition
        self.moving_avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        
        # Trend projection: linear layer on full sequence
        self.trend_linear = nn.Sequential(
            nn.Linear(input_len, hidden_dim := self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_len)
        )
        
        # Seasonality projection
        self.season_linear = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_len)
        )
        
        # Channel mixing: after per-feature projection, mix channels
        self.channel_mix = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(n_features, n_features)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Decompose: trend = moving average, seasonality = residual
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Moving average decomposition
        trend = self.moving_avg(x_perm)  # (batch, n_features, input_len)
        # Handle padding effects
        if trend.shape[-1] != x_perm.shape[-1]:
            trend = trend[:, :, :x_perm.shape[-1]]
        
        seasonality = x_perm - trend  # (batch, n_features, input_len)
        
        # Project trend and seasonality separately
        trend_out = self.trend_linear(trend)  # (batch, n_features, output_len)
        season_out = self.season_linear(seasonality)  # (batch, n_features, output_len)
        
        # Combine
        out = trend_out + season_out  # (batch, n_features, output_len)
        
        # Channel mixing
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        out = self.channel_mix(out)  # (batch, output_len, n_features)
        
        # Add residual from direct linear projection
        direct = x_perm  # (batch, n_features, input_len)
        direct_out = self.trend_linear(direct) + self.season_linear(direct)  # (batch, n_features, output_len)
        direct_out = direct_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = out + self.residual_weight * direct_out  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

