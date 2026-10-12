import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DLinearChannelHybrid(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'kernel_size': [3, 13, 25],
            'hidden_dim': [64, 256, 1024],
            'n_mixing_layers': [1, 2, 4],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.kernel_size = self.config.get('kernel_size', 13)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_mixing_layers = self.config.get('n_mixing_layers', 2)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        # Ensure kernel_size is odd
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Decomposition: moving average for trend
        self.decomp = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        
        # Linear layers for trend and seasonality (channel-independent)
        self.trend_linear = nn.Linear(input_len, output_len)
        self.seasonal_linear = nn.Linear(input_len, output_len)
        
        # Cross-channel mixing layers
        self.mixing_layers = nn.ModuleList()
        for _ in range(self.n_mixing_layers):
            self.mixing_layers.append(
                nn.Sequential(
                    nn.Conv1d(n_features, self.hidden_dim, 1),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Conv1d(self.hidden_dim, n_features, 1)
                )
            )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Decomposition per channel
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Trend component
        trend = self.decomp(x_perm)  # (batch, n_features, input_len)
        # Handle padding effects
        if trend.shape[-1] > self.input_len:
            trend = trend[:, :, :self.input_len]
        elif trend.shape[-1] < self.input_len:
            pad = self.input_len - trend.shape[-1]
            trend = F.pad(trend, (0, pad))
        
        # Seasonal component
        seasonal = x_perm - trend  # (batch, n_features, input_len)
        
        # Linear projection
        trend_out = self.trend_linear(trend)  # (batch, n_features, output_len)
        seasonal_out = self.seasonal_linear(seasonal)  # (batch, n_features, output_len)
        
        # Combine
        out = trend_out + seasonal_out  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Cross-channel mixing with residual connections
        for mixing in self.mixing_layers:
            residual = out
            out = out.permute(0, 2, 1)  # (batch, n_features, output_len)
            out = mixing(out)  # (batch, n_features, output_len)
            out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
            out = self.layer_norm(out + residual)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
