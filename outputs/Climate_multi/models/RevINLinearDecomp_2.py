import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class RevINLinearDecomp(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'kernel_size': [3, 7, 15],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.kernel_size = self.config.get('kernel_size', 7)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Ensure odd kernel size for padding
        self.pad = self.kernel_size // 2
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Trend extractor: moving average
        self.trend_conv = nn.Conv1d(
            self.n_features, self.n_features, 
            kernel_size=self.kernel_size, 
            padding=self.pad, 
            groups=self.n_features,  # channel-wise
            bias=False
        )
        # Initialize as averaging filter
        with torch.no_grad():
            self.trend_conv.weight.fill_(1.0 / self.kernel_size)
        
        # Seasonal = input - trend
        # Project trend and seasonal separately
        self.trend_fc = nn.Linear(self.input_len, self.output_len)
        self.seasonal_fc = nn.Linear(self.input_len, self.output_len)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.n_features * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode="norm")  # (batch, input_len, n_features)
        
        # Extract trend
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        trend = self.trend_conv(x_perm)  # (batch, n_features, input_len)
        trend = trend.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Seasonal component
        seasonal = x - trend  # (batch, input_len, n_features)
        
        # Project each component
        trend_perm = trend.permute(0, 2, 1)  # (batch, n_features, input_len)
        seasonal_perm = seasonal.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        trend_out = self.trend_fc(trend_perm)  # (batch, n_features, output_len)
        seasonal_out = self.seasonal_fc(seasonal_perm)  # (batch, n_features, output_len)
        
        trend_out = trend_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        seasonal_out = seasonal_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Concatenate and fuse
        combined = torch.cat([trend_out, seasonal_out], dim=-1)  # (batch, output_len, n_features * 2)
        out = self.fusion(combined)  # (batch, output_len, n_features)
        
        # Residual connection
        out = out + trend_out  # (batch, output_len, n_features)
        
        out = self.revin(out, mode="denorm")  # (batch, output_len, n_features)
        return out
