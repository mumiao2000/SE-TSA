import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DLinearResidual(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'kernel_size': [3, 7, 25],
            'hidden_dim': [64, 256, 1024],
            'use_seasonal': [True, False],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.kernel_size = self.config.get('kernel_size', 7)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.use_seasonal = self.config.get('use_seasonal', True)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Ensure kernel_size is odd and valid
        self.kernel_size = min(self.kernel_size, input_len // 2 * 2 + 1)
        self.kernel_size = max(self.kernel_size, 3)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
        self.padding = (self.kernel_size - 1) // 2
        
        # Moving average for trend extraction
        self.avg_pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=self.padding)
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Trend branch
        self.trend_linear = nn.Sequential(
            nn.Linear(input_len, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, output_len)
        )
        
        # Seasonal/residual branch
        if self.use_seasonal:
            self.seasonal_linear = nn.Sequential(
                nn.Linear(input_len, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, output_len)
            )
        
        # Learnable combination weights
        self.combine = nn.Parameter(torch.ones(2))
        
        # Direct residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # Instance normalization
        x_norm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_norm = self.norm(x_norm)  # (batch, n_features, input_len)
        x_norm = x_norm.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Get mean and std for denormalization
        mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        
        # Normalize
        x_input = (x - mean) / std  # (batch, input_len, n_features)
        
        # Decompose into trend and seasonal
        x_perm = x_input.permute(0, 2, 1)  # (batch, n_features, input_len)
        trend = self.avg_pool(x_perm)  # (batch, n_features, input_len)
        seasonal = x_perm - trend  # (batch, n_features, input_len)
        
        # Trend prediction
        trend_out = self.trend_linear(trend)  # (batch, n_features, output_len)
        
        if self.use_seasonal:
            # Seasonal prediction
            seasonal_out = self.seasonal_linear(seasonal)  # (batch, n_features, output_len)
            
            # Learnable combination
            weights = F.softmax(self.combine, dim=0)
            out = weights[0] * trend_out + weights[1] * seasonal_out  # (batch, n_features, output_len)
        else:
            out = trend_out  # (batch, n_features, output_len)
        
        # Add residual connection from last values
        last_vals = x_perm[:, :, -1:]  # (batch, n_features, 1)
        residual = last_vals.expand(-1, -1, self.output_len)  # (batch, n_features, output_len)
        out = out + self.residual_weight * residual  # (batch, n_features, output_len)
        
        # Permute and denormalize
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        out = out * std + mean  # (batch, output_len, n_features)
        
        return out

