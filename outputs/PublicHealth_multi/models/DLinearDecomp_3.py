import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DLinearDecomp(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'kernel_size': [3, 13, 25],
            'hidden_ratio': [1, 2, 4],
            'use_seasonal': [True, False],
            'individual': [True, False]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.kernel_size = self.config.get('kernel_size', 13)
        self.hidden_ratio = self.config.get('hidden_ratio', 2)
        self.use_seasonal = self.config.get('use_seasonal', True)
        self.individual = self.config.get('individual', True)
        
        # Ensure odd kernel size
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Decomposition: moving average for trend
        self.decomp = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        
        hidden_dim = int(self.n_features * self.hidden_ratio)
        
        if self.individual:
            # Channel-wise linear layers
            self.trend_linears = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_len, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.output_len)
                ) for _ in range(self.n_features)
            ])
            if self.use_seasonal:
                self.seasonal_linears = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.input_len, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, self.output_len)
                    ) for _ in range(self.n_features)
                ])
        else:
            # Shared across channels
            self.trend_linears = nn.Sequential(
                nn.Linear(self.input_len, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.output_len)
            )
            if self.use_seasonal:
                self.seasonal_linears = nn.Sequential(
                    nn.Linear(self.input_len, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.output_len)
                )
        
        # Residual connection
        self.residual = nn.Linear(self.input_len, self.output_len)
        
        self.use_seasonal = self.use_seasonal
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Decomposition
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Trend: moving average
        trend = self.decomp(x_perm)  # (batch, n_features, input_len)
        seasonal = x_perm - trend  # (batch, n_features, input_len)
        
        # Process trend
        if self.individual:
            trend_out = []
            for i, linear in enumerate(self.trend_linears):
                t = linear(trend[:, i, :])  # (batch, output_len)
                trend_out.append(t)
            trend_out = torch.stack(trend_out, dim=1)  # (batch, n_features, output_len)
        else:
            trend_out = self.trend_linears(trend)  # (batch, n_features, output_len)
        
        # Process seasonal
        if self.use_seasonal:
            if self.individual:
                seasonal_out = []
                for i, linear in enumerate(self.seasonal_linears):
                    s = linear(seasonal[:, i, :])  # (batch, output_len)
                    seasonal_out.append(s)
                seasonal_out = torch.stack(seasonal_out, dim=1)  # (batch, n_features, output_len)
            else:
                seasonal_out = self.seasonal_linears(seasonal)  # (batch, n_features, output_len)
            
            out = trend_out + seasonal_out  # (batch, n_features, output_len)
        else:
            out = trend_out  # (batch, n_features, output_len)
        
        # Add residual
        res = self.residual(x_perm)  # (batch, n_features, output_len)
        out = out + res  # (batch, n_features, output_len)
        
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
