import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DLinearChannelMix(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'mixing_ratio': [0.25, 0.5, 0.75],
            'decomp_kernel': [3, 11, 25],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.mixing_ratio = self.config.get('mixing_ratio', 0.5)
        self.decomp_kernel = self.config.get('decomp_kernel', 11)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Ensure kernel is odd
        if self.decomp_kernel % 2 == 0:
            self.decomp_kernel += 1
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Moving average decomposition
        self.moving_avg = nn.AvgPool1d(kernel_size=self.decomp_kernel, stride=1, padding=self.decomp_kernel//2)
        
        # Feature-wise linear layers for trend and seasonality
        self.trend_linears = nn.ModuleList([
            nn.Linear(self.input_len, self.output_len) for _ in range(self.n_features)
        ])
        self.seasonal_linears = nn.ModuleList([
            nn.Linear(self.input_len, self.output_len) for _ in range(self.n_features)
        ])
        
        # Channel mixing layers
        self.mix_in = nn.Linear(self.n_features, int(self.n_features * self.mixing_ratio))
        self.mix_out = nn.Linear(int(self.n_features * self.mixing_ratio), self.n_features)
        
        # Residual connection with channel mixing
        self.residual_mix = nn.Sequential(
            nn.Linear(self.input_len, self.output_len),
            nn.Dropout(self.dropout)
        )
        
        # Adaptive weights for trend/seasonal combination
        self.adaptive_weights = nn.Parameter(torch.ones(self.n_features, 2))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Decompose each feature
        # x: (batch, input_len, n_features) -> (batch, n_features, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Moving average on each feature
        trend = self.moving_avg(x_perm)  # (batch, n_features, input_len)
        seasonal = x_perm - trend  # (batch, n_features, input_len)
        
        # Apply feature-wise linear projections
        trend_outs = []
        seasonal_outs = []
        for feat_idx in range(self.n_features):
            trend_feat = trend[:, feat_idx, :]  # (batch, input_len)
            seasonal_feat = seasonal[:, feat_idx, :]  # (batch, input_len)
            
            trend_out = self.trend_linears[feat_idx](trend_feat)  # (batch, output_len)
            seasonal_out = self.seasonal_linears[feat_idx](seasonal_feat)  # (batch, output_len)
            
            trend_outs.append(trend_out)
            seasonal_outs.append(seasonal_out)
        
        trend_stack = torch.stack(trend_outs, dim=-1)  # (batch, output_len, n_features)
        seasonal_stack = torch.stack(seasonal_outs, dim=-1)  # (batch, output_len, n_features)
        
        # Adaptive combination
        weights = F.softmax(self.adaptive_weights, dim=-1)  # (n_features, 2)
        combined = (weights[:, 0] * trend_stack + weights[:, 1] * seasonal_stack)  # (batch, output_len, n_features)
        
        # Channel mixing
        mixed = self.mix_in(combined)  # (batch, output_len, n_features * mixing_ratio)
        mixed = F.gelu(mixed)
        mixed = self.mix_out(mixed)  # (batch, output_len, n_features)
        
        # Residual connection
        residual = self.residual_mix(x_perm).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = combined + mixed + residual  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
