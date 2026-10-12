import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveDecomp(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_blocks': [1, 4, 8],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_blocks = self.config.get('n_blocks', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Per-channel learnable decomposition: each feature has its own smoothing kernel
        self.trend_kernels = nn.Parameter(torch.randn(self.n_features, 7) * 0.02)  # learnable 1D conv kernel per channel
        self.trend_bias = nn.Parameter(torch.zeros(self.n_features))
        
        # Per-channel MLPs for trend and seasonality
        self.trend_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_len, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, output_len)
            ) for _ in range(n_features)
        ])
        
        self.season_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_len, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, output_len)
            ) for _ in range(n_features)
        ])
        
        # Cross-channel gating
        self.gate_proj = nn.Sequential(
            nn.Linear(n_features * 2, n_features * 2),
            nn.Sigmoid()
        )
        
        # Residual connection
        self.residual = nn.Linear(input_len, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Per-channel trend extraction with learnable smoothing
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Apply per-channel 1D convolution for trend (padding to maintain length)
        trend_components = []
        for i in range(self.n_features):
            xi = x_perm[:, i:i+1, :]  # (batch, 1, input_len)
            kernel = self.trend_kernels[i:i+1, :]  # (1, 7)
            # Manual conv1d with padding
            pad = 3
            xi_padded = F.pad(xi, (pad, pad), mode='replicate')  # (batch, 1, input_len+6)
            # Unfold for convolution
            xi_unfold = F.unfold(xi_padded.unsqueeze(-1), kernel_size=(7, 1), stride=1)  # (batch, 7, input_len)
            xi_unfold = xi_unfold.permute(0, 2, 1)  # (batch, input_len, 7)
            trend_i = torch.matmul(xi_unfold, kernel.squeeze().unsqueeze(-1)).squeeze(-1)  # (batch, input_len)
            trend_i = trend_i + self.trend_bias[i]
            trend_components.append(trend_i.unsqueeze(1))  # (batch, 1, input_len)
        
        trend = torch.cat(trend_components, dim=1)  # (batch, n_features, input_len)
        season = x_perm - trend  # (batch, n_features, input_len)
        
        # Per-channel forecasting
        trend_out = []
        season_out = []
        for i in range(self.n_features):
            ti = self.trend_mlps[i](trend[:, i, :])  # (batch, output_len)
            si = self.season_mlps[i](season[:, i, :])  # (batch, output_len)
            trend_out.append(ti.unsqueeze(1))  # (batch, 1, output_len)
            season_out.append(si.unsqueeze(1))  # (batch, 1, output_len)
        
        trend_out = torch.cat(trend_out, dim=1)  # (batch, n_features, output_len)
        season_out = torch.cat(season_out, dim=1)  # (batch, n_features, output_len)
        
        # Cross-channel gating
        combined = torch.cat([trend_out, season_out], dim=1)  # (batch, n_features*2, output_len)
        combined = combined.permute(0, 2, 1)  # (batch, output_len, n_features*2)
        gate = self.gate_proj(combined)  # (batch, output_len, n_features*2)
        
        trend_gate = gate[:, :, :self.n_features].permute(0, 2, 1)  # (batch, n_features, output_len)
        season_gate = gate[:, :, self.n_features:].permute(0, 2, 1)  # (batch, n_features, output_len)
        
        out = trend_out * trend_gate + season_out * season_gate  # (batch, n_features, output_len)
        
        # Residual
        res = self.residual(x_perm)  # (batch, n_features, output_len)
        out = out + res  # (batch, n_features, output_len)
        
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

