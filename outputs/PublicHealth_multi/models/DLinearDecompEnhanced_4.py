import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DLinearDecompEnhanced(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'kernel_size': [3, 7, 13],
            'hidden_ratio': [1, 2, 4],
            'use_seasonal': [True, False],
            'individual': [True, False],
            'use_gate': [True, False],
            'use_channel_mix': [True, False],
            'mix_ratio': [4, 8, 16],  # reduction ratio for channel mixing
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.kernel_size = cfg.get('kernel_size', 3)
        self.hidden_ratio = cfg.get('hidden_ratio', 2)
        self.use_seasonal = cfg.get('use_seasonal', True)
        self.individual = cfg.get('individual', True)
        self.use_gate = cfg.get('use_gate', True)
        self.use_channel_mix = cfg.get('use_channel_mix', False)
        self.mix_ratio = cfg.get('mix_ratio', 8)
        
        # Ensure odd kernel size for symmetric padding
        self.pad_size = self.kernel_size // 2
        
        # Multi-scale decomposition: primary + secondary kernel for capturing 51-week period
        self.avg_pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)
        # Secondary larger kernel for long-term trends
        self.large_kernel = max(self.kernel_size * 3 + 1, 25)  # at least 25 for yearly patterns
        self.large_pad = self.large_kernel // 2
        self.avg_pool_large = nn.AvgPool1d(kernel_size=self.large_kernel, stride=1, padding=0)
        
        # Learnable weights for combining multi-scale trends
        self.trend_weights = nn.Parameter(torch.ones(2) / 2)  # (2,)
        
        hidden_dim = int(self.output_len * self.hidden_ratio)
        
        # Trend projection
        if self.individual:
            self.trend_linears = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.input_len, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.output_len)
                ) for _ in range(self.n_features)
            ])
        else:
            self.trend_linear = nn.Sequential(
                nn.Linear(self.input_len, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.output_len)
            )
        
        # Seasonal projection (optional)
        if self.use_seasonal:
            if self.individual:
                self.seasonal_linears = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.input_len, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, self.output_len)
                    ) for _ in range(self.n_features)
                ])
            else:
                self.seasonal_linear = nn.Sequential(
                    nn.Linear(self.input_len, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.output_len)
                )
        
        # Gating mechanism for adaptive trend/seasonal blending
        if self.use_gate:
            self.gate_proj = nn.Linear(self.input_len, self.n_features)  # per-feature gate
        
        # Channel mixing via low-rank bottleneck
        if self.use_channel_mix:
            self.mix_down = nn.Linear(self.n_features, self.n_features // self.mix_ratio)
            self.mix_up = nn.Linear(self.n_features // self.mix_ratio, self.n_features)
            self.mix_act = nn.GELU()
        
        self.revin = RevIN(self.n_features, affine=True)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for 1D operations: (batch, n_features, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Multi-scale trend extraction with reflection padding
        x_padded = F.pad(x_perm, (self.pad_size, self.pad_size), mode='reflect')  # (batch, n_features, input_len + 2*pad)
        trend_primary = self.avg_pool(x_padded)  # (batch, n_features, input_len)
        
        x_padded_large = F.pad(x_perm, (self.large_pad, self.large_pad), mode='reflect')  # (batch, n_features, input_len + 2*large_pad)
        trend_large = self.avg_pool_large(x_padded_large)  # (batch, n_features, input_len)
        
        # Adaptive combination of multi-scale trends
        w = F.softmax(self.trend_weights, dim=0)  # (2,)
        trend = w[0] * trend_primary + w[1] * trend_large  # (batch, n_features, input_len)
        
        # Seasonal = original - trend
        seasonal = x_perm - trend  # (batch, n_features, input_len)
        
        # Project trend
        if self.individual:
            trend_out = torch.stack([
                self.trend_linears[i](trend[:, i, :])  # (batch, output_len)
                for i in range(self.n_features)
            ], dim=1)  # (batch, n_features, output_len)
        else:
            trend_out = self.trend_linear(trend)  # (batch, n_features, output_len)
        
        # Project seasonal (optional)
        if self.use_seasonal:
            if self.individual:
                seasonal_out = torch.stack([
                    self.seasonal_linears[i](seasonal[:, i, :])  # (batch, output_len)
                    for i in range(self.n_features)
                ], dim=1)  # (batch, n_features, output_len)
            else:
                seasonal_out = self.seasonal_linear(seasonal)  # (batch, n_features, output_len)
        else:
            seasonal_out = 0
        
        # Adaptive gating
        if self.use_gate:
            # Compute per-feature gate based on input statistics
            gate_input = x_perm.mean(dim=-1)  # (batch, n_features)
            gate = torch.sigmoid(self.gate_proj(gate_input))  # (batch, n_features)
            gate = gate.unsqueeze(-1)  # (batch, n_features, 1)
            out = gate * trend_out + (1 - gate) * seasonal_out if self.use_seasonal else trend_out  # (batch, n_features, output_len)
        else:
            out = trend_out + seasonal_out if self.use_seasonal else trend_out  # (batch, n_features, output_len)
        
        # Channel mixing
        if self.use_channel_mix:
            # Transpose: (batch, output_len, n_features)
            out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
            out = out + self.mix_up(self.mix_act(self.mix_down(out)))  # (batch, output_len, n_features), residual
            out = out.permute(0, 2, 1)  # (batch, n_features, output_len)
        
        # Transpose back: (batch, output_len, n_features)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out
