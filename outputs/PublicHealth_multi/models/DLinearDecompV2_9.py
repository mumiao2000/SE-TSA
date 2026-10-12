import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DLinearDecompV2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_kernels': [2, 3, 4],
            'dropout': [0.0, 0.3],
            'freq_dim': [8, 32, 128],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_kernels = self.config.get('n_kernels', 3)
        self.dropout_rate = self.config.get('dropout', 0.3)
        self.freq_dim = self.config.get('freq_dim', 32)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale decomposition kernels (learnable moving averages)
        # Kernel sizes chosen to capture weekly, monthly, and seasonal patterns
        kernel_sizes = [max(3, input_len // (2 ** i)) | 1 for i in range(self.n_kernels)]  # odd sizes
        self.decomp_kernels = nn.ModuleList()
        for k in kernel_sizes:
            # Ensure kernel size is valid (odd and <= input_len)
            k = min(k, input_len)
            if k % 2 == 0:
                k = max(1, k - 1)
            padding = k // 2
            self.decomp_kernels.append(
                nn.AvgPool1d(kernel_size=k, stride=1, padding=padding, count_include_pad=False)
            )
        
        # Decomposition projection: combine multi-scale trends
        self.trend_proj = nn.Linear(self.n_kernels * input_len, input_len)
        self.seasonal_proj = nn.Linear(self.n_kernels * input_len, input_len)
        
        # Adaptive channel gating
        self.channel_gate = nn.Sequential(
            nn.Linear(n_features, n_features // 4),
            nn.ReLU(),
            nn.Linear(n_features // 4, n_features),
            nn.Sigmoid()
        )
        
        # Frequency-domain refinement branch
        self.freq_encoder = nn.Sequential(
            nn.Linear(n_features, self.freq_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.freq_dim, n_features)
        )
        
        # Channel-dependent temporal projections (separate for trend and seasonal)
        self.trend_linears = nn.ModuleList([
            nn.Linear(input_len, output_len) for _ in range(n_features)
        ])
        self.seasonal_linears = nn.ModuleList([
            nn.Linear(input_len, output_len) for _ in range(n_features)
        ])
        
        # Residual connection for direct mapping
        self.residual_linear = nn.Linear(input_len, output_len)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(n_features * 3, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, n_features)
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Multi-scale decomposition
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Compute trends at multiple scales
        trends = []
        seasonal_all = []
        for kernel in self.decomp_kernels:
            trend = kernel(x_perm)  # (batch, n_features, input_len)
            # Handle size mismatch from padding
            if trend.shape[-1] != x_perm.shape[-1]:
                trend = F.interpolate(trend, size=x_perm.shape[-1], mode='linear', align_corners=False)
            seasonal = x_perm - trend
            trends.append(trend)
            seasonal_all.append(seasonal)
        
        # Concatenate multi-scale decompositions and project back
        trend_cat = torch.cat(trends, dim=-1)  # (batch, n_features, n_kernels * input_len)
        seasonal_cat = torch.cat(seasonal_all, dim=-1)  # (batch, n_features, n_kernels * input_len)
        
        trend = self.trend_proj(trend_cat)  # (batch, n_features, input_len)
        seasonal = self.seasonal_proj(seasonal_cat)  # (batch, n_features, input_len)
        
        # Adaptive channel gating based on input feature statistics
        feature_stats = x.mean(dim=1)  # (batch, n_features)
        gate = self.channel_gate(feature_stats)  # (batch, n_features)
        gate = gate.unsqueeze(-1)  # (batch, n_features, 1)
        
        # Apply gating
        trend = trend * gate  # (batch, n_features, input_len)
        seasonal = seasonal * gate  # (batch, n_features, input_len)
        
        # Frequency-domain refinement
        # Compute FFT on each feature's time series
        x_fft = torch.fft.rfft(x_perm, dim=-1)  # (batch, n_features, freq_bins)
        freq_magnitude = torch.abs(x_fft)  # (batch, n_features, freq_bins)
        # Use mean frequency profile as feature descriptor
        freq_profile = freq_magnitude.mean(dim=-1, keepdim=True)  # (batch, n_features, 1)
        freq_profile = freq_profile.permute(0, 2, 1)  # (batch, 1, n_features)
        freq_refined = self.freq_encoder(freq_profile.squeeze(1))  # (batch, n_features)
        freq_refined = freq_refined.unsqueeze(-1)  # (batch, n_features, 1)
        
        # Channel-dependent temporal projections
        trend_out = []
        seasonal_out = []
        for i in range(self.n_features):
            t = self.trend_linears[i](trend[:, i, :])  # (batch, output_len)
            s = self.seasonal_linears[i](seasonal[:, i, :])  # (batch, output_len)
            trend_out.append(t)
            seasonal_out.append(s)
        
        trend_out = torch.stack(trend_out, dim=1)  # (batch, n_features, output_len)
        seasonal_out = torch.stack(seasonal_out, dim=1)  # (batch, n_features, output_len)
        
        # Residual connection
        residual = self.residual_linear(x_perm)  # (batch, n_features, output_len)
        
        # Combine with frequency refinement
        combined = trend_out + seasonal_out + residual
        combined = combined * (1 + freq_refined)  # frequency-aware modulation
        
        # Fusion with gating information
        # Stack different components for fusion
        comp_stack = torch.cat([
            trend_out, seasonal_out, residual
        ], dim=1)  # (batch, 3 * n_features, output_len)
        comp_stack = comp_stack.permute(0, 2, 1)  # (batch, output_len, 3 * n_features)
        
        fused = self.fusion(comp_stack)  # (batch, output_len, n_features)
        
        # Final residual: combine fused with direct sum
        output = combined.permute(0, 2, 1) + self.dropout(fused)  # (batch, output_len, n_features)
        
        output = self.revin(output, mode='denorm')  # (batch, output_len, n_features)
        return output
