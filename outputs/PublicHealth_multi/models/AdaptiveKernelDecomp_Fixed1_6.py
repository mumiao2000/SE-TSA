import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveKernelDecomp_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_kernels': [3, 5, 7],
            'trend_dim': [16, 64, 256],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_kernels = self.config.get('n_kernels', 5)
        self.trend_dim = self.config.get('trend_dim', 64)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Kernel sizes: exponentially spaced from small to large
        self.kernel_sizes = [2**i + 1 for i in range(self.n_kernels)]  # 3, 5, 9, 17, 33, ...
        # Clip to input_len and ensure odd (critical fix: even kernel sizes cause length mismatch)
        self.kernel_sizes = [min(k, self.input_len) for k in self.kernel_sizes]
        self.kernel_sizes = [k if k % 2 == 1 else max(k - 1, 1) for k in self.kernel_sizes]
        # Remove duplicates that may arise from clipping
        seen = set()
        self.kernel_sizes = []
        for k in [min(2**i + 1, self.input_len) for i in range(self.n_kernels)]:
            k = k if k % 2 == 1 else max(k - 1, 1)
            if k not in seen:
                seen.add(k)
                self.kernel_sizes.append(k)
        self.n_kernels = len(self.kernel_sizes)  # update actual number of kernels
        
        # Periodicity estimator: simple FFT-based module
        fft_len = self.input_len // 2 + 1
        self.periodicity_est = nn.Sequential(
            nn.Linear(fft_len, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.n_kernels)
        )
        
        # Smoothing layers using Conv1d with uniform weights (moving average)
        # This guarantees output length equals input length with proper padding (odd kernels)
        self.smoothers = nn.ModuleList()
        for k in self.kernel_sizes:
            smoother = nn.Conv1d(
                in_channels=self.n_features,
                out_channels=self.n_features,
                kernel_size=k,
                padding=k // 2,
                groups=self.n_features,
                bias=False
            )
            # Initialize with uniform weights to mimic average pooling
            with torch.no_grad():
                smoother.weight.fill_(1.0 / k)
            self.smoothers.append(smoother)
        
        # Trend processors
        self.trend_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_len, self.trend_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.trend_dim, self.output_len)
            ) for _ in range(self.n_kernels)
        ])
        
        # Seasonal processor
        self.seasonal_proj = nn.Sequential(
            nn.Linear(self.input_len, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.output_len)
        )
        
        # Residual processor
        self.residual_proj = nn.Sequential(
            nn.Linear(self.input_len, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.output_len)
        )
        
        # Mixing weights
        self.mix = nn.Linear(self.n_kernels + 2, 1)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Estimate periodicity via FFT
        x_fft = torch.fft.rfft(x_perm, dim=-1)  # (batch, n_features, input_len // 2 + 1)
        x_fft_mag = torch.abs(x_fft)  # (batch, n_features, input_len // 2 + 1)
        
        # Get kernel weights: (batch, n_features, n_kernels)
        kernel_weights = F.softmax(self.periodicity_est(x_fft_mag), dim=-1)  # (batch, n_features, n_kernels)
        
        # Extract trends with multiple kernels
        trends = []
        for i, smoother in enumerate(self.smoothers):
            trend_i = smoother(x_perm)  # (batch, n_features, input_len)
            # Project to output
            trend_out = self.trend_projs[i](trend_i)  # (batch, n_features, output_len)
            # Weight by learned importance
            w = kernel_weights[:, :, i:i+1]  # (batch, n_features, 1)
            trends.append(trend_out * w)  # (batch, n_features, output_len)
        
        # Combined trend
        trend_combined = sum(trends)  # (batch, n_features, output_len)
        
        # Extract seasonal (residual after weighted trend removal)
        # Use weighted average of trends for seasonal extraction
        trend_for_seasonal = sum([
            self.smoothers[i](x_perm) * kernel_weights[:, :, i:i+1] 
            for i in range(self.n_kernels)
        ])  # (batch, n_features, input_len)
        
        seasonal = x_perm - trend_for_seasonal  # (batch, n_features, input_len)
        seasonal_out = self.seasonal_proj(seasonal)  # (batch, n_features, output_len)
        
        # Residual pathway
        residual_out = self.residual_proj(x_perm)  # (batch, n_features, output_len)
        
        # Stack and mix
        stacked = torch.stack(trends + [seasonal_out, residual_out], dim=-1)  # (batch, n_features, output_len, n_kernels + 2)
        
        # Learned mixing per timestep and feature
        out = self.mix(stacked).squeeze(-1)  # (batch, n_features, output_len)
        
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

