import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DLinearDecompV2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_kernels': [2, 3, 5],
            'mix_ratio': [0.0, 0.3, 0.7],
            'dropout': [0.0, 0.3],
            'n_freq_bases': [4, 8, 16],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_kernels = self.config.get('n_kernels', 3)
        self.mix_ratio = self.config.get('mix_ratio', 0.3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.n_freq_bases = self.config.get('n_freq_bases', 8)
        
        # Multi-scale decomposition kernels: capture weekly, monthly, and annual patterns
        # Dominant period is ~51 weeks, so we use kernels that divide this reasonably
        kernel_candidates = [3, 7, 13, 25, 51]
        self.kernel_sizes = kernel_candidates[:self.n_kernels]
        
        # Learnable moving average decompositions
        self.decomp_layers = nn.ModuleList()
        for k in self.kernel_sizes:
            # Padding to maintain sequence length
            padding = k // 2
            self.decomp_layers.append(
                nn.AvgPool1d(kernel_size=k, stride=1, padding=padding, count_include_pad=False)
            )
        
        # Adaptive weights for combining multi-scale trends
        self.trend_weights = nn.Parameter(torch.ones(self.n_kernels) / self.n_kernels)
        
        # Frequency-aware projections: learnable Fourier bases for seasonal modeling
        # Create frequency bases for the dominant period and harmonics
        self.freq_bases = nn.Parameter(torch.randn(self.n_freq_bases, input_len) * 0.02)
        self.freq_proj = nn.Linear(self.n_freq_bases, output_len, bias=False)
        
        # Seasonal projection: direct linear mapping with frequency augmentation
        self.seasonal_proj = nn.Linear(input_len, output_len)
        
        # Trend projection: deeper MLP for slowly-varying components
        self.trend_proj = nn.Sequential(
            nn.Linear(input_len, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, output_len)
        )
        
        # Channel-wise residual mixing (1x1 conv = linear mixing across channels)
        if self.mix_ratio > 0:
            self.channel_mix = nn.Conv1d(n_features, n_features, kernel_size=1, bias=False)
            # Initialize near identity to preserve channel-wise behavior
            with torch.no_grad():
                self.channel_mix.weight.copy_(torch.eye(n_features).unsqueeze(-1) * 0.1)
        else:
            self.channel_mix = None
        
        # Output residual connection for stability
        self.output_rescale = nn.Parameter(torch.ones(1))
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Layer norm for feature dimension stability
        self.norm = nn.LayerNorm(n_features)
    
    def _decompose(self, x):
        # x: (batch, n_features, input_len)
        # Multi-scale trend extraction with adaptive weighting
        trends = []
        for decomp in self.decomp_layers:
            trend = decomp(x)  # (batch, n_features, input_len)
            trends.append(trend)
        
        # Softmax-weighted combination of scales
        weights = F.softmax(self.trend_weights, dim=0)  # (n_kernels,)
        trend = sum(w * t for w, t in zip(weights, trends))  # (batch, n_features, input_len)
        
        # Seasonal = residual
        seasonal = x - trend  # (batch, n_features, input_len)
        return trend, seasonal
    
    def _frequency_projection(self, x):
        # x: (batch, n_features, input_len)
        # Project onto learnable frequency bases
        # freq_bases: (n_freq_bases, input_len)
        # Compute coefficients via inner product
        coeffs = torch.matmul(x, self.freq_bases.t())  # (batch, n_features, n_freq_bases)
        coeffs = self.dropout(coeffs)
        # Project to output length
        out = self.freq_proj(coeffs)  # (batch, n_features, output_len)
        return out
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for channel-wise operations: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Multi-scale decomposition
        trend, seasonal = self._decompose(x)  # both (batch, n_features, input_len)
        
        # Trend path: deep projection
        trend_out = self.trend_proj(trend)  # (batch, n_features, output_len)
        
        # Seasonal path: direct + frequency-aware projection
        seasonal_direct = self.seasonal_proj(seasonal)  # (batch, n_features, output_len)
        seasonal_freq = self._frequency_projection(seasonal)  # (batch, n_features, output_len)
        seasonal_out = seasonal_direct + seasonal_freq  # (batch, n_features, output_len)
        
        # Combine with learned balance (implicit via separate projections)
        out = trend_out + seasonal_out  # (batch, n_features, output_len)
        
        # Residual channel mixing (controlled by mix_ratio)
        if self.channel_mix is not None and self.mix_ratio > 0:
            mixed = self.channel_mix(out)  # (batch, n_features, output_len)
            out = out + self.mix_ratio * mixed  # (batch, n_features, output_len)
        
        # Apply output scaling for training stability
        out = out * self.output_rescale  # (batch, n_features, output_len)
        
        # Transpose back: (batch, output_len, n_features)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Feature-wise normalization for stability
        out = self.norm(out)  # (batch, output_len, n_features)
        
        # Denormalize back to original scale
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
