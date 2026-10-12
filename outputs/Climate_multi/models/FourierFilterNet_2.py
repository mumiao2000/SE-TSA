import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierFilterNet(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_filters': [4, 16, 64],
            'filter_width': [3, 7, 15],
            'hidden_dim': [64, 256, 1024],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.n_filters = self.config.get('n_filters', 16)
        self.filter_width = self.config.get('filter_width', 7)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.dropout = self.config.get('dropout', 0.1)
        
        # FFT size (next power of 2 for efficiency)
        self.fft_size = 2 ** ((self.input_len + self.output_len - 1).bit_length())
        self.freq_bins = self.fft_size // 2 + 1
        
        # Learnable frequency filters (complex: real and imag parts)
        self.filter_real = nn.Parameter(torch.randn(self.n_features, self.n_filters, self.freq_bins) * 0.02)
        self.filter_imag = nn.Parameter(torch.randn(self.n_features, self.n_filters, self.freq_bins) * 0.02)
        
        # Smoothing constraint on filters
        self.smooth_conv = nn.Conv1d(
            self.n_filters, self.n_filters,
            kernel_size=self.filter_width,
            padding=self.filter_width // 2,
            groups=self.n_filters,
            bias=False
        )
        
        # MLP to combine filter outputs
        self.combiner = nn.Sequential(
            nn.Linear(self.n_filters, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Time-domain refinement
        self.time_refiner = nn.Sequential(
            nn.Linear(self.input_len + self.output_len, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_len)
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.norm(x)  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Pad to fft_size for both input and desired output
        x_padded = F.pad(x, (0, 0, 0, self.fft_size - self.input_len))  # (batch, fft_size, n_features)
        
        # FFT: operate on each feature independently
        x_perm = x_padded.permute(0, 2, 1)  # (batch, n_features, fft_size)
        x_fft = torch.fft.rfft(x_perm, dim=-1)  # (batch, n_features, freq_bins)
        
        # Apply learnable filters
        # filters: (n_features, n_filters, freq_bins)
        filters = torch.complex(self.filter_real, self.filter_imag)  # (n_features, n_filters, freq_bins)
        
        # Expand for batch: (batch, n_features, n_filters, freq_bins)
        filters = filters.unsqueeze(0).expand(batch_size, -1, -1, -1)
        x_fft_exp = x_fft.unsqueeze(2)  # (batch, n_features, 1, freq_bins)
        
        # Filtered outputs: multiply in complex domain
        filtered = x_fft_exp * filters  # (batch, n_features, n_filters, freq_bins)
        
        # Inverse FFT to get time-domain filtered signals
        filtered_flat = filtered.reshape(batch_size * self.n_features * self.n_filters, self.freq_bins)
        filtered_time = torch.fft.irfft(filtered_flat, n=self.fft_size, dim=-1)
        filtered_time = filtered_time.reshape(batch_size, self.n_features, self.n_filters, self.fft_size)
        
        # Apply smoothing
        filtered_time = filtered_time.permute(0, 1, 3, 2)  # (batch, n_features, fft_size, n_filters)
        filtered_flat = filtered_time.reshape(batch_size * self.n_features, self.fft_size, self.n_filters)
        filtered_flat = filtered_flat.permute(0, 2, 1)  # (batch*n_features, n_filters, fft_size)
        smoothed = self.smooth_conv(filtered_flat)  # (batch*n_features, n_filters, fft_size)
        smoothed = smoothed.permute(0, 2, 1)  # (batch*n_features, fft_size, n_filters)
        smoothed = smoothed.reshape(batch_size, self.n_features, self.fft_size, self.n_filters)
        
        # Extract output portion and combine filters
        out_portion = smoothed[:, :, :self.output_len, :]  # (batch, n_features, output_len, n_filters)
        
        # Combine filter outputs with MLP
        combined = self.combiner(out_portion).squeeze(-1)  # (batch, n_features, output_len)
        combined = combined.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Time-domain refinement with original input
        x_full = torch.cat([
            x.permute(0, 2, 1),  # (batch, n_features, input_len)
            combined.permute(0, 2, 1)  # (batch, n_features, output_len)
        ], dim=-1)  # (batch, n_features, input_len + output_len)
        
        refined = self.time_refiner(x_full)  # (batch, n_features, output_len)
        refined = refined.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Residual
        out = combined + refined  # (batch, output_len, n_features)
        
        return out
