import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNOAdaptiveSpectral_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'modes': [8, 32, 128],
            'width': [32, 128, 512],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.modes = self.config.get('modes', 32)
        self.width = self.config.get('width', 128)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Padded length for FFT (power of 2 or at least >= input_len + output_len)
        self.fft_len = 1
        while self.fft_len < self.input_len + self.output_len:
            self.fft_len *= 2
        
        # Actual available frequency modes after rfft
        self.max_modes = self.fft_len // 2 + 1
        
        # Input lifting
        self.lift = nn.Linear(self.n_features, self.width)
        
        # Spectral convolution layers
        self.spectral_layers = nn.ParameterList()
        self.spatial_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(self.n_layers):
            # Complex weights for spectral convolution (use max_modes to ensure enough capacity)
            effective_modes = min(self.modes, self.max_modes)
            scale = 1.0 / (self.width * self.width)
            self.spectral_layers.append(nn.Parameter(
                scale * torch.randn(self.width, self.width, effective_modes, 2)
            ))
            
            # Spatial MLP for mixing
            self.spatial_layers.append(nn.Sequential(
                nn.Linear(self.width, self.width * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.width * 2, self.width)
            ))
            self.norms.append(nn.LayerNorm(self.width))
        
        # Adaptive spectral gate
        self.spectral_gate = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.Sigmoid()
        )
        
        # Output projection
        self.project = nn.Sequential(
            nn.Linear(self.width, self.width // 2),
            nn.GELU(),
            nn.Linear(self.width // 2, self.n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Pad to fft_len
        batch_size = x.shape[0]
        x_padded = torch.zeros(batch_size, self.fft_len, self.n_features, device=x.device)
        x_padded[:, :self.input_len, :] = x  # (batch, fft_len, n_features)
        
        # Lift to higher dimension
        x_padded = self.lift(x_padded)  # (batch, fft_len, width)
        
        # Apply spectral layers
        for i, (w, mlp, norm) in enumerate(zip(self.spectral_layers, self.spatial_layers, self.norms)):
            residual = x_padded  # (batch, fft_len, width)
            
            # FFT
            x_fft = torch.fft.rfft(x_padded, dim=1)  # (batch, fft_len//2+1, width)
            
            # Determine effective modes for this forward pass
            effective_modes = min(self.modes, x_fft.shape[1])
            
            # Truncate to modes and apply complex weights
            x_fft_trunc = x_fft[:, :effective_modes, :]  # (batch, effective_modes, width)
            
            # Build complex weight tensor (no inplace operations)
            w_complex = torch.view_as_complex(w[:, :, :effective_modes, :].contiguous())  # (width, width, effective_modes)
            
            # Apply spectral convolution: for each mode, matrix multiply
            # x_fft_trunc: (batch, effective_modes, width), need (batch, effective_modes, width)
            x_fft_out = torch.zeros_like(x_fft_trunc)
            for m in range(effective_modes):
                x_fft_out[:, m, :] = x_fft_trunc[:, m, :] @ w_complex[:, :, m].T  # (batch, width)
            
            # Pad back and inverse FFT
            x_fft_full = torch.zeros_like(x_fft)
            x_fft_full[:, :effective_modes, :] = x_fft_out  # (batch, fft_len//2+1, width)
            x_spatial = torch.fft.irfft(x_fft_full, n=self.fft_len, dim=1)  # (batch, fft_len, width)
            
            # Spectral gate
            gate = self.spectral_gate(x_spatial)  # (batch, fft_len, width)
            x_spatial = x_spatial * gate  # (batch, fft_len, width)
            
            # Spatial MLP with residual
            x_mlp = mlp(x_spatial)  # (batch, fft_len, width)
            x_padded = norm(x_spatial + x_mlp + residual)  # (batch, fft_len, width)
        
        # Extract output_len from the end (causal: predict future)
        x_out = x_padded[:, self.input_len:self.input_len + self.output_len, :]  # (batch, output_len, width)
        
        # Project to n_features
        x_out = self.project(x_out)  # (batch, output_len, n_features)
        x_out = self.revin(x_out, mode='denorm')  # (batch, output_len, n_features)
        return x_out
