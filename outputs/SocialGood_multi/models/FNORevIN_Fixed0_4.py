import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNORevIN_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'n_modes': [16, 64, 256]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_modes = self.config.get('n_modes', 64)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Project to hidden dimension per feature (channel-independent)
        self.input_proj = nn.Linear(self.input_len, self.hidden_dim)
        
        # FNO layers: spectral convolutions in Fourier space
        self.fno_layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.fno_layers.append(FNOLayer(
                n_features=self.n_features,
                hidden_dim=self.hidden_dim,
                n_modes=self.n_modes,
                dropout=self.dropout
            ))
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for feature-wise processing: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project to hidden dimension
        x = self.input_proj(x)  # (batch, n_features, hidden_dim)
        
        # FNO layers
        for layer in self.fno_layers:
            x = layer(x)  # (batch, n_features, hidden_dim)
        
        # Project to output length
        x = self.output_proj(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class FNOLayer(nn.Module):
    def __init__(self, n_features, hidden_dim, n_modes, dropout):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_modes = n_modes
        
        self.spectral_weight = nn.Parameter(
            torch.randn(n_features, n_modes, 2) * 0.02
        )
        self.spatial_conv = nn.Conv1d(n_features, n_features, 1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, n_features, hidden_dim)
        residual = x  # (batch, n_features, hidden_dim)
        
        # FFT to frequency domain
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')  # (batch, n_features, hidden_dim//2+1)
        
        # Truncate to n_modes and apply learnable weights
        n_ft = min(self.n_modes, x_ft.shape[-1])
        x_ft_trunc = x_ft[:, :, :n_ft]  # (batch, n_features, n_modes)
        
        # Complex multiplication with learned weights
        weight_real = self.spectral_weight[:, :n_ft, 0]  # (n_features, n_modes)
        weight_imag = self.spectral_weight[:, :n_ft, 1]  # (n_features, n_modes)
        
        # Apply complex weights: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
        x_ft_real = x_ft_trunc.real * weight_real.unsqueeze(0) - x_ft_trunc.imag * weight_imag.unsqueeze(0)
        x_ft_imag = x_ft_trunc.real * weight_imag.unsqueeze(0) + x_ft_trunc.imag * weight_real.unsqueeze(0)
        x_ft_filtered = torch.complex(x_ft_real, x_ft_imag)  # (batch, n_features, n_modes)
        
        # Pad back and inverse FFT
        x_ft_full = torch.zeros_like(x_ft)  # (batch, n_features, hidden_dim//2+1)
        x_ft_full[:, :, :n_ft] = x_ft_filtered  # (batch, n_features, hidden_dim//2+1)
        x_spatial = torch.fft.irfft(x_ft_full, n=x.shape[-1], dim=-1, norm='ortho')  # (batch, n_features, hidden_dim)
        
        # Spatial convolution skip connection
        x_spatial = x_spatial + self.spatial_conv(residual)  # (batch, n_features, hidden_dim)
        x = self.norm(x_spatial)  # (batch, n_features, hidden_dim)
        x = F.gelu(x)  # (batch, n_features, hidden_dim)
        x = self.dropout(x)  # (batch, n_features, hidden_dim)
        x = x + residual  # (batch, n_features, hidden_dim)
        return x
