import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNOChannelMixing(BaseForecastModel):
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
        
        # Input projection per feature
        self.input_proj = nn.Linear(self.input_len, self.hidden_dim)
        
        # FNO blocks with channel-wise spectral processing
        self.fno_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.fno_layers.append(nn.ModuleDict({
                'freq_mlp': nn.Sequential(
                    nn.Linear(self.n_modes * 2, self.n_modes * 2),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.n_modes * 2, self.n_modes * 2)
                ),
                'channel_mix': nn.Sequential(
                    nn.Linear(self.n_features, self.n_features * 2),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.n_features * 2, self.n_features)
                ),
                'norm': nn.LayerNorm(self.hidden_dim)
            }))
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Permute to (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project to hidden dim
        x = self.input_proj(x)  # (batch, n_features, hidden_dim)
        
        # FNO layers
        for layer in self.fno_layers:
            residual = x  # (batch, n_features, hidden_dim)
            
            # FFT along time dimension
            x_fft = torch.fft.rfft(x, dim=-1)  # (batch, n_features, hidden_dim//2+1)
            
            # Keep only low modes
            n_freq = x_fft.shape[-1]
            n_modes_eff = min(self.n_modes, n_freq)
            x_fft_low = x_fft[..., :n_modes_eff]  # (batch, n_features, n_modes_eff)
            
            # Real and imaginary parts
            x_fft_real = torch.view_as_real(x_fft_low)  # (batch, n_features, n_modes_eff, 2)
            x_fft_flat = x_fft_real.reshape(x_fft_real.size(0), x_fft_real.size(1), -1)  # (batch, n_features, n_modes_eff*2)
            
            # Apply MLP in frequency domain
            x_fft_flat = layer['freq_mlp'](x_fft_flat)  # (batch, n_features, n_modes_eff*2)
            
            # Reshape back
            x_fft_new = x_fft_flat.reshape(x_fft_flat.size(0), x_fft_flat.size(1), n_modes_eff, 2)
            x_fft_new = torch.view_as_complex(x_fft_new.contiguous())  # (batch, n_features, n_modes_eff)
            
            # Pad back to original frequency size
            x_fft_padded = torch.zeros_like(x_fft)  # (batch, n_features, n_freq)
            x_fft_padded[..., :n_modes_eff] = x_fft_new  # (batch, n_features, n_freq)
            
            # Inverse FFT
            x = torch.fft.irfft(x_fft_padded, n=x.shape[-1], dim=-1)  # (batch, n_features, hidden_dim)
            
            # Channel mixing
            x = x.permute(0, 2, 1)  # (batch, hidden_dim, n_features)
            x = layer['channel_mix'](x)  # (batch, hidden_dim, n_features)
            x = x.permute(0, 2, 1)  # (batch, n_features, hidden_dim)
            
            # Residual and norm
            x = layer['norm'](x + residual)  # (batch, n_features, hidden_dim)
        
        # Output projection
        x = self.output_proj(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x
