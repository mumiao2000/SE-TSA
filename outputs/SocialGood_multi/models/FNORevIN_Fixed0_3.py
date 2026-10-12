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
            'n_modes': [16, 64, 256],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_modes = self.config.get('n_modes', 64)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # FNO layers: spectral convolutions
        self.fno_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.fno_layers.append(FNOBlock(self.hidden_dim, self.n_modes, self.dropout))
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        
        # Temporal interpolation for variable length
        self.temporal_fc = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden dim
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Apply FNO layers in frequency domain
        for layer in self.fno_layers:
            x = layer(x)  # (batch, input_len, hidden_dim)
        
        # Project to features and interpolate time
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        x = self.temporal_fc(x)  # (batch, hidden_dim, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        x = self.output_proj(x)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class FNOBlock(nn.Module):
    def __init__(self, hidden_dim, n_modes, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modes = n_modes
        
        # Spectral weights: complex weights stored as real and imag parts
        self.spectral_weight = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, n_modes, 2) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, input_len, hidden_dim)
        # FFT to frequency domain
        x_fft = torch.fft.rfft(x, dim=1)  # (batch, input_len//2+1, hidden_dim)
        
        # Truncate to n_modes
        n_fft = min(self.n_modes, x_fft.size(1))
        x_fft_trunc = x_fft[:, :n_fft, :]  # (batch, n_fft, hidden_dim)
        
        # Complex multiplication with learned weights
        # weight shape: (hidden_dim, hidden_dim, n_modes, 2) -> real and imag
        weight = self.spectral_weight[:, :, :n_fft, :]  # (hidden_dim, hidden_dim, n_fft, 2)
        
        # Prepare complex tensor
        x_fft_real = x_fft_trunc.real  # (batch, n_fft, hidden_dim)
        x_fft_imag = x_fft_trunc.imag  # (batch, n_fft, hidden_dim)
        
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # Using einsum for batch matrix multiplication
        out_real = torch.einsum('bti,iof->bto', x_fft_real, weight[..., 0]) - \
                   torch.einsum('bti,iof->bto', x_fft_imag, weight[..., 1])  # (batch, n_fft, hidden_dim)
        out_imag = torch.einsum('bti,iof->bto', x_fft_real, weight[..., 1]) + \
                   torch.einsum('bti,iof->bto', x_fft_imag, weight[..., 0])  # (batch, n_fft, hidden_dim)
        
        # Reconstruct complex tensor
        out_fft = torch.complex(out_real, out_imag)  # (batch, n_fft, hidden_dim)
        
        # Pad back to original FFT size
        if n_fft < x_fft.size(1):
            padding = torch.zeros(x_fft.size(0), x_fft.size(1) - n_fft, self.hidden_dim, 
                                 dtype=out_fft.dtype, device=out_fft.device)
            out_fft = torch.cat([out_fft, padding], dim=1)  # (batch, input_len//2+1, hidden_dim)
        
        # Inverse FFT
        x_out = torch.fft.irfft(out_fft, n=x.size(1), dim=1)  # (batch, input_len, hidden_dim)
        x_out = x_out + self.bias.unsqueeze(0).unsqueeze(0)  # (batch, input_len, hidden_dim)
        
        # Residual, norm, dropout
        x = x + self.dropout(self.norm(x_out))  # (batch, input_len, hidden_dim)
        return x
