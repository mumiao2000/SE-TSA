import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNORevIN(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_modes': [8, 32, 128],
            'n_layers': [1, 4, 8],
            'hidden_dim': [64, 256, 512],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_modes = self.config.get('n_modes', 32)
        self.n_layers = self.config.get('n_layers', 4)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # FNO layers
        self.fno_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.fno_layers.append(SpectralConv1d(self.hidden_dim, self.hidden_dim, self.n_modes))
        
        # MLP after each FNO layer
        self.mlps = nn.ModuleList()
        for _ in range(self.n_layers):
            self.mlps.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            ))
        
        # Output projection: predict future sequence
        # We need to map from input_len to output_len in frequency domain
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
        # Learned temporal interpolation for output_len
        self.temporal_mixer = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden dim
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # FNO layers
        for fno, mlp in zip(self.fno_layers, self.mlps):
            # Spectral convolution
            x_fno = fno(x)  # (batch, input_len, hidden_dim)
            # MLP
            x_mlp = mlp(x)  # (batch, input_len, hidden_dim)
            # Skip connection
            x = x + x_fno + x_mlp  # (batch, input_len, hidden_dim)
        
        # Temporal interpolation to output_len
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        x = self.temporal_mixer(x)  # (batch, hidden_dim, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Project to feature space
        out = self.output_proj(x)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        
        # Complex weights for frequency domain
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, n_modes, 2)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, in_channels)
        batch_size, seq_len, _ = x.shape
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=1)  # (batch, seq_len//2+1, in_channels)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, x_ft.shape[1], self.out_channels, 
                            dtype=torch.cfloat, device=x.device)
        
        # Apply learned weights to first n_modes
        n_modes = min(self.n_modes, x_ft.shape[1])
        
        weights_complex = torch.view_as_complex(self.weights[:, :, :n_modes, :])
        # x_ft: (batch, freq, in), weights: (in, out, freq)
        # Need: (batch, freq, out)
        
        for i in range(n_modes):
            # x_ft[:, i, :]: (batch, in)
            # weights[:, :, i]: (in, out)
            out_ft[:, i, :] = torch.matmul(x_ft[:, i, :], weights_complex[:, :, i])
        
        # Inverse FFT
        out = torch.fft.irfft(out_ft, n=seq_len, dim=1)  # (batch, seq_len, out_channels)
        return out
