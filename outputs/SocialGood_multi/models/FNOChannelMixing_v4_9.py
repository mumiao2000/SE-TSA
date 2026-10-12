import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNOChannelMixing_v4(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.3],
            'n_modes_coarse': [4, 16, 64],
            'n_modes_fine': [16, 64, 256],
            'bottleneck_ratio': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout = self.config.get('dropout', 0.3)
        self.n_modes_coarse = self.config.get('n_modes_coarse', 16)
        self.n_modes_fine = self.config.get('n_modes_fine', 64)
        self.bottleneck_ratio = self.config.get('bottleneck_ratio', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Multi-scale FNO layers
        self.fno_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.fno_layers.append(
                MultiScaleFNOLayer(
                    self.hidden_dim,
                    self.n_modes_coarse,
                    self.n_modes_fine,
                    self.dropout,
                    self.bottleneck_ratio
                )
            )
        
        # Output projection: project from hidden to n_features, then map from input_len to output_len
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_features),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        )
        
        # Temporal mapping: input_len -> output_len
        self.temporal_fc = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden dim
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Apply FNO layers
        for layer in self.fno_layers:
            x = x + layer(x)  # (batch, input_len, hidden_dim)
        
        # Project back to n_features
        x = self.output_proj(x)  # (batch, input_len, n_features)
        
        # Map temporal dimension: input_len -> output_len
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.temporal_fc(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class MultiScaleFNOLayer(nn.Module):
    def __init__(self, hidden_dim, n_modes_coarse, n_modes_fine, dropout, bottleneck_ratio):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modes_coarse = n_modes_coarse
        self.n_modes_fine = n_modes_fine
        
        # Coarse spectral path (aggressive truncation)
        self.spectral_coarse = SpectralConv1d(hidden_dim, n_modes_coarse)
        
        # Fine spectral path (less truncation)
        self.spectral_fine = SpectralConv1d(hidden_dim, n_modes_fine)
        
        # Learnable gate to combine coarse and fine
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Bottleneck MLP for channel mixing
        bottleneck_dim = max(hidden_dim // bottleneck_ratio, 16)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        batch, seq_len, hidden = x.shape  # (batch, seq_len, hidden_dim)
        
        # Layer norm
        x_norm = self.norm1(x)  # (batch, seq_len, hidden_dim)
        
        # Permute for spectral conv: (batch, hidden, seq_len)
        x_perm = x_norm.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
        
        # Coarse path
        x_coarse = self.spectral_coarse(x_perm)  # (batch, hidden_dim, seq_len)
        
        # Fine path
        x_fine = self.spectral_fine(x_perm)  # (batch, hidden_dim, seq_len)
        
        # Compute gate weights based on input
        gate_input = x_perm  # (batch, hidden_dim, seq_len)
        gate_weights = self.gate(gate_input)  # (batch, 2)
        
        # Apply gating: broadcast weights
        gate_coarse = gate_weights[:, 0:1].unsqueeze(-1)  # (batch, 1, 1)
        gate_fine = gate_weights[:, 1:2].unsqueeze(-1)  # (batch, 1, 1)
        
        x_spectral = gate_coarse * x_coarse + gate_fine * x_fine  # (batch, hidden_dim, seq_len)
        
        # Permute back
        x_spectral = x_spectral.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)
        
        # Residual connection 1
        x = x + x_spectral  # (batch, seq_len, hidden_dim)
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))  # (batch, seq_len, hidden_dim)
        
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, n_modes):
        super().__init__()
        self.in_channels = in_channels
        self.n_modes = n_modes
        
        # Complex weights for frequency domain
        scale = 1.0 / (in_channels * in_channels)
        self.weights_real = nn.Parameter(torch.randn(in_channels, in_channels, n_modes) * scale)
        self.weights_imag = nn.Parameter(torch.randn(in_channels, in_channels, n_modes) * scale)
    
    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        batch, channels, seq_len = x.shape  # (batch, in_channels, seq_len)
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, in_channels, seq_len//2 + 1)
        
        # Truncate to n_modes
        n_ft = min(self.n_modes, x_ft.shape[-1])
        x_ft_trunc = x_ft[:, :, :n_ft]  # (batch, in_channels, n_ft)
        
        # Apply complex weights
        weights = torch.view_as_complex(
            torch.stack([self.weights_real[:, :, :n_ft], self.weights_imag[:, :, :n_ft]], dim=-1)
        )  # (in_channels, in_channels, n_ft)
        
        # Batch matrix multiply in frequency domain
        # x_ft_trunc: (batch, in_channels, n_ft)
        # weights: (in_channels, in_channels, n_ft)
        out_ft = torch.einsum('bif,oif->bof', x_ft_trunc, weights)  # (batch, in_channels, n_ft)
        
        # Pad back to original frequency size
        out_ft_full = torch.zeros(batch, channels, x_ft.shape[-1], dtype=torch.complex64, device=x.device)
        out_ft_full[:, :, :n_ft] = out_ft  # (batch, in_channels, seq_len//2 + 1)
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft_full, n=seq_len, dim=-1)  # (batch, in_channels, seq_len)
        
        return x  # (batch, in_channels, seq_len)
