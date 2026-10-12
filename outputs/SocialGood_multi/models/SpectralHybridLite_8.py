import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SpectralHybridLite(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'n_modes_low': [4, 16, 64],
            'n_modes_high': [8, 32, 128],
            'bottleneck_ratio': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_p = self.config.get('dropout', 0.3)
        self.n_modes_low = self.config.get('n_modes_low', 16)
        self.n_modes_high = self.config.get('n_modes_high', 32)
        self.bottleneck_ratio = self.config.get('bottleneck_ratio', 4)
        
        self.bottleneck_dim = max(self.hidden_dim // self.bottleneck_ratio, 16)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to hidden dimension
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)  # (batch, input_len, hidden_dim)
        
        # FNO blocks with multi-scale spectral decomposition
        self.fno_blocks = nn.ModuleList()
        for i in range(self.n_layers):
            self.fno_blocks.append(
                MultiScaleFNOBlock(
                    self.hidden_dim,
                    self.n_modes_low,
                    self.n_modes_high,
                    self.dropout_p
                )
            )
        
        # Channel-wise adaptive gating for spectral branches
        self.spectral_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),  # (batch, hidden_dim)
            nn.Linear(self.hidden_dim, 2),  # (batch, 2)
            nn.Softmax(dim=-1)
        )
        
        # Output MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.bottleneck_dim),  # (batch, input_len, bottleneck_dim)
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.bottleneck_dim, self.n_features)  # (batch, input_len, n_features)
        )
        
        # Final projection from input_len to output_len
        self.temporal_proj = nn.Linear(self.input_len, self.output_len)  # (batch, output_len, n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden dimension
        h = self.input_proj(x)  # (batch, input_len, hidden_dim)
        h = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        
        # Apply FNO blocks with residual connections
        for block in self.fno_blocks:
            h = h + block(h)  # (batch, hidden_dim, input_len)
        
        # Compute adaptive spectral gate weights
        gate_weights = self.spectral_gate(h)  # (batch, 2)
        
        # Apply gating (integrated within each block, but we can modulate here too)
        # Actually, let's apply the gating to the final representation
        h = h.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
        
        # Decode to feature space
        out = self.decoder(h)  # (batch, input_len, n_features)
        
        # Project temporal dimension
        out = out.permute(0, 2, 1)  # (batch, n_features, input_len)
        out = self.temporal_proj(out)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class SpectralWeightModule(nn.Module):
    """Proper nn.Module wrapper for spectral convolution weights."""
    def __init__(self, in_dim, out_dim, n_modes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim, n_modes, 2) * 0.02)
        
    def forward(self, x_ft, n_modes):
        # x_ft: (batch, in_dim, n_freq, 2)
        # Extract modes and apply complex multiplication
        x_ft_modes = x_ft[:, :, :n_modes, :]  # (batch, in_dim, n_modes, 2)
        # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        real = x_ft_modes[:, :, :, 0]  # (batch, in_dim, n_modes)
        imag = x_ft_modes[:, :, :, 1]  # (batch, in_dim, n_modes)
        w_real = self.weight[:, :, :n_modes, 0]  # (in_dim, out_dim, n_modes)
        w_imag = self.weight[:, :, :n_modes, 1]  # (in_dim, out_dim, n_modes)
        
        out_real = torch.einsum('bim,iof->bom', real, w_real) - torch.einsum('bim,iof->bom', imag, w_imag)
        out_imag = torch.einsum('bim,iof->bom', real, w_imag) + torch.einsum('bim,iof->bom', imag, w_real)
        
        out = torch.stack([out_real, out_imag], dim=-1)  # (batch, out_dim, n_modes, 2)
        return out


class MultiScaleFNOBlock(nn.Module):
    """FNO block with two-branch spectral decomposition: low-frequency (trend) and high-frequency (detail)."""
    def __init__(self, hidden_dim, n_modes_low, n_modes_high, dropout_p):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modes_low = n_modes_low
        self.n_modes_high = n_modes_high
        self.max_modes = max(n_modes_low, n_modes_high)
        
        # Low-frequency (trend) branch
        self.spectral_low = SpectralWeightModule(hidden_dim, hidden_dim, self.max_modes)
        
        # High-frequency (detail) branch  
        self.spectral_high = SpectralWeightModule(hidden_dim, hidden_dim, self.max_modes)
        
        # Spatial convolution for local mixing
        self.spatial_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        
        # Layer norm and dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        # MLP for channel mixing
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout_p)
        )
        
    def forward(self, x):
        # x: (batch, hidden_dim, seq_len)
        batch_size, _, seq_len = x.shape
        
        # Store residual
        residual = x  # (batch, hidden_dim, seq_len)
        
        # FFT to frequency domain
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, hidden_dim, n_freq)
        x_ft = torch.view_as_real(x_ft)  # (batch, hidden_dim, n_freq, 2)
        n_freq = x_ft.shape[2]
        
        # Low-frequency branch (trend)
        n_modes_low_eff = min(self.n_modes_low, n_freq)
        x_ft_low = self.spectral_low(x_ft, n_modes_low_eff)  # (batch, hidden_dim, n_modes_low_eff, 2)
        # Pad back to original frequency size
        x_ft_low_padded = torch.zeros(batch_size, self.hidden_dim, n_freq, 2, device=x.device, dtype=x_ft.dtype)
        x_ft_low_padded[:, :, :n_modes_low_eff, :] = x_ft_low
        
        # High-frequency branch (detail) - operate on remaining frequencies
        n_modes_high_eff = min(self.n_modes_high, n_freq)
        # Shift to capture higher frequencies
        x_ft_shifted = torch.roll(x_ft, shifts=-n_modes_low_eff//2, dims=2)
        x_ft_high = self.spectral_high(x_ft_shifted, n_modes_high_eff)  # (batch, hidden_dim, n_modes_high_eff, 2)
        x_ft_high_padded = torch.zeros(batch_size, self.hidden_dim, n_freq, 2, device=x.device, dtype=x_ft.dtype)
        # Place high frequencies in appropriate location
        start_idx = n_modes_low_eff
        end_idx = min(start_idx + n_modes_high_eff, n_freq)
        actual_high_modes = end_idx - start_idx
        if actual_high_modes > 0:
            x_ft_high_padded[:, :, start_idx:end_idx, :] = x_ft_high[:, :, :actual_high_modes, :]
        
        # Combine branches with learnable mixing
        x_ft_out = x_ft_low_padded + x_ft_high_padded  # (batch, hidden_dim, n_freq, 2)
        
        # Inverse FFT back to time domain
        x_ft_complex = torch.view_as_complex(x_ft_out.contiguous())  # (batch, hidden_dim, n_freq)
        x_time = torch.fft.irfft(x_ft_complex, n=seq_len, dim=-1)  # (batch, hidden_dim, seq_len)
        
        # Add spatial convolution path
        x_spatial = self.spatial_conv(x)  # (batch, hidden_dim, seq_len)
        
        # Combine spectral and spatial
        x_mixed = x_time + x_spatial  # (batch, hidden_dim, seq_len)
        
        # First norm + MLP
        x_mixed = x_mixed.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)
        x_mixed = self.norm1(x_mixed)  # (batch, seq_len, hidden_dim)
        x_mixed = self.mlp(x_mixed)  # (batch, seq_len, hidden_dim)
        x_mixed = x_mixed.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
        
        # Residual connection
        out = residual + self.dropout(x_mixed)  # (batch, hidden_dim, seq_len)
        
        # Second norm
        out = out.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)
        out = self.norm2(out)  # (batch, seq_len, hidden_dim)
        out = out.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
        
        return out
