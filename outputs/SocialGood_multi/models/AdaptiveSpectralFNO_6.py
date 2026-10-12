import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveSpectralFNO(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'n_modes_coarse': [4, 16, 64],
            'n_modes_fine': [16, 64, 256],
            'bottleneck_ratio': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.3)
        self.n_modes_coarse = self.config.get('n_modes_coarse', 16)
        self.n_modes_fine = self.config.get('n_modes_fine', 64)
        self.bottleneck_ratio = self.config.get('bottleneck_ratio', 4)
        
        self.bottleneck_dim = max(self.hidden_dim // self.bottleneck_ratio, 16)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to hidden dimension
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Positional encoding for temporal information
        self.pos_enc = nn.Parameter(torch.randn(1, self.input_len, self.hidden_dim) * 0.02)
        
        # FNO layers with adaptive multi-scale spectral gating
        self.fno_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.fno_layers.append(
                AdaptiveSpectralBlock(
                    self.hidden_dim, 
                    self.n_modes_coarse,
                    self.n_modes_fine,
                    self.dropout_rate
                )
            )
            self.norm_layers.append(nn.LayerNorm(self.hidden_dim))
            self.dropout_layers.append(nn.Dropout(self.dropout_rate))
        
        # Output projection: map from input_len to output_len
        self.time_proj = nn.Linear(self.input_len, self.output_len)
        
        # Feature projection back to n_features
        self.feature_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.bottleneck_dim, self.n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden dimension
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        x = x + self.pos_enc[:, :self.input_len, :]  # (batch, input_len, hidden_dim)
        
        # Apply FNO layers with adaptive spectral gating
        for fno, norm, drop in zip(self.fno_layers, self.norm_layers, self.dropout_layers):
            residual = x  # (batch, input_len, hidden_dim)
            x = fno(x)  # (batch, input_len, hidden_dim)
            x = norm(x + residual)  # (batch, input_len, hidden_dim)
            x = drop(x)  # (batch, input_len, hidden_dim)
        
        # Project time dimension: input_len -> output_len
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        x = self.time_proj(x)  # (batch, hidden_dim, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Project back to feature dimension
        x = self.feature_proj(x)  # (batch, output_len, n_features)
        
        # Denormalize
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)


class AdaptiveSpectralBlock(nn.Module):
    def __init__(self, hidden_dim, n_modes_coarse, n_modes_fine, dropout_rate):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modes_coarse = n_modes_coarse
        self.n_modes_fine = n_modes_fine
        self.max_modes = max(n_modes_coarse, n_modes_fine)
        
        # Coarse and fine spectral weights (complex)
        self.scale_coarse = 1.0 / (hidden_dim * hidden_dim)
        self.scale_fine = 1.0 / (hidden_dim * hidden_dim)
        
        self.weight_coarse_real = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, n_modes_coarse) * self.scale_coarse
        )
        self.weight_coarse_imag = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, n_modes_coarse) * self.scale_coarse
        )
        self.weight_fine_real = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, n_modes_fine) * self.scale_fine
        )
        self.weight_fine_imag = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, n_modes_fine) * self.scale_fine
        )
        
        # Adaptive gating: learn to weight coarse vs fine per sample
        self.gate_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 2),
            nn.Softmax(dim=-1)
        )
        
        # Spatial MLP for local mixing
        self.spatial_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape  # (batch, seq_len, hidden_dim)
        
        # Compute gating weights per sample
        x_perm = x.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
        gate_input = x_perm  # (batch, hidden_dim, seq_len)
        gate_weights = self.gate_proj(gate_input)  # (batch, 2)
        gate_coarse = gate_weights[:, 0:1]  # (batch, 1)
        gate_fine = gate_weights[:, 1:2]  # (batch, 1)
        
        # FFT for spectral domain
        x_ft = torch.fft.rfft(x, dim=1)  # (batch, seq_len//2+1, hidden_dim)
        n_ft = x_ft.shape[1]  # int
        
        # Initialize output spectrum
        out_ft = torch.zeros_like(x_ft)  # (batch, seq_len//2+1, hidden_dim)
        
        # Coarse modes (low frequencies)
        n_coarse = min(self.n_modes_coarse, n_ft)  # int
        if n_coarse > 0:
            x_coarse = x_ft[:, :n_coarse, :]  # (batch, n_coarse, hidden_dim)
            # Apply complex spectral weight: (hidden_dim, hidden_dim, n_coarse)
            w_coarse = torch.view_as_complex(
                torch.stack([self.weight_coarse_real[:, :, :n_coarse], 
                            self.weight_coarse_imag[:, :, :n_coarse]], dim=-1)
            )  # (hidden_dim, hidden_dim, n_coarse)
            # einsum: b t i, i j t -> b t j
            x_coarse_out = torch.einsum('bti,ijt->btj', x_coarse, w_coarse)  # (batch, n_coarse, hidden_dim)
            out_ft[:, :n_coarse, :] = gate_coarse.view(batch_size, 1, 1) * x_coarse_out  # (batch, n_coarse, hidden_dim)
        
        # Fine modes (high frequencies, use fewer modes for regularization)
        n_fine = min(self.n_modes_fine, n_ft - n_coarse)  # int
        if n_fine > 0 and n_coarse < n_ft:
            x_fine = x_ft[:, n_coarse:n_coarse + n_fine, :]  # (batch, n_fine, hidden_dim)
            w_fine = torch.view_as_complex(
                torch.stack([self.weight_fine_real[:, :, :n_fine], 
                            self.weight_fine_imag[:, :, :n_fine]], dim=-1)
            )  # (hidden_dim, hidden_dim, n_fine)
            x_fine_out = torch.einsum('bti,ijt->btj', x_fine, w_fine)  # (batch, n_fine, hidden_dim)
            out_ft[:, n_coarse:n_coarse + n_fine, :] = gate_fine.view(batch_size, 1, 1) * x_fine_out  # (batch, n_fine, hidden_dim)
        
        # Inverse FFT back to time domain
        x_spec = torch.fft.irfft(out_ft, n=seq_len, dim=1)  # (batch, seq_len, hidden_dim)
        
        # Spatial MLP for local feature mixing
        x_spatial = self.spatial_mlp(x_spec)  # (batch, seq_len, hidden_dim)
        
        return x_spatial  # (batch, seq_len, hidden_dim)
