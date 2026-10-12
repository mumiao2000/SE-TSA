import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNOChannelMixer_Fixed2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_modes': [4, 8, 16],
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.3],
            'gate_dim': [32, 128, 512],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_modes = self.config.get('n_modes', 8)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.gate_dim = self.config.get('gate_dim', 128)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Project input to hidden dimension per channel
        self.input_proj = nn.Linear(self.input_len, self.hidden_dim)
        
        # FNO spectral layers with multi-scale gating
        self.fno_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.fno_layers.append(
                FNOSpectralLayer(
                    self.hidden_dim,
                    self.n_features,
                    self.n_modes,
                    self.dropout
                )
            )
        
        # Channel-adaptive gating MLP
        self.channel_gate = nn.Sequential(
            nn.Linear(self.n_features, self.gate_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.gate_dim, self.n_features),
            nn.Sigmoid()
        )
        
        # Output projection with residual
        self.output_proj = nn.Linear(self.hidden_dim, self.output_len)
        
        # Residual connection for output
        self.residual_proj = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save for residual
        x_input = x  # (batch, input_len, n_features)
        
        # Project to hidden dim: (batch, n_features, hidden_dim)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.input_proj(x)  # (batch, n_features, hidden_dim)
        
        # Apply FNO layers
        for layer in self.fno_layers:
            x = x + layer(x)  # (batch, n_features, hidden_dim), residual inside layer
        
        # Output projection: (batch, n_features, output_len)
        out = self.output_proj(x)  # (batch, n_features, output_len)
        
        # Channel-adaptive gating
        # Compute channel importance from input statistics
        channel_stats = x_input.mean(dim=1)  # (batch, n_features)
        gate = self.channel_gate(channel_stats)  # (batch, n_features)
        gate = gate.unsqueeze(-1)  # (batch, n_features, 1)
        
        # Apply gating
        out = out * gate  # (batch, n_features, output_len)
        
        # Add residual from input
        residual = self.residual_proj(x_input.permute(0, 2, 1))  # (batch, n_features, output_len)
        out = out + residual  # (batch, n_features, output_len)
        
        # Permute back
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class FNOSpectralLayer(nn.Module):
    def __init__(self, hidden_dim, n_features, n_modes, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.n_modes = n_modes
        
        # Learnable spectral weights: (n_features, n_modes, 2) for real/imag
        self.scale = 1.0 / (n_features * n_modes)
        self.spectral_weight = nn.Parameter(
            torch.randn(n_features, n_modes, 2) * self.scale
        )
        
        # Multi-scale frequency gating: split low/mid/high frequencies
        self.low_gate = nn.Parameter(torch.ones(n_features, 1))
        self.mid_gate = nn.Parameter(torch.ones(n_features, 1))
        self.high_gate = nn.Parameter(torch.ones(n_features, 1))
        
        # Spatial MLP for channel mixing
        self.spatial_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, n_features, hidden_dim)
        batch_size = x.shape[0]
        
        # FFT along hidden_dim (temporal/frequency dimension)
        x_ft = torch.fft.rfft(x, dim=-1, norm='ortho')  # (batch, n_features, hidden_dim//2+1)
        
        # Get actual number of modes after FFT
        actual_modes = x_ft.shape[-1]
        modes_to_use = min(self.n_modes, actual_modes)
        
        # Truncate or pad spectral weights to match actual modes
        if modes_to_use < self.n_modes:
            weight = self.spectral_weight[:, :modes_to_use, :]  # (n_features, modes_to_use, 2)
        else:
            weight = self.spectral_weight  # (n_features, n_modes, 2)
        
        # Build complex weights
        weight_real = weight[:, :, 0]  # (n_features, modes_to_use)
        weight_imag = weight[:, :, 1]  # (n_features, modes_to_use)
        weight_complex = torch.complex(weight_real, weight_imag)  # (n_features, modes_to_use)
        
        # Apply spectral filtering with multi-scale gating
        # Split into frequency bands
        out_ft = torch.zeros_like(x_ft)  # (batch, n_features, actual_modes)
        
        # Low frequencies (first 1/3 of modes)
        low_end = max(1, modes_to_use // 3)
        # Mid frequencies
        mid_end = max(low_end + 1, 2 * modes_to_use // 3)
        
        # Apply to low band
        if low_end > 0:
            low_weight = weight_complex[:, :low_end]  # (n_features, low_end)
            low_ft = x_ft[:, :, :low_end]  # (batch, n_features, low_end)
            # einsum: batch, features, modes * features, modes -> batch, features, modes
            low_out = torch.einsum('bfn,fn->bfn', low_ft, low_weight)
            low_gate = torch.sigmoid(self.low_gate)  # (n_features, 1)
            out_ft[:, :, :low_end] = low_out * low_gate.unsqueeze(0)  # (batch, n_features, low_end)
        
        # Apply to mid band
        if mid_end > low_end:
            mid_weight = weight_complex[:, low_end:mid_end]  # (n_features, mid_end-low_end)
            mid_ft = x_ft[:, :, low_end:mid_end]  # (batch, n_features, mid_end-low_end)
            mid_out = torch.einsum('bfn,fn->bfn', mid_ft, mid_weight)
            mid_gate = torch.sigmoid(self.mid_gate)  # (n_features, 1)
            out_ft[:, :, low_end:mid_end] = mid_out * mid_gate.unsqueeze(0)
        
        # Apply to high band
        if modes_to_use > mid_end:
            high_weight = weight_complex[:, mid_end:modes_to_use]  # (n_features, modes_to_use-mid_end)
            high_ft = x_ft[:, :, mid_end:modes_to_use]  # (batch, n_features, modes_to_use-mid_end)
            high_out = torch.einsum('bfn,fn->bfn', high_ft, high_weight)
            high_gate = torch.sigmoid(self.high_gate)  # (n_features, 1)
            out_ft[:, :, mid_end:modes_to_use] = high_out * high_gate.unsqueeze(0)
        
        # Preserve remaining high frequencies without modification
        if actual_modes > modes_to_use:
            out_ft[:, :, modes_to_use:] = x_ft[:, :, modes_to_use:]
        
        # Inverse FFT
        x_spec = torch.fft.irfft(out_ft, n=self.hidden_dim, dim=-1, norm='ortho')  # (batch, n_features, hidden_dim)
        
        # Spatial MLP with residual
        x_out = self.norm(x_spec + x)  # (batch, n_features, hidden_dim)
        x_out = self.spatial_mlp(x_out)  # (batch, n_features, hidden_dim)
        x_out = self.dropout(x_out)
        
        return x_out  # (batch, n_features, hidden_dim)
