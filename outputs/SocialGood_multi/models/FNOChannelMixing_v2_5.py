import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNOChannelMixing_v2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [256, 1024, 2048],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.2, 0.4],
            'n_modes_coarse': [16, 64, 256],
            'n_modes_fine': [4, 16, 64],
            'bottleneck_ratio': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 1024)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.2)
        self.n_modes_coarse = self.config.get('n_modes_coarse', 64)
        self.n_modes_fine = self.config.get('n_modes_fine', 16)
        self.bottleneck_ratio = self.config.get('bottleneck_ratio', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # FNO layers with multi-scale spectral processing
        self.fno_layers = nn.ModuleList()
        self.channel_mix_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.fno_layers.append(
                MultiScaleSpectralConv(
                    self.hidden_dim,
                    self.n_modes_coarse,
                    self.n_modes_fine,
                    self.n_features
                )
            )
            self.channel_mix_layers.append(
                BottleneckChannelMLP(
                    self.hidden_dim,
                    self.bottleneck_ratio,
                    self.dropout_rate
                )
            )
            self.norm_layers.append(nn.LayerNorm(self.hidden_dim))
            self.gate_layers.append(
                AdaptiveSpectralGate(self.hidden_dim)
            )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        
        # Direct residual connection from input to output horizon
        self.horizon_proj = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save for residual
        x_input = x  # (batch, input_len, n_features)
        
        # Input projection: (batch, input_len, hidden_dim)
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # FNO layers
        for i in range(self.n_layers):
            residual = x  # (batch, input_len, hidden_dim)
            
            # Multi-scale spectral convolution
            x = self.fno_layers[i](x)  # (batch, input_len, hidden_dim)
            
            # Adaptive spectral gating
            x = self.gate_layers[i](x)  # (batch, input_len, hidden_dim)
            
            # Channel mixing with bottleneck
            x = self.channel_mix_layers[i](x)  # (batch, input_len, hidden_dim)
            
            # Residual + norm
            x = self.norm_layers[i](x + residual)  # (batch, input_len, hidden_dim)
            x = self.dropout(x)  # (batch, input_len, hidden_dim)
        
        # Output projection: (batch, input_len, n_features)
        x = self.output_proj(x)  # (batch, input_len, n_features)
        
        # Direct residual from input: project input_len to output_len per feature
        # (batch, n_features, input_len) -> (batch, n_features, output_len) -> (batch, output_len, n_features)
        x_res = self.horizon_proj(x_input.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Truncate or interpolate x to output_len
        if x.shape[1] >= self.output_len:
            x = x[:, :self.output_len, :]  # (batch, output_len, n_features)
        else:
            # Repeat last values if needed (shouldn't happen with proper setup)
            repeat_factor = (self.output_len + x.shape[1] - 1) // x.shape[1]
            x = x.repeat(1, repeat_factor, 1)[:, :self.output_len, :]  # (batch, output_len, n_features)
        
        # Add residual
        x = x + x_res  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)


class MultiScaleSpectralConv(nn.Module):
    def __init__(self, hidden_dim, n_modes_coarse, n_modes_fine, n_features):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modes_coarse = n_modes_coarse
        self.n_modes_fine = n_modes_fine
        self.n_features = n_features
        
        # Coarse scale: low frequencies
        self.scale_coarse = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, n_modes_coarse, 2) * 0.02
        )
        # Fine scale: high frequencies
        self.scale_fine = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, n_modes_fine, 2) * 0.02
        )
        
        # Channel-specific phase shifts
        self.phase_coarse = nn.Parameter(torch.zeros(n_features, n_modes_coarse))
        self.phase_fine = nn.Parameter(torch.zeros(n_features, n_modes_fine))
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.shape
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=1)  # (batch, seq_len//2+1, hidden_dim)
        
        out_ft = torch.zeros_like(x_ft)  # (batch, seq_len//2+1, hidden_dim)
        
        # Coarse frequencies (low modes)
        n_c = min(self.n_modes_coarse, x_ft.shape[1])
        if n_c > 0:
            # Build complex weights
            w_c = torch.view_as_complex(self.scale_coarse[:, :, :n_c, :])  # (hidden_dim, hidden_dim, n_c)
            # Apply with channel-dependent phase
            x_c = x_ft[:, :n_c, :].permute(0, 2, 1)  # (batch, hidden_dim, n_c)
            # Phase modulation per feature group
            phase_c = torch.exp(1j * self.phase_coarse[:, :n_c])  # (n_features, n_c)
            # Simple approach: apply weights
            out_c = torch.einsum('bif,oif->bof', x_c, w_c[:, :, :n_c])  # (batch, hidden_dim, n_c)
            out_ft[:, :n_c, :] = out_c.permute(0, 2, 1)  # (batch, n_c, hidden_dim)
        
        # Fine frequencies (higher modes)
        start_idx = n_c
        n_f = min(self.n_modes_fine, x_ft.shape[1] - start_idx)
        if n_f > 0 and start_idx < x_ft.shape[1]:
            w_f = torch.view_as_complex(self.scale_fine[:, :, :n_f, :])  # (hidden_dim, hidden_dim, n_f)
            x_f = x_ft[:, start_idx:start_idx+n_f, :].permute(0, 2, 1)  # (batch, hidden_dim, n_f)
            out_f = torch.einsum('bif,oif->bof', x_f, w_f[:, :, :n_f])  # (batch, hidden_dim, n_f)
            out_ft[:, start_idx:start_idx+n_f, :] = out_f.permute(0, 2, 1)  # (batch, n_f, hidden_dim)
        
        # IFFT
        out = torch.fft.irfft(out_ft, n=seq_len, dim=1)  # (batch, seq_len, hidden_dim)
        return out  # (batch, seq_len, hidden_dim)


class AdaptiveSpectralGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate_gen = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Will be applied on seq dimension
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        # Use 1D conv for pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        # Global average pooling over sequence
        x_pooled = x.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
        x_pooled = self.pool(x_pooled).squeeze(-1)  # (batch, hidden_dim)
        gate = torch.sigmoid(x_pooled)  # (batch, hidden_dim)
        gate = gate.unsqueeze(1)  # (batch, 1, hidden_dim)
        return x * gate  # (batch, seq_len, hidden_dim)


class BottleneckChannelMLP(nn.Module):
    def __init__(self, hidden_dim, bottleneck_ratio, dropout_rate):
        super().__init__()
        self.bottleneck_dim = max(hidden_dim // bottleneck_ratio, 16)
        self.fc1 = nn.Linear(hidden_dim, self.bottleneck_dim)
        self.fc2 = nn.Linear(self.bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        x = self.fc1(x)  # (batch, seq_len, bottleneck_dim)
        x = self.activation(x)  # (batch, seq_len, bottleneck_dim)
        x = self.dropout(x)  # (batch, seq_len, bottleneck_dim)
        x = self.fc2(x)  # (batch, seq_len, hidden_dim)
        return x  # (batch, seq_len, hidden_dim)
