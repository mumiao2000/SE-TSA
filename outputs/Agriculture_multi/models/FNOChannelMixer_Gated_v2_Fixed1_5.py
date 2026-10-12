import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNOChannelMixer_Gated_v2_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_modes': [4, 8, 16],
            'gate_dim': [32, 128, 512],
            'dropout': [0.0, 0.3],
            'n_layers': [1, 2, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_modes = self.config.get('n_modes', 8)
        self.gate_dim = self.config.get('gate_dim', 128)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_layers = self.config.get('n_layers', 2)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.input_len, self.hidden_dim)
        
        # Maximum modes for rfft with hidden_dim
        self.max_modes = self.hidden_dim // 2 + 1
        
        # FNO spectral layers - use full max_modes to avoid size mismatch
        self.spectral_layers = nn.ModuleList()
        self.time_mlp_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            # Spectral convolution: weights shaped (n_features, max_modes, 2)
            # We use max_modes to cover all possible FFT outputs, then mask to n_modes
            spectral_weight = nn.Parameter(torch.randn(self.n_features, self.max_modes, 2) * 0.02)
            self.spectral_layers.append(nn.ParameterDict({'weight': spectral_weight}))
            
            # Time-domain MLP for local interactions
            self.time_mlp_layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Dropout(self.dropout)
            ))
            
            self.norm_layers.append(nn.LayerNorm(self.hidden_dim))
        
        # Channel-wise gating network (inspired by ChannelAdaptiveMLP)
        self.gate_encoder = nn.Sequential(
            nn.Linear(self.n_features, self.gate_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.gate_dim, self.n_features),
            nn.Sigmoid()
        )
        
        # Output projection: hidden_dim -> output_len
        self.output_proj = nn.Linear(self.hidden_dim, self.output_len)
        
        # Residual frequency preservation: direct linear mapping
        self.freq_skip = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store for skip connection
        x_input = x  # (batch, input_len, n_features)
        
        # Transpose for feature-wise processing: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project to hidden dimension
        x = self.input_proj(x)  # (batch, n_features, hidden_dim)
        
        # FNO layers
        for i in range(self.n_layers):
            x_res = x  # (batch, n_features, hidden_dim)
            
            # Spectral path with corrected einsum
            # FFT: (batch, n_features, hidden_dim) -> complex (batch, n_features, hidden_dim//2+1)
            x_ft = torch.fft.rfft(x, dim=-1)  # (batch, n_features, actual_modes)
            actual_modes = x_ft.shape[-1]  # actual_modes = hidden_dim // 2 + 1
            
            # Get spectral weights and truncate/pad to actual modes
            weight = self.spectral_layers[i]['weight']  # (n_features, max_modes, 2)
            weight = weight[:, :actual_modes, :]  # (n_features, actual_modes, 2)
            
            # Apply mode masking: only keep first n_modes active (low-frequency)
            if actual_modes > self.n_modes:
                # Create mask: 1 for first n_modes, 0 for rest
                mask = torch.zeros(actual_modes, device=x.device)
                mask[:self.n_modes] = 1.0
                weight = weight * mask.view(1, -1, 1)  # (n_features, actual_modes, 2)
            
            weight_real = weight[..., 0]  # (n_features, actual_modes)
            weight_imag = weight[..., 1]  # (n_features, actual_modes)
            
            # Corrected einsum: 'bfn,fn->bfn' where f=n_features, n=actual_modes
            # Both x_ft and weight now have matching n dimension
            x_ft_real = torch.einsum('bfn,fn->bfn', x_ft.real, weight_real) - \
                        torch.einsum('bfn,fn->bfn', x_ft.imag, weight_imag)  # (batch, n_features, actual_modes)
            x_ft_imag = torch.einsum('bfn,fn->bfn', x_ft.real, weight_imag) + \
                        torch.einsum('bfn,fn->bfn', x_ft.imag, weight_real)  # (batch, n_features, actual_modes)
            x_ft = torch.complex(x_ft_real, x_ft_imag)  # (batch, n_features, actual_modes)
            
            # Inverse FFT
            x_spec = torch.fft.irfft(x_ft, n=self.hidden_dim, dim=-1)  # (batch, n_features, hidden_dim)
            
            # Time-domain MLP
            x_time = self.time_mlp_layers[i](x)  # (batch, n_features, hidden_dim)
            
            # Combine and normalize
            x = self.norm_layers[i](x_spec + x_time + x_res)  # (batch, n_features, hidden_dim)
        
        # Channel-wise gating
        # Compute per-channel statistics for gating
        channel_stats = x.mean(dim=-1)  # (batch, n_features)
        gate = self.gate_encoder(channel_stats)  # (batch, n_features)
        gate = gate.unsqueeze(-1)  # (batch, n_features, 1)
        
        # Apply gating
        x = x * gate  # (batch, n_features, hidden_dim)
        
        # Output projection to forecast horizon
        out = self.output_proj(x)  # (batch, n_features, output_len)
        
        # Transpose back: (batch, output_len, n_features)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Add residual frequency skip connection from raw input
        freq_skip = self.freq_skip(x_input.permute(0, 2, 1))  # (batch, n_features, output_len)
        freq_skip = freq_skip.permute(0, 2, 1)  # (batch, output_len, n_features)
        out = out + freq_skip * 0.5  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

