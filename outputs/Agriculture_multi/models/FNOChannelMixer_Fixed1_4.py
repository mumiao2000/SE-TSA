import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SpectralWeightModule(nn.Module):
    def __init__(self, n_features, actual_modes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, actual_modes, 2) * 0.02)

    def forward(self):
        return self.weight


class FNOChannelMixer_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_modes': [8, 32, 128],
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_modes = self.config.get('n_modes', 32)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Ensure n_modes doesn't exceed input_len//2 + 1
        self.actual_modes = min(self.n_modes, self.input_len // 2 + 1)
        
        # FFT-based spectral layers with channel-wise processing
        self.spectral_layers = nn.ModuleList()
        self.time_mlps = nn.ModuleList()
        
        for _ in range(self.n_layers):
            # Complex weights for frequency domain (real and imag separately)
            self.spectral_layers.append(SpectralWeightModule(self.n_features, self.actual_modes))
            
            # Time-domain MLP for channel mixing
            self.time_mlps.append(nn.Sequential(
                nn.Linear(self.input_len, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.input_len)
            ))
        
        # Channel gating for frequency importance
        self.freq_gate = nn.Sequential(
            nn.Linear(self.actual_modes * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.actual_modes)
        )
        
        # Output projection: from input_len to output_len
        self.output_proj = nn.Sequential(
            nn.Linear(self.input_len, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.output_len)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Process per feature in frequency domain
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        for i, (spectral_module, time_mlp) in enumerate(zip(self.spectral_layers, self.time_mlps)):
            # FFT
            x_fft = torch.fft.rfft(x, dim=-1)  # (batch, n_features, input_len//2+1)
            
            # Truncate to actual_modes
            x_fft_trunc = x_fft[:, :, :self.actual_modes]  # (batch, n_features, actual_modes)
            
            # Separate real and imag
            x_fft_real = x_fft_trunc.real  # (batch, n_features, actual_modes)
            x_fft_imag = x_fft_trunc.imag  # (batch, n_features, actual_modes)
            
            # Build complex weight
            spectral_weight = spectral_module()  # (n_features, actual_modes, 2)
            weight_real = spectral_weight[:, :, 0]  # (n_features, actual_modes)
            weight_imag = spectral_weight[:, :, 1]  # (n_features, actual_modes)
            
            # Apply complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            # Use einsum for batch-wise operation
            # Fixed: changed 'nf' to 'fn' to match weight shape (n_features, actual_modes)
            out_real = torch.einsum('bfn,fn->bfn', x_fft_real, weight_real) - \
                       torch.einsum('bfn,fn->bfn', x_fft_imag, weight_imag)  # (batch, n_features, actual_modes)
            out_imag = torch.einsum('bfn,fn->bfn', x_fft_real, weight_imag) + \
                       torch.einsum('bfn,fn->bfn', x_fft_imag, weight_real)  # (batch, n_features, actual_modes)
            
            # Channel-wise frequency gating
            gate_input = torch.cat([out_real, out_imag], dim=-1)  # (batch, n_features, actual_modes*2)
            gate = self.freq_gate(gate_input.mean(dim=1))  # (batch, actual_modes)
            gate = torch.sigmoid(gate).unsqueeze(1)  # (batch, 1, actual_modes)
            
            out_real = out_real * gate  # (batch, n_features, actual_modes)
            out_imag = out_imag * gate  # (batch, n_features, actual_modes)
            
            # Reconstruct complex tensor
            out_fft = torch.complex(out_real, out_imag)  # (batch, n_features, actual_modes)
            
            # Pad back to full FFT size
            full_fft = torch.zeros_like(x_fft)  # (batch, n_features, input_len//2+1)
            full_fft[:, :, :self.actual_modes] = out_fft  # (batch, n_features, input_len//2+1)
            
            # Inverse FFT
            x_spectral = torch.fft.irfft(full_fft, n=x.size(-1), dim=-1)  # (batch, n_features, input_len)
            
            # Time domain MLP with residual
            x_time = time_mlp(x)  # (batch, n_features, input_len)
            
            x = x + x_spectral + x_time  # (batch, n_features, input_len)
        
        # Output projection
        out = self.output_proj(x)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

