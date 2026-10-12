import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class LinearAdvancedFNO_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [32, 128, 512],
            'n_modes': [8, 32, 128],
            'n_refinement_layers': [1, 2, 4],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 128)
        self.n_modes = self.config.get('n_modes', 32)
        self.n_refinement_layers = self.config.get('n_refinement_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Advanced normalization: per-feature adaptive normalization
        self.feature_norm = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.hidden_dim // 4),
                nn.GELU(),
                nn.Linear(self.hidden_dim // 4, 2)  # scale and shift
            ) for _ in range(self.n_features)
        ])
        
        # Main linear predictor
        self.linear_predictor = nn.Linear(self.input_len, self.output_len)
        
        # FNO refinement network
        self.refinement_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        self.refinement_layers = nn.ModuleList()
        for _ in range(self.n_refinement_layers):
            self.refinement_layers.append(
                nn.ModuleDict({
                    'spectral': SpectralConv1d(self.hidden_dim, self.hidden_dim, self.n_modes),
                    'mlp': nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_dim * 4, self.hidden_dim)
                    ),
                    'norm1': nn.LayerNorm(self.hidden_dim),
                    'norm2': nn.LayerNorm(self.hidden_dim)
                })
            )
        
        # Output projection from refinement
        self.refinement_out = nn.Linear(self.hidden_dim, self.n_features)
        
        # Gating mechanism to blend linear and refined predictions
        self.gate = nn.Sequential(
            nn.Linear(self.n_features + self.n_features, self.n_features),
            nn.Sigmoid()
        )
    
    def adaptive_normalize(self, x):
        # x: (batch, seq_len, n_features)
        normalized = []
        for i in range(self.n_features):
            feat = x[:, :, i:i+1]  # (batch, seq_len, 1)
            params = self.feature_norm[i](feat.mean(dim=1, keepdim=True))  # (batch, 1, 2)
            scale = params[:, :, 0:1]  # (batch, 1, 1)
            shift = params[:, :, 1:2]  # (batch, 1, 1)
            norm_feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-5)
            norm_feat = norm_feat * (1 + scale) + shift
            normalized.append(norm_feat)
        return torch.cat(normalized, dim=-1)  # (batch, seq_len, n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # Adaptive normalization
        x_norm = self.adaptive_normalize(x)  # (batch, input_len, n_features)
        
        # Linear prediction
        x_perm = x_norm.permute(0, 2, 1)  # (batch, n_features, input_len)
        linear_out = self.linear_predictor(x_perm)  # (batch, n_features, output_len)
        linear_out = linear_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Compute residual for refinement
        linear_upsampled = F.interpolate(
            linear_out.permute(0, 2, 1), 
            size=self.input_len, 
            mode='linear', 
            align_corners=False
        ).permute(0, 2, 1)  # (batch, input_len, n_features)
        
        residual = x_norm - linear_upsampled  # (batch, input_len, n_features)
        
        # Refine residual with FNO
        r = self.refinement_proj(residual)  # (batch, input_len, hidden_dim)
        
        for layer in self.refinement_layers:
            # Spectral block
            r_spec = layer['spectral'](r)  # (batch, input_len, hidden_dim)
            r = r + layer['norm1'](r_spec)  # (batch, input_len, hidden_dim)
            
            # MLP block
            r = r + layer['norm2'](layer['mlp'](r))  # (batch, input_len, hidden_dim)
        
        # Project refinement to output
        r_out = self.refinement_out(r)  # (batch, input_len, n_features)
        
        # Downsample refinement to output length
        r_out = r_out.permute(0, 2, 1)  # (batch, n_features, input_len)
        r_out = F.interpolate(r_out, size=self.output_len, mode='linear', align_corners=False)
        r_out = r_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Adaptive gating
        gate_input = torch.cat([linear_out, r_out], dim=-1)  # (batch, output_len, 2*n_features)
        gate = self.gate(gate_input)  # (batch, output_len, n_features)
        
        out = gate * linear_out + (1 - gate) * r_out  # (batch, output_len, n_features)
        
        # Denormalize: reverse adaptive normalization approximately
        for i in range(self.n_features):
            feat = x[:, :, i:i+1]  # (batch, input_len, 1)
            mean = feat.mean(dim=1, keepdim=True)  # (batch, 1, 1)
            std = feat.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, 1)
            params = self.feature_norm[i](mean)  # (batch, 1, 2)
            scale = 1 + params[:, :, 0:1]  # (batch, 1, 1)
            shift = params[:, :, 1:2]  # (batch, 1, 1)
            out[:, :, i:i+1] = (out[:, :, i:i+1] - shift) / scale * std + mean  # (batch, output_len, 1)
        
        return out  # (batch, output_len, n_features)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        
        # Complex weights for spectral convolution: (in_channels, out_channels, n_modes, 2)
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, n_modes, 2) * 0.02)
    
    def forward(self, x):
        # x: (batch, seq_len, in_channels)
        batch_size, seq_len, in_channels = x.shape
        
        # FFT along time dimension
        x_fft = torch.fft.rfft(x, dim=1)  # (batch, freq_len, in_channels)
        n_fft = min(self.n_modes, x_fft.size(1))
        
        # Truncate weights to available frequencies
        weight = self.weight[:, :, :n_fft, :]  # (in_channels, out_channels, n_fft, 2)
        
        # Extract real and imaginary parts
        x_fft_real = x_fft[:, :n_fft, :].real  # (batch, n_fft, in_channels)
        x_fft_imag = x_fft[:, :n_fft, :].imag  # (batch, n_fft, in_channels)
        
        # Complex multiplication: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
        out_real = torch.einsum('bti,iof->bto', x_fft_real, weight[..., 0]) - \
                   torch.einsum('bti,iof->bto', x_fft_imag, weight[..., 1])  # (batch, n_fft, out_channels)
        out_imag = torch.einsum('bti,iof->bto', x_fft_real, weight[..., 1]) + \
                   torch.einsum('bti,iof->bto', x_fft_imag, weight[..., 0])  # (batch, n_fft, out_channels)
        
        # Reconstruct complex tensor
        out_fft = torch.complex(out_real, out_imag)  # (batch, n_fft, out_channels)
        
        # Pad with zeros if needed to match original frequency length
        if n_fft < x_fft.size(1):
            padding = torch.zeros(batch_size, x_fft.size(1) - n_fft, self.out_channels,
                                 dtype=out_fft.dtype, device=out_fft.device)  # (batch, pad_len, out_channels)
            out_fft = torch.cat([out_fft, padding], dim=1)  # (batch, freq_len, out_channels)
        
        # Inverse FFT back to time domain
        out = torch.fft.irfft(out_fft, n=seq_len, dim=1)  # (batch, seq_len, out_channels)
        
        return out  # (batch, seq_len, out_channels)
