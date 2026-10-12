import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SpectralHybrid_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_scales': [2, 3, 4],
            'n_modes': [8, 32, 128],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_scales = self.config.get('n_scales', 3)
        self.n_modes = self.config.get('n_modes', 32)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Instance normalization per feature
        self.instance_norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Wavelet decomposition using simple averaging (Haar-like)
        self.scale_projs = nn.ModuleList()
        for s in range(self.n_scales):
            scale_len = max(self.input_len // (2 ** s), 1)
            self.scale_projs.append(
                nn.Sequential(
                    nn.Linear(self.n_features, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout)
                )
            )
        
        # Spectral processing per scale: wrap parameter in a Module
        class SpectralWeight(nn.Module):
            def __init__(self, hidden_dim, n_modes):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(hidden_dim, hidden_dim, n_modes, 2) * 0.02
                )
            def forward(self):
                return self.weight
        
        self.spectral_convs = nn.ModuleList()
        for s in range(self.n_scales):
            self.spectral_convs.append(
                SpectralWeight(self.hidden_dim, self.n_modes)
            )
        
        # Channel attention for cross-scale fusion
        self.channel_attn = nn.Sequential(
            nn.Linear(self.hidden_dim * self.n_scales, self.hidden_dim),
            nn.Sigmoid()
        )
        
        # Temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output generation
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
        # Direct residual connection
        self.residual_proj = nn.Linear(self.input_len, self.output_len)
    
    def haar_decompose(self, x, levels):
        # x: (batch, seq_len, hidden_dim)
        # Simple Haar wavelet: average and difference
        coeffs = []
        current = x
        for _ in range(levels):
            if current.size(1) < 2:
                break
            even = current[:, ::2, :]  # (batch, ceil(seq_len/2), hidden_dim)
            odd = current[:, 1::2, :]   # (batch, floor(seq_len/2), hidden_dim)
            # Pad if needed
            if odd.size(1) < even.size(1):
                odd = torch.cat([odd, even[:, -1:, :]], dim=1)
            approx = (even + odd) / 2  # (batch, ceil(seq_len/2), hidden_dim)
            detail = (even - odd) / 2  # (batch, ceil(seq_len/2), hidden_dim)
            coeffs.append((approx, detail))
            current = approx
        coeffs.append((current, None))  # Final approximation
        return coeffs
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # Instance normalization
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_norm = self.instance_norm(x_perm)  # (batch, n_features, input_len)
        x_norm = x_norm.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Store residual
        residual = x_norm.permute(0, 2, 1)  # (batch, n_features, input_len)
        residual = self.residual_proj(residual)  # (batch, n_features, output_len)
        residual = residual.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Project to hidden dim
        x_hid = self.scale_projs[0](x_norm)  # (batch, input_len, hidden_dim)
        
        # Multi-scale decomposition and spectral processing
        scale_outputs = []
        current = x_hid
        
        for s in range(min(self.n_scales, 3)):
            # Downsample for scale s
            if s > 0:
                current = F.avg_pool1d(
                    current.permute(0, 2, 1), 
                    kernel_size=2**s, 
                    stride=2**s
                ).permute(0, 2, 1)  # (batch, input_len//2^s, hidden_dim)
            
            # Spectral convolution
            x_fft = torch.fft.rfft(current, dim=1)  # (batch, freq_len, hidden_dim)
            n_fft = min(self.n_modes, x_fft.size(1))
            
            weight = self.spectral_convs[s].weight[:, :, :n_fft, :]  # (hidden_dim, hidden_dim, n_fft, 2)
            
            x_fft_real = x_fft[:, :n_fft, :].real  # (batch, n_fft, hidden_dim)
            x_fft_imag = x_fft[:, :n_fft, :].imag  # (batch, n_fft, hidden_dim)
            
            out_real = torch.einsum('bti,iof->bto', x_fft_real, weight[..., 0]) - \
                       torch.einsum('bti,iof->bto', x_fft_imag, weight[..., 1])  # (batch, n_fft, hidden_dim)
            out_imag = torch.einsum('bti,iof->bto', x_fft_real, weight[..., 1]) + \
                       torch.einsum('bti,iof->bto', x_fft_imag, weight[..., 0])  # (batch, n_fft, hidden_dim)
            
            out_fft = torch.complex(out_real, out_imag)
            if n_fft < x_fft.size(1):
                padding = torch.zeros(x_fft.size(0), x_fft.size(1) - n_fft, self.hidden_dim,
                                     dtype=out_fft.dtype, device=out_fft.device)
                out_fft = torch.cat([out_fft, padding], dim=1)
            
            x_spec = torch.fft.irfft(out_fft, n=current.size(1), dim=1)  # (batch, current_len, hidden_dim)
            
            # Global temporal pooling
            x_pooled = self.temporal_pool(x_spec.permute(0, 2, 1)).squeeze(-1)  # (batch, hidden_dim)
            scale_outputs.append(x_pooled)  # (batch, hidden_dim)
        
        # Concatenate scale representations
        multi_scale = torch.cat(scale_outputs, dim=-1)  # (batch, hidden_dim * n_scales)
        
        # Channel attention
        attn = self.channel_attn(multi_scale)  # (batch, hidden_dim)
        
        # Weighted combination (use first scale as base)
        fused = scale_outputs[0] * attn  # (batch, hidden_dim)
        for i in range(1, len(scale_outputs)):
            fused = fused + scale_outputs[i] * attn * (0.5 ** i)  # (batch, hidden_dim)
        
        # Generate output for each timestep
        # Expand to output length
        fused_expanded = fused.unsqueeze(1).expand(-1, self.output_len, -1)  # (batch, output_len, hidden_dim)
        
        out = self.decoder(fused_expanded)  # (batch, output_len, n_features)
        
        # Add residual
        out = out + residual  # (batch, output_len, n_features)
        
        # Denormalize: reverse instance norm
        out_perm = out.permute(0, 2, 1)  # (batch, n_features, output_len)
        # Use running statistics from input for denormalization
        mean = x_perm.mean(dim=-1, keepdim=True)  # (batch, n_features, 1)
        std = x_perm.std(dim=-1, keepdim=True) + 1e-5  # (batch, n_features, 1)
        out_denorm = out_perm * std + mean  # (batch, n_features, output_len)
        out = out_denorm.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        return out

