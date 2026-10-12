import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveFNO_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_groups': [2, 4, 8],
            'group_hidden': [32, 128, 512],
            'n_modes_base': [8, 32, 128],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_groups = self.config.get('n_groups', 4)
        self.group_hidden = self.config.get('group_hidden', 128)
        self.n_modes_base = self.config.get('n_modes_base', 32)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Ensure n_groups doesn't exceed features
        self.n_groups = min(self.n_groups, self.n_features)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Feature grouping: learned assignment
        self.group_logits = nn.Parameter(torch.randn(self.n_features, self.n_groups))
        
        # Per-group input projections
        self.group_projs = nn.ModuleList()
        for g in range(self.n_groups):
            self.group_projs.append(nn.Linear(self.n_features, self.group_hidden))
        
        # Per-group spectral convolutions with adaptive modes
        # Wrap in a custom Module so nn.ModuleList accepts it
        self.group_spectral = nn.ModuleList()
        for g in range(self.n_groups):
            n_modes = max(self.n_modes_base // (g + 1), 4)
            self.group_spectral.append(
                SpectralConv(self.group_hidden, n_modes)
            )
        
        # Cross-group attention
        self.cross_attn_q = nn.Linear(self.group_hidden, self.group_hidden)
        self.cross_attn_k = nn.Linear(self.group_hidden, self.group_hidden)
        self.cross_attn_v = nn.Linear(self.group_hidden, self.group_hidden)
        self.cross_attn_scale = self.group_hidden ** -0.5
        
        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Linear(self.group_hidden * self.n_groups, self.group_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Temporal decoder
        self.temporal_decoder = nn.Sequential(
            nn.Linear(self.input_len, self.output_len),
            nn.GELU()
        )
        
        self.output_proj = nn.Linear(self.group_hidden, self.n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Soft feature grouping
        group_assign = F.softmax(self.group_logits, dim=-1)  # (n_features, n_groups)
        
        # Process each group
        group_outputs = []
        for g in range(self.n_groups):
            # Project all features for this group
            x_proj = self.group_projs[g](x)  # (batch, input_len, group_hidden)
            
            # Apply spectral convolution
            x_fft = torch.fft.rfft(x_proj, dim=1)  # (batch, freq_len, group_hidden)
            n_modes = self.group_spectral[g].weight.size(2)
            n_fft = min(n_modes, x_fft.size(1))
            
            weight = self.group_spectral[g].weight[:, :, :n_fft, :]  # (group_hidden, group_hidden, n_fft, 2)
            
            x_fft_real = x_fft[:, :n_fft, :].real  # (batch, n_fft, group_hidden)
            x_fft_imag = x_fft[:, :n_fft, :].imag  # (batch, n_fft, group_hidden)
            
            out_real = torch.einsum('bti,iof->bto', x_fft_real, weight[..., 0]) - \
                       torch.einsum('bti,iof->bto', x_fft_imag, weight[..., 1])  # (batch, n_fft, group_hidden)
            out_imag = torch.einsum('bti,iof->bto', x_fft_real, weight[..., 1]) + \
                       torch.einsum('bti,iof->bto', x_fft_imag, weight[..., 0])  # (batch, n_fft, group_hidden)
            
            out_fft = torch.complex(out_real, out_imag)
            if n_fft < x_fft.size(1):
                padding = torch.zeros(x_fft.size(0), x_fft.size(1) - n_fft, self.group_hidden,
                                     dtype=out_fft.dtype, device=out_fft.device)
                out_fft = torch.cat([out_fft, padding], dim=1)
            
            x_spec = torch.fft.irfft(out_fft, n=x_proj.size(1), dim=1)  # (batch, input_len, group_hidden)
            group_outputs.append(x_spec)  # (batch, input_len, group_hidden)
        
        # Stack group outputs: (batch, input_len, n_groups, group_hidden)
        stacked = torch.stack(group_outputs, dim=2)  # (batch, input_len, n_groups, group_hidden)
        
        # Cross-group attention per timestep
        b, t, g, h = stacked.shape
        stacked_flat = stacked.view(b * t, g, h)  # (batch*input_len, n_groups, group_hidden)
        
        q = self.cross_attn_q(stacked_flat)  # (batch*input_len, n_groups, group_hidden)
        k = self.cross_attn_k(stacked_flat)  # (batch*input_len, n_groups, group_hidden)
        v = self.cross_attn_v(stacked_flat)  # (batch*input_len, n_groups, group_hidden)
        
        attn_scores = torch.einsum('bgh,bkh->bgk', q, k) * self.cross_attn_scale  # (batch*input_len, n_groups, n_groups)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch*input_len, n_groups, n_groups)
        attn_out = torch.einsum('bgk,bkh->bgh', attn_weights, v)  # (batch*input_len, n_groups, group_hidden)
        
        # Reshape back
        attn_out = attn_out.view(b, t, g, h)  # (batch, input_len, n_groups, group_hidden)
        
        # Fuse groups
        fused = attn_out.view(b, t, g * h)  # (batch, input_len, n_groups * group_hidden)
        fused = self.fusion(fused)  # (batch, input_len, group_hidden)
        
        # Temporal decoding
        fused = fused.permute(0, 2, 1)  # (batch, group_hidden, input_len)
        decoded = self.temporal_decoder(fused)  # (batch, group_hidden, output_len)
        decoded = decoded.permute(0, 2, 1)  # (batch, output_len, group_hidden)
        
        out = self.output_proj(decoded)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class SpectralConv(nn.Module):
    def __init__(self, hidden_dim, n_modes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim, n_modes, 2) * 0.02)
    
    def forward(self, x):
        # This module is used for its weight parameter; actual computation happens in parent
        return x

