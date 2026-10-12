import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAdaptiveWavelet_Fixed2_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'wavelet_depth': [2, 4, 8],
            'hidden_dim': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3],
            'bottleneck_ratio': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.wavelet_depth = self.config.get('wavelet_depth', 4)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout = self.config.get('dropout', 0.1)
        self.bottleneck_ratio = self.config.get('bottleneck_ratio', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Haar wavelet filters (fixed, not learnable for stability)
        self.register_buffer('low_pass', torch.tensor([0.5, 0.5]))
        self.register_buffer('high_pass', torch.tensor([0.5, -0.5]))
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Channel-adaptive threshold parameters - learnable per channel, per scale
        # Shape: (wavelet_depth, 1, 1, hidden_dim) for proper broadcasting
        # NOTE: Operates on hidden_dim space after input projection, not n_features
        self.thresholds = nn.Parameter(torch.ones(self.wavelet_depth, 1, 1, self.hidden_dim) * 0.1)
        self.threshold_gates = nn.Parameter(torch.ones(self.wavelet_depth, 1, 1, self.hidden_dim) * 0.5)
        
        # Cross-scale fusion with bottleneck
        self.bottleneck_dim = max(self.hidden_dim // self.bottleneck_ratio, 16)
        
        # Progressive fusion layers: fuse scale i with accumulated representation
        self.fusion_down = nn.ModuleList()
        self.fusion_up = nn.ModuleList()
        self.fusion_gates = nn.ModuleList()
        for i in range(self.wavelet_depth):
            # Down-project both inputs
            self.fusion_down.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.bottleneck_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ))
            # Up-project fused representation
            self.fusion_up.append(nn.Sequential(
                nn.Linear(self.bottleneck_dim, self.hidden_dim),
                nn.Dropout(self.dropout)
            ))
            # Gating for cross-scale combination
            self.fusion_gates.append(nn.Linear(self.bottleneck_dim * 2, self.bottleneck_dim))
        
        # Periodic-aware position encoding
        # Dominant period from dataset stats: ~370 days
        self.period = 370.0
        self.period_embed_dim = self.hidden_dim
        # Learnable frequency modulation
        self.freq_mod = nn.Parameter(torch.ones(1))
        
        # Periodic-aware temporal attention
        self.period_norm = nn.LayerNorm(self.hidden_dim)
        self.period_attention = nn.MultiheadAttention(
            self.hidden_dim, self.n_heads, dropout=self.dropout, batch_first=True
        )
        
        # Local temporal convolution for fine-grained patterns
        self.local_conv = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=self.hidden_dim)
        self.local_norm = nn.LayerNorm(self.hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        
        # Direct residual connection from input to output for stability
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def get_periodic_encoding(self, seq_len, device):
        # Sinusoidal encoding modulated by learned frequency
        position = torch.arange(seq_len, device=device).unsqueeze(1).float()  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, self.period_embed_dim, 2, device=device).float() * 
                           (-torch.log(torch.tensor(10000.0)) / self.period_embed_dim))
        # Modulate by learned frequency
        effective_period = self.period / (1.0 + F.softplus(self.freq_mod))
        angle = position * 2.0 * 3.141592653589793 / effective_period
        
        pe = torch.zeros(seq_len, self.period_embed_dim, device=device)
        pe[:, 0::2] = torch.sin(angle * div_term[:self.period_embed_dim // 2])
        if self.period_embed_dim % 2 == 1:
            pe[:, 1::2] = torch.sin(angle * div_term[:self.period_embed_dim // 2])
        else:
            pe[:, 1::2] = torch.cos(angle * div_term[:self.period_embed_dim // 2])
        return pe  # (seq_len, hidden_dim)
    
    def haar_decompose(self, x):
        # x: (batch, seq_len, features)
        B, L, n_feat = x.shape  # (batch, seq_len, features)
        if L % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))  # (batch, seq_len+1, features)
            L = L + 1  # scalar
        x = x.reshape(B, L // 2, 2, n_feat)  # (batch, L/2, 2, features)
        approx = x[:, :, 0, :] * self.low_pass[0] + x[:, :, 1, :] * self.low_pass[1]  # (batch, L/2, features)
        detail = x[:, :, 0, :] * self.high_pass[0] + x[:, :, 1, :] * self.high_pass[1]  # (batch, L/2, features)
        return approx, detail
    
    def channel_adaptive_threshold(self, detail, scale_idx):
        # detail: (batch, seq_len, hidden_dim)
        # thresholds[scale_idx]: (1, 1, hidden_dim)
        # threshold_gates[scale_idx]: (1, 1, hidden_dim)
        
        # Soft thresholding: sign(x) * max(|x| - threshold, 0)
        threshold = F.softplus(self.thresholds[scale_idx])  # (1, 1, hidden_dim)
        gate = torch.sigmoid(self.threshold_gates[scale_idx])  # (1, 1, hidden_dim)
        
        # Adaptive: threshold proportional to channel std
        channel_std = detail.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, hidden_dim)
        adaptive_thresh = threshold * channel_std  # (batch, 1, hidden_dim)
        
        sign = torch.sign(detail)
        magnitude = torch.abs(detail) - adaptive_thresh
        magnitude = F.relu(magnitude)
        thresholded = sign * magnitude  # (batch, seq_len, hidden_dim)
        
        # Gated combination of original and thresholded
        return gate * thresholded + (1 - gate) * detail  # (batch, seq_len, hidden_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        B, L_in, n_feat = x.shape  # (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        x_input = x.clone()  # Save for residual
        
        # Input projection
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Multi-scale wavelet decomposition with channel-adaptive thresholding
        scales = []
        current = x
        for i in range(self.wavelet_depth):
            if current.shape[1] < 2:
                break
            approx, detail = self.haar_decompose(current)  # both (batch, seq_len/2, hidden_dim)
            # Apply channel-adaptive thresholding to detail coefficients
            # For hidden_dim features, treat each dimension as a channel
            detail = self.channel_adaptive_threshold(detail, i)  # (batch, seq_len/2, hidden_dim)
            scales.append((approx, detail))
            current = approx
        
        # Hierarchical cross-scale fusion (from coarsest to finest)
        # Start with coarsest approximation
        if len(scales) == 0:
            fused = x  # (batch, input_len, hidden_dim)
        else:
            # Initialize with coarsest scale (last in list)
            fused = scales[-1][0]  # (batch, coarsest_len, hidden_dim)
            
            # Progressively upsample and fuse with finer scales
            for i in range(len(scales) - 1, -1, -1):
                approx, detail = scales[i]
                
                # Upsample fused to match current scale if needed
                if fused.shape[1] != approx.shape[1]:
                    fused = F.interpolate(
                        fused.transpose(1, 2),  # (batch, hidden_dim, fused_len)
                        size=approx.shape[1],
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)  # (batch, approx_len, hidden_dim)
                
                # Combine approximation and detail
                combined = approx + detail  # (batch, approx_len, hidden_dim)
                
                # Bottleneck fusion
                fused_b = self.fusion_down[i](fused)  # (batch, approx_len, bottleneck_dim)
                combined_b = self.fusion_down[i](combined)  # (batch, approx_len, bottleneck_dim)
                
                # Gated fusion
                gate_input = torch.cat([fused_b, combined_b], dim=-1)  # (batch, approx_len, 2*bottleneck_dim)
                # Need to project gate_input to bottleneck_dim for the gate
                gate = torch.sigmoid(self.fusion_gates[i](gate_input))  # (batch, approx_len, bottleneck_dim)
                fused_b = gate * fused_b + (1 - gate) * combined_b  # (batch, approx_len, bottleneck_dim)
                
                fused = self.fusion_up[i](fused_b)  # (batch, approx_len, hidden_dim)
        
        # Upsample to input length if needed
        if fused.shape[1] != L_in:
            fused = F.interpolate(
                fused.transpose(1, 2),  # (batch, hidden_dim, fused_len)
                size=L_in,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # (batch, input_len, hidden_dim)
        
        # Periodic-aware temporal attention
        pe = self.get_periodic_encoding(L_in, x.device)  # (input_len, hidden_dim)
        pe = pe.unsqueeze(0).expand(B, -1, -1)  # (batch, input_len, hidden_dim)
        
        fused_with_pe = fused + pe  # (batch, input_len, hidden_dim)
        fused_with_pe = self.period_norm(fused_with_pe)  # (batch, input_len, hidden_dim)
        
        # Self-attention with periodic bias
        attn_out, _ = self.period_attention(
            fused_with_pe, fused_with_pe, fused_with_pe
        )  # (batch, input_len, hidden_dim)
        attn_out = fused + attn_out  # Residual connection
        
        # Local temporal convolution for fine-grained patterns
        conv_out = self.local_conv(attn_out.transpose(1, 2)).transpose(1, 2)  # (batch, input_len, hidden_dim)
        conv_out = self.local_norm(attn_out + conv_out)  # (batch, input_len, hidden_dim)
        
        # Final representation
        final = conv_out  # (batch, input_len, hidden_dim)
        
        # Project to output features
        out = self.output_proj(final)  # (batch, input_len, n_features)
        
        # Interpolate to output length
        if L_in != self.output_len:
            out = F.interpolate(
                out.transpose(1, 2),  # (batch, n_features, input_len)
                size=self.output_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # (batch, output_len, n_features)
        
        # Add residual connection from normalized input
        # Need to interpolate input to output length
        if L_in != self.output_len:
            x_resid = F.interpolate(
                x_input.transpose(1, 2),  # (batch, n_features, input_len)
                size=self.output_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # (batch, output_len, n_features)
        else:
            x_resid = x_input  # (batch, output_len, n_features)
        
        out = out + self.residual_weight * x_resid  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)

