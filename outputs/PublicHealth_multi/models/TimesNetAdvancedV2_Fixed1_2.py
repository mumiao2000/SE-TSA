import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class TimesNetAdvancedV2_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 512],
            'n_layers': [1, 4, 8],
            'top_k': [2, 4, 8],
            'd_ff': [128, 512, 2048],
            'dropout': [0.0, 0.3],
            'channel_mix_dim': [32, 64, 128],
            'n_hierarchical_scales': [2, 3, 4],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 256)
        self.n_layers = cfg.get('n_layers', 4)
        self.top_k = cfg.get('top_k', 4)
        self.d_ff = cfg.get('d_ff', 512)
        self.dropout = cfg.get('dropout', 0.3)
        self.channel_mix_dim = cfg.get('channel_mix_dim', 64)
        self.n_hierarchical_scales = cfg.get('n_hierarchical_scales', 3)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Channel mixing projection to enable cross-feature period discovery
        self.channel_mix = nn.Sequential(
            nn.Linear(self.n_features, self.channel_mix_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.channel_mix_dim, self.n_features)
        )  # (batch, seq, n_features) -> (batch, seq, n_features)
        
        # Input projection
        self.input_projection = nn.Linear(self.n_features, self.d_model)
        
        # TimesNet blocks with hierarchical period analysis
        self.blocks = nn.ModuleList([
            TimesNetBlockV2(
                d_model=self.d_model,
                d_ff=self.d_ff,
                top_k=self.top_k,
                dropout=self.dropout,
                n_hierarchical_scales=self.n_hierarchical_scales
            ) for _ in range(self.n_layers)
        ])
        
        # Layer norms
        self.norm = nn.LayerNorm(self.d_model)
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.n_features)
        
        # Direct skip connection for stable training
        self.skip_projection = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # RevIN normalization
        x_norm = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel mixing for cross-feature interaction
        x_mixed = x_norm + self.channel_mix(x_norm)  # (batch, input_len, n_features)
        
        # Input projection
        x_proj = self.input_projection(x_mixed)  # (batch, input_len, d_model)
        
        # TimesNet blocks with hierarchical period analysis
        residual = x_proj
        for block in self.blocks:
            x_proj = block(x_proj) + residual  # (batch, input_len, d_model)
            residual = x_proj
        
        # Final normalization
        x_proj = self.norm(x_proj)  # (batch, input_len, d_model)
        
        # Output projection to features
        x_out = self.output_projection(x_proj)  # (batch, input_len, n_features)
        
        # Direct skip from normalized input
        x_skip = self.skip_projection(x_norm.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Interpolate or project to output length
        if self.input_len != self.output_len:
            x_out = F.interpolate(
                x_out.permute(0, 2, 1), 
                size=self.output_len, 
                mode='linear', 
                align_corners=False
            ).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Combine with skip
        x_out = x_out + x_skip  # (batch, output_len, n_features)
        
        # Denormalize
        x_out = self.revin(x_out, mode='denorm')  # (batch, output_len, n_features)
        
        return x_out


class SEInceptionBlock(nn.Module):
    """Inception block with Squeeze-and-Excitation channel attention"""
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Multi-scale convolutions
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=5, padding=2)
        )
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        )
        
        # Squeeze-and-Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 16),
            nn.ReLU(),
            nn.Linear(out_channels // 16, out_channels),
            nn.Sigmoid()
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x: (batch, in_channels, H, W)
        b, c, h, w = x.shape
        
        # Multi-scale features
        out1 = self.conv1x1(x)  # (batch, out//4, H, W)
        out2 = self.conv3x3(x)  # (batch, out//4, H, W)
        out3 = self.conv5x5(x)  # (batch, out//4, H, W)
        out4 = self.pool_conv(x)  # (batch, out//4, H, W)
        
        out = torch.cat([out1, out2, out3, out4], dim=1)  # (batch, out_channels, H, W)
        
        # SE attention
        se_weight = self.se(out)  # (batch, out_channels)
        se_weight = se_weight.view(b, self.out_channels, 1, 1)  # (batch, out_channels, 1, 1)
        out = out * se_weight  # (batch, out_channels, H, W)
        
        out = self.bn(out)
        out = self.dropout(out)
        out = self.activation(out)
        
        return out


class TimesNetBlockV2(nn.Module):
    """Enhanced TimesNet block with hierarchical period discovery"""
    def __init__(self, d_model, d_ff, top_k=4, dropout=0.1, n_hierarchical_scales=3):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.n_hierarchical_scales = n_hierarchical_scales
        
        # Hierarchical scale factors (coarse to fine)
        self.scale_factors = [2 ** i for i in range(n_hierarchical_scales)]
        
        # Period discovery for each scale
        self.period_discoveries = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_hierarchical_scales)
        ])
        
        # 2D convolutions for each scale
        self.conv_layers = nn.ModuleList([
            nn.ModuleList([
                SEInceptionBlock(d_model, d_model, dropout) 
                for _ in range(top_k)
            ]) for _ in range(n_hierarchical_scales)
        ])
        
        # Scale fusion: project each timestep's d_model features to n_scales weights
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model, n_hierarchical_scales),
            nn.Softmax(dim=-1)
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        residual = x
        x = self.norm1(x)
        
        # Hierarchical period analysis
        scale_outputs = []
        
        for scale_idx, scale_factor in enumerate(self.scale_factors):
            # Downsample for coarse scales
            if scale_factor > 1:
                # Use strided pooling for downsampling
                scale_len = max(seq_len // scale_factor, 1)
                x_scaled = F.adaptive_avg_pool1d(
                    x.permute(0, 2, 1), scale_len
                ).permute(0, 2, 1)  # (batch, scale_len, d_model)
            else:
                x_scaled = x
                scale_len = seq_len
            
            # Period discovery via FFT
            xf = torch.fft.rfft(x_scaled, dim=1)  # (batch, freq_len, d_model)
            period_amp = torch.abs(xf).mean(dim=-1)  # (batch, freq_len)
            
            # Find top-k periods
            _, top_indices = torch.topk(period_amp, k=min(self.top_k, period_amp.shape[-1]), dim=-1)
            
            # Process each period
            period_outs = []
            for i in range(min(self.top_k, top_indices.shape[-1])):
                period = top_indices[:, i]  # (batch,)
                
                # Use a single scalar period for the batch to avoid tensor boolean issues
                # Use the mean period across batch, clamped to valid range
                p_scalar = int(period.float().mean().clamp(min=2, max=max(scale_len // 2, 2)).item())
                
                # Reshape for 2D conv
                x_reshaped = x_scaled.permute(0, 2, 1)  # (batch, d_model, scale_len)
                
                # Pad to make divisible by scalar period
                pad_len = (p_scalar - (scale_len % p_scalar)) % p_scalar
                if pad_len > 0:
                    x_reshaped = F.pad(x_reshaped, (0, pad_len))  # (batch, d_model, scale_len+pad)
                
                new_len = x_reshaped.shape[-1]
                x_2d = x_reshaped.reshape(batch_size, d_model, p_scalar, new_len // p_scalar)  # (batch, d_model, p, time//p)
                
                # Apply 2D convolution
                conv_out = self.conv_layers[scale_idx][i](x_2d)  # (batch, d_model, p, time//p)
                
                # Reshape back
                conv_out = conv_out.reshape(batch_size, d_model, new_len)  # (batch, d_model, new_len)
                conv_out = conv_out[:, :, :scale_len]  # Remove padding
                conv_out = conv_out.permute(0, 2, 1)  # (batch, scale_len, d_model)
                
                period_outs.append(conv_out)
            
            # Aggregate period outputs
            if period_outs:
                scale_out = torch.stack(period_outs, dim=0).mean(dim=0)  # (batch, scale_len, d_model)
            else:
                scale_out = x_scaled
            
            # Upsample back to original length if needed
            if scale_factor > 1:
                scale_out = F.interpolate(
                    scale_out.permute(0, 2, 1), 
                    size=seq_len, 
                    mode='linear', 
                    align_corners=False
                ).permute(0, 2, 1)  # (batch, seq_len, d_model)
            
            scale_outputs.append(scale_out)
        
        # Fuse scales with learned attention weights
        # Compute attention weights from the first (finest) scale output
        # scale_outputs[0]: (batch, seq_len, d_model)
        attn_weights = self.scale_fusion(scale_outputs[0])  # (batch, seq_len, n_hierarchical_scales)
        
        # Stack scale outputs: (n_scales, batch, seq_len, d_model)
        stacked_scales = torch.stack(scale_outputs, dim=0)  # (n_scales, batch, seq_len, d_model)
        
        # Apply attention weights with proper broadcasting
        # attn_weights: (batch, seq_len, n_scales) -> (n_scales, batch, seq_len, 1)
        attn_weights = attn_weights.permute(2, 0, 1).unsqueeze(-1)  # (n_scales, batch, seq_len, 1)
        
        # Weighted sum: (n_scales, batch, seq_len, d_model) * (n_scales, batch, seq_len, 1)
        out = (stacked_scales * attn_weights).sum(dim=0)  # (batch, seq_len, d_model)
        
        # Residual connection 1
        out = out + residual
        
        # FFN with residual
        residual2 = out
        out = self.norm2(out)
        out = self.ffn(out) + residual2
        
        return out

