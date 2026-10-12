import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveMLPMixer_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_blocks': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'dilation_base': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        hidden_dim = self.config.get('hidden_dim', 256)
        n_blocks = self.config.get('n_blocks', 4)
        dropout = self.config.get('dropout', 0.1)
        dilation_base = self.config.get('dilation_base', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Project to hidden dim
        self.input_proj = nn.Linear(self.n_features, hidden_dim)
        
        # Channel importance gating
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = dilation_base ** (i % 3)
            self.blocks.append(nn.ModuleDict({
                'temporal_mix': nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation, groups=hidden_dim),
                'temporal_norm': nn.LayerNorm(hidden_dim),
                'channel_mix': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'channel_norm': nn.LayerNorm(hidden_dim)
            }))
        
        # Multi-scale temporal aggregation
        self.scale_pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True),
            nn.AvgPool1d(kernel_size=4, stride=4, ceil_mode=True),
            nn.AvgPool1d(kernel_size=8, stride=8, ceil_mode=True)
        ])
        
        # Project multi-scale features to compressed dim
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim // 4)
        self.fusion_norm = nn.LayerNorm(hidden_dim // 4)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.n_features)
        )
        
        # Direct horizon mapping
        self.horizon_mapper = nn.Linear(input_len, self.output_len)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden dim
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Apply channel gating
        x_perm = x.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        gate = self.channel_gate(x_perm)  # (batch, hidden_dim)
        gate = gate.unsqueeze(1)  # (batch, 1, hidden_dim)
        x = x * gate  # (batch, input_len, hidden_dim)
        
        # MLP-Mixer blocks
        for block in self.blocks:
            # Temporal mixing with dilation
            residual = x  # (batch, input_len, hidden_dim)
            x_temp = x.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
            x_temp = block['temporal_mix'](x_temp)  # (batch, hidden_dim, input_len)
            x_temp = x_temp.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
            x = block['temporal_norm'](x_temp + residual)  # (batch, input_len, hidden_dim)
            
            # Channel mixing
            residual = x  # (batch, input_len, hidden_dim)
            x = block['channel_mix'](x)  # (batch, input_len, hidden_dim)
            x = block['channel_norm'](x + residual)  # (batch, input_len, hidden_dim)
        
        # Multi-scale temporal aggregation
        x_perm = x.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        
        pooled_2 = self.scale_pools[0](x_perm)  # (batch, hidden_dim, pooled_len_2)
        pooled_4 = self.scale_pools[1](x_perm)  # (batch, hidden_dim, pooled_len_4)
        pooled_8 = self.scale_pools[2](x_perm)  # (batch, hidden_dim, pooled_len_8)
        
        # Interpolate back and concatenate
        pooled_2_up = F.interpolate(pooled_2, size=self.input_len, mode='linear', align_corners=False)  # (batch, hidden_dim, input_len)
        pooled_4_up = F.interpolate(pooled_4, size=self.input_len, mode='linear', align_corners=False)  # (batch, hidden_dim, input_len)
        pooled_8_up = F.interpolate(pooled_8, size=self.input_len, mode='linear', align_corners=False)  # (batch, hidden_dim, input_len)
        
        multi_scale = x_perm + 0.3 * pooled_2_up + 0.3 * pooled_4_up + 0.3 * pooled_8_up  # (batch, hidden_dim, input_len)
        
        # Map to output horizon
        multi_scale = multi_scale.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
        
        # Compress features
        compressed = self.fusion_proj(multi_scale)  # (batch, input_len, hidden_dim//4)
        compressed = self.fusion_norm(compressed)  # (batch, input_len, hidden_dim//4)
        
        # Map to output length
        compressed = compressed.permute(0, 2, 1)  # (batch, hidden_dim//4, input_len)
        out = self.horizon_mapper(compressed)  # (batch, hidden_dim//4, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, hidden_dim//4)
        
        # Final projection to features
        out = self.output_proj(out)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
