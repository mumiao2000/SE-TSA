import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelGatedMLP_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'gate_dim': [16, 32, 64],
            'n_levels': [2, 3, 4],
            'dropout': [0.0, 0.15, 0.3],
            'temporal_ratio': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.gate_dim = self.config.get('gate_dim', 32)
        self.n_levels = self.config.get('n_levels', 3)
        self.dropout = self.config.get('dropout', 0.15)
        self.temporal_ratio = self.config.get('temporal_ratio', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Feature-wise projection to hidden dimension
        self.feature_proj = nn.Linear(self.n_features, self.hidden_dim)
        # (batch, input_len, hidden_dim)
        
        # Hierarchical temporal decomposition levels
        self.level_blocks = nn.ModuleList()
        for level in range(self.n_levels):
            level_ratio = self.temporal_ratio ** level
            # Ensure minimum length of 1, and cap ratio to avoid zero-length
            effective_ratio = min(level_ratio, self.input_len)
            level_input_len = max(self.input_len // effective_ratio, 1)
            level_output_len = max(self.output_len // effective_ratio, 1)
            
            # Skip levels that would have identical resolution to previous (no new info)
            if level > 0 and level_input_len == max(self.input_len // (self.temporal_ratio ** (level - 1)), 1):
                continue
            
            self.level_blocks.append(
                TemporalLevelBlock(
                    level_input_len, level_output_len, 
                    self.hidden_dim, self.gate_dim, self.dropout
                )
            )
        
        # Update n_levels to actual number of blocks created
        self.n_levels = len(self.level_blocks)
        
        # Cross-channel gating mechanism
        self.gate_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.gate_dim),
            nn.Tanh(),
            nn.Linear(self.gate_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        # (batch, seq_len, hidden_dim)
        
        # Feature-wise output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        # (batch, output_len, n_features)
        
        # Residual connection for temporal shortcut
        self.temporal_shortcut = nn.Linear(self.input_len, self.output_len)
        # (batch, n_features, output_len)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode="norm")  # (batch, input_len, n_features)
        
        # Save for residual
        x_input = x  # (batch, input_len, n_features)
        
        # Project to hidden dimension
        h = self.feature_proj(x)  # (batch, input_len, hidden_dim)
        
        # Hierarchical temporal processing with multi-scale aggregation
        level_outputs = []
        for level, block in enumerate(self.level_blocks):
            level_ratio = self.temporal_ratio ** level
            # Downsample for this level
            if level_ratio > 1:
                # Clamp pool_size to not exceed input length
                pool_size = min(level_ratio, h.size(1))
                # Ensure pool_size doesn't make output too small
                if pool_size > h.size(1):
                    h_level = h  # (batch, input_len, hidden_dim)
                else:
                    h_level = F.avg_pool1d(
                        h.transpose(1, 2),  # (batch, hidden_dim, input_len)
                        kernel_size=pool_size,
                        stride=pool_size
                    ).transpose(1, 2)  # (batch, level_input_len, hidden_dim)
            else:
                h_level = h  # (batch, input_len, hidden_dim)
            
            # Process at this temporal scale
            h_level_out = block(h_level)  # (batch, level_output_len, hidden_dim)
            
            # Upsample back to output resolution if needed
            if level_ratio > 1:
                h_level_out = F.interpolate(
                    h_level_out.transpose(1, 2),  # (batch, hidden_dim, level_output_len)
                    size=self.output_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)  # (batch, output_len, hidden_dim)
            
            level_outputs.append(h_level_out)  # (batch, output_len, hidden_dim)
        
        # Aggregate multi-scale representations with learned weights
        if len(level_outputs) > 1:
            # Stack and weighted combination
            stacked = torch.stack(level_outputs, dim=-1)  # (batch, output_len, hidden_dim, n_levels)
            # Simple mean aggregation (could be learned)
            h_out = stacked.mean(dim=-1)  # (batch, output_len, hidden_dim)
        else:
            h_out = level_outputs[0]  # (batch, output_len, hidden_dim)
        
        # Cross-channel gating
        gate = self.gate_generator(h_out)  # (batch, output_len, hidden_dim)
        h_gated = h_out * gate  # (batch, output_len, hidden_dim)
        
        # Add residual from ungated path
        h_gated = h_gated + 0.1 * h_out  # (batch, output_len, hidden_dim)
        
        # Project to output features
        out = self.output_proj(h_gated)  # (batch, output_len, n_features)
        
        # Temporal shortcut connection from input
        shortcut = self.temporal_shortcut(
            x_input.transpose(1, 2)  # (batch, n_features, input_len)
        ).transpose(1, 2)  # (batch, output_len, n_features)
        
        out = out + 0.1 * shortcut  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode="denorm")  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)


class TemporalLevelBlock(nn.Module):
    def __init__(self, input_len, output_len, hidden_dim, gate_dim, dropout):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        
        # Temporal MLP for this level - bottleneck design
        bottleneck = max(hidden_dim // 4, 1)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(input_len, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, output_len)
        )
        # (batch, hidden_dim, output_len) after transpose
        
        # Channel-wise processing
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        # (batch, output_len, hidden_dim)
        
    def forward(self, x):
        # x: (batch, input_len, hidden_dim)
        # Transpose for temporal MLP
        x_t = x.transpose(1, 2)  # (batch, hidden_dim, input_len)
        x_t = self.temporal_mlp(x_t)  # (batch, hidden_dim, output_len)
        x_t = x_t.transpose(1, 2)  # (batch, output_len, hidden_dim)
        
        # Channel-wise residual MLP
        x_t = self.channel_norm(x_t)  # (batch, output_len, hidden_dim)
        x_t = x_t + self.channel_mlp(x_t)  # (batch, output_len, hidden_dim)
        
        return x_t  # (batch, output_len, hidden_dim)
