import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttentionMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'temporal_dim': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'mlp_ratio': [1, 2, 4],
            'n_blocks': [1, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.temporal_dim = self.config.get('temporal_dim', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.mlp_ratio = self.config.get('mlp_ratio', 2)
        self.n_blocks = self.config.get('n_blocks', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Temporal projection: each feature gets its own temporal embedding
        self.temporal_proj = nn.Linear(self.input_len, self.temporal_dim)
        
        # Cross-channel attention blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(nn.ModuleDict({
                'channel_norm': nn.LayerNorm(self.temporal_dim),
                'channel_attn': nn.MultiheadAttention(
                    embed_dim=self.temporal_dim,
                    num_heads=self.n_heads,
                    batch_first=True,
                    dropout=0.1
                ),
                'mlp_norm': nn.LayerNorm(self.temporal_dim),
                'mlp': nn.Sequential(
                    nn.Linear(self.temporal_dim, int(self.temporal_dim * self.mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(int(self.temporal_dim * self.mlp_ratio), self.temporal_dim)
                ),
                'gate': nn.Sequential(
                    nn.Linear(self.temporal_dim, self.temporal_dim),
                    nn.Sigmoid()
                )
            }))
        
        # Output: predict per-feature, per-timestep
        self.output_proj = nn.Linear(self.temporal_dim, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project each feature to temporal dimension: (batch, n_features, temporal_dim)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.temporal_proj(x)  # (batch, n_features, temporal_dim)
        
        # Apply cross-channel attention blocks
        for block in self.blocks:
            residual = x  # (batch, n_features, temporal_dim)
            
            # Cross-channel attention: treat n_features as sequence length
            x_norm = block['channel_norm'](x)  # (batch, n_features, temporal_dim)
            attn_out, _ = block['channel_attn'](x_norm, x_norm, x_norm)  # (batch, n_features, temporal_dim)
            x = x + attn_out  # (batch, n_features, temporal_dim)
            
            # Gated MLP
            x_norm = block['mlp_norm'](x)  # (batch, n_features, temporal_dim)
            mlp_out = block['mlp'](x_norm)  # (batch, n_features, temporal_dim)
            gate = block['gate'](x_norm)  # (batch, n_features, temporal_dim)
            x = x + mlp_out * gate  # (batch, n_features, temporal_dim)
        
        # Output projection per feature
        x = self.output_proj(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)   # (batch, output_len, n_features)
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x
