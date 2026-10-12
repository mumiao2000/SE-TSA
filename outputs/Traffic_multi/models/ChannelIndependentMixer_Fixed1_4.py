import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelIndependentMixer_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'expansion_factor': [2, 4, 8],
            'n_blocks': [2, 4, 8],
            'dropout': [0.0, 0.3],
            'use_glu': [True, False],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        hidden_dim = self.config.get('hidden_dim', 256)
        expansion_factor = self.config.get('expansion_factor', 2)
        n_blocks = self.config.get('n_blocks', 4)
        dropout = self.config.get('dropout', 0.3)
        use_glu = self.config.get('use_glu', True)
        
        self.hidden_dim = hidden_dim
        self.use_glu = use_glu
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: (batch, input_len, n_features) -> (batch, n_features, hidden_dim)
        self.input_proj = nn.Linear(self.input_len, hidden_dim)
        
        # Mixer blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(MixerBlock(hidden_dim, self.n_features, expansion_factor, dropout, use_glu))
        
        # Two-stage projection head for richer representations
        self.proj_mid = nn.Linear(hidden_dim, hidden_dim // 2)
        self.proj_out = nn.Linear(hidden_dim // 2, self.output_len)
        
        # Learnable residual scaling for output
        self.residual_scale = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save residual for later
        residual = x  # (batch, input_len, n_features)
        
        # Transpose to channel-independent format: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project to hidden dim: (batch, n_features, hidden_dim)
        x = self.input_proj(x)  # (batch, n_features, hidden_dim)
        
        # Apply mixer blocks
        for block in self.blocks:
            x = block(x)  # (batch, n_features, hidden_dim)
        
        # Two-stage projection to output length
        x = self.proj_mid(x)  # (batch, n_features, hidden_dim // 2)
        x = F.gelu(x)  # (batch, n_features, hidden_dim // 2)
        x = self.proj_out(x)  # (batch, n_features, output_len)
        
        # Transpose back: (batch, output_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Add scaled residual connection (project input_len to output_len via interpolation)
        if self.input_len != self.output_len:
            residual = F.interpolate(
                residual.permute(0, 2, 1),  # (batch, n_features, input_len)
                size=self.output_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = x + self.residual_scale * residual  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)


class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, n_features, expansion_factor, dropout, use_glu=True):
        super().__init__()
        self.use_glu = use_glu
        
        # Temporal mixing: operates on hidden_dim dimension
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        temporal_expanded = hidden_dim * expansion_factor
        if use_glu:
            self.temporal_fc1 = nn.Linear(hidden_dim, temporal_expanded * 2)
            self.temporal_fc2 = nn.Linear(temporal_expanded, hidden_dim)
        else:
            self.temporal_fc1 = nn.Linear(hidden_dim, temporal_expanded)
            self.temporal_fc2 = nn.Linear(temporal_expanded, hidden_dim)
        self.temporal_dropout = nn.Dropout(dropout)
        
        # Channel mixing: operates on n_features dimension
        self.channel_norm = nn.LayerNorm(n_features)
        channel_expanded = n_features * expansion_factor
        if use_glu:
            self.channel_fc1 = nn.Linear(n_features, channel_expanded * 2)
            self.channel_fc2 = nn.Linear(channel_expanded, n_features)
        else:
            self.channel_fc1 = nn.Linear(n_features, channel_expanded)
            self.channel_fc2 = nn.Linear(channel_expanded, n_features)
        self.channel_dropout = nn.Dropout(dropout)
        
        # Learnable residual scaling
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # x: (batch, n_features, hidden_dim)
        
        # Temporal mixing
        residual = x  # (batch, n_features, hidden_dim)
        x = self.temporal_norm(x)  # (batch, n_features, hidden_dim)
        if self.use_glu:
            x = self.temporal_fc1(x)  # (batch, n_features, temporal_expanded * 2)
            x, gate = x.chunk(2, dim=-1)  # each: (batch, n_features, temporal_expanded)
            x = F.gelu(x) * gate  # (batch, n_features, temporal_expanded)
        else:
            x = F.gelu(self.temporal_fc1(x))  # (batch, n_features, temporal_expanded)
        x = self.temporal_dropout(x)  # (batch, n_features, temporal_expanded)
        x = self.temporal_fc2(x)  # (batch, n_features, hidden_dim)
        x = residual + self.alpha1 * x  # (batch, n_features, hidden_dim)
        
        # Channel mixing
        residual = x  # (batch, n_features, hidden_dim)
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, n_features)
        x = self.channel_norm(x)  # (batch, hidden_dim, n_features)
        if self.use_glu:
            x = self.channel_fc1(x)  # (batch, hidden_dim, channel_expanded * 2)
            x, gate = x.chunk(2, dim=-1)  # each: (batch, hidden_dim, channel_expanded)
            x = F.gelu(x) * gate  # (batch, hidden_dim, channel_expanded)
        else:
            x = F.gelu(self.channel_fc1(x))  # (batch, hidden_dim, channel_expanded)
        x = self.channel_dropout(x)  # (batch, hidden_dim, channel_expanded)
        x = self.channel_fc2(x)  # (batch, hidden_dim, n_features)
        x = x.permute(0, 2, 1)  # (batch, n_features, hidden_dim)
        x = residual + self.alpha2 * x  # (batch, n_features, hidden_dim)
        
        return x  # (batch, n_features, hidden_dim)
