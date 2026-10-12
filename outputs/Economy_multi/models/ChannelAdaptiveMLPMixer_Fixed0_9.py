import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveMLPMixer_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_groups': [2, 4, 7],
            'mixer_dim': [128, 512, 2048],
            'dropout': [0.0, 0.2, 0.4],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_groups = self.config.get('n_groups', 4)
        self.mixer_dim = self.config.get('mixer_dim', 512)
        self.dropout = self.config.get('dropout', 0.2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Feature grouping: assign each feature to a group
        self.group_logits = nn.Parameter(torch.randn(n_features, self.n_groups))
        
        # Per-group temporal and channel MLPs
        self.temporal_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_len, self.mixer_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.mixer_dim, input_len)
            ) for _ in range(self.n_groups)
        ])
        
        # Channel MLPs operate on n_features dimension
        self.channel_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features, self.mixer_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.mixer_dim, n_features)
            ) for _ in range(self.n_groups)
        ])
        
        # Feature projection per group: input_len -> hidden_dim
        self.feat_projs = nn.ModuleList([
            nn.Linear(input_len, self.hidden_dim) for _ in range(self.n_groups)
        ])
        
        # Cross-group fusion
        self.cross_fusion = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, output_len)
        )
        
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Soft group assignment
        group_weights = F.softmax(self.group_logits, dim=-1)  # (n_features, n_groups)
        
        # Process each group
        group_outputs = []
        for g in range(self.n_groups):
            # Weight features by group membership
            w = group_weights[:, g:g+1]  # (n_features, 1)
            # Apply to batch: (batch, n_features, input_len) * (1, n_features, 1)
            x_g = x * w.unsqueeze(0)  # (batch, n_features, input_len)
            
            # Temporal mixing: operates on last dim (input_len)
            x_g = x_g + self.temporal_mlps[g](x_g)  # (batch, n_features, input_len)
            
            # Channel mixing: transpose to make n_features last dim
            x_g_t = x_g.permute(0, 2, 1)  # (batch, input_len, n_features)
            x_g_t = x_g_t + self.channel_mlps[g](x_g_t)  # (batch, input_len, n_features)
            x_g = x_g_t.permute(0, 2, 1)  # (batch, n_features, input_len)
            
            # Project to hidden dim
            h_g = self.feat_projs[g](x_g)  # (batch, n_features, hidden_dim)
            
            group_outputs.append(h_g)
        
        # Combine group outputs with weighted sum
        h = torch.stack(group_outputs, dim=-1)  # (batch, n_features, hidden_dim, n_groups)
        w = group_weights.unsqueeze(0).unsqueeze(2)  # (1, n_features, 1, n_groups)
        h = (h * w).sum(dim=-1)  # (batch, n_features, hidden_dim)
        
        # Cross-feature attention for global interactions
        h_norm = self.norm1(h)  # (batch, n_features, hidden_dim)
        h_attn, _ = self.cross_fusion(h_norm, h_norm, h_norm)  # (batch, n_features, hidden_dim)
        h = h + h_attn  # (batch, n_features, hidden_dim)
        
        h = self.norm2(h)  # (batch, n_features, hidden_dim)
        
        # Output projection per feature
        out = self.output_proj(h)  # (batch, n_features, output_len)
        
        # Transpose back
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

