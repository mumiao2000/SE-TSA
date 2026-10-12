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
            'n_mlp_layers': [1, 3, 6],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.temporal_dim = self.config.get('temporal_dim', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_mlp_layers = self.config.get('n_mlp_layers', 3)
        
        assert self.temporal_dim % self.n_heads == 0, "temporal_dim must be divisible by n_heads"
        self.head_dim = self.temporal_dim // self.n_heads
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Temporal MLP encoder (per-channel initially)
        self.temporal_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_len if i == 0 else self.temporal_dim, self.temporal_dim),
                nn.GELU(),
                nn.LayerNorm(self.temporal_dim),
                nn.Dropout(0.1)
            ) for i in range(self.n_mlp_layers)
        ])
        
        # Cross-channel attention
        self.channel_query = nn.Linear(self.temporal_dim, self.temporal_dim)
        self.channel_key = nn.Linear(self.temporal_dim, self.temporal_dim)
        self.channel_value = nn.Linear(self.temporal_dim, self.temporal_dim)
        self.channel_out = nn.Linear(self.temporal_dim, self.temporal_dim)
        
        # Adaptive feature importance
        self.feature_importance = nn.Sequential(
            nn.Linear(n_features * self.temporal_dim, n_features),
            nn.Sigmoid()
        )
        
        # Temporal decoder (per-channel)
        self.temporal_decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.temporal_dim if i == 0 else self.temporal_dim, self.temporal_dim),
                nn.GELU(),
                nn.LayerNorm(self.temporal_dim),
                nn.Dropout(0.1)
            ) for i in range(self.n_mlp_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.temporal_dim, output_len)
        
        # Direct shortcut
        self.shortcut = nn.Linear(input_len, output_len)
        
    def channel_attention(self, x):
        # x: (batch, n_features, temporal_dim)
        batch_size = x.shape[0]
        
        Q = self.channel_query(x)  # (batch, n_features, temporal_dim)
        K = self.channel_key(x)  # (batch, n_features, temporal_dim)
        V = self.channel_value(x)  # (batch, n_features, temporal_dim)
        
        # Multi-head attention across channels
        Q = Q.view(batch_size, self.n_features, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, self.n_features, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, self.n_features, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch, n_heads, n_features, head_dim)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # (batch, n_heads, n_features, n_features)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, V)  # (batch, n_heads, n_features, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, self.n_features, self.temporal_dim)
        out = self.channel_out(out)  # (batch, n_features, temporal_dim)
        return out
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Per-channel temporal encoding
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        h = x_perm  # (batch, n_features, input_len)
        for layer in self.temporal_encoder:
            h = layer(h)  # (batch, n_features, temporal_dim)
        
        # Cross-channel attention
        h_attn = self.channel_attention(h)  # (batch, n_features, temporal_dim)
        h = h + h_attn  # (batch, n_features, temporal_dim) - residual
        
        # Adaptive feature importance
        feat_flat = h.reshape(x.shape[0], -1)  # (batch, n_features * temporal_dim)
        importance = self.feature_importance(feat_flat)  # (batch, n_features)
        h = h * importance.unsqueeze(-1)  # (batch, n_features, temporal_dim)
        
        # Temporal decoding
        for layer in self.temporal_decoder:
            h = layer(h)  # (batch, n_features, temporal_dim)
        
        # Output projection
        out = self.output_proj(h)  # (batch, n_features, output_len)
        
        # Shortcut
        shortcut = self.shortcut(x_perm)  # (batch, n_features, output_len)
        out = out + shortcut  # (batch, n_features, output_len)
        
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

