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
            'n_mixer_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.temporal_dim = self.config.get('temporal_dim', 256)
        self.n_mixer_layers = self.config.get('n_mixer_layers', 3)
        self.dropout = self.config.get('dropout', 0.0)
        
        # Temporal projection: input_len -> temporal_dim for each feature
        self.temporal_proj = nn.Linear(self.input_len, self.temporal_dim)
        
        # Channel attention blocks
        self.channel_attn_layers = nn.ModuleList()
        for _ in range(self.n_mixer_layers):
            layer = nn.ModuleDict({
                'norm1': nn.LayerNorm(self.temporal_dim),
                'qkv': nn.Linear(self.n_features, self.n_features * 3),
                'proj': nn.Linear(self.n_features, self.n_features),
                'norm2': nn.LayerNorm(self.temporal_dim),
                'mlp': nn.Sequential(
                    nn.Linear(self.temporal_dim, self.temporal_dim * 4),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.temporal_dim * 4, self.temporal_dim),
                    nn.Dropout(self.dropout)
                )
            })
            self.channel_attn_layers.append(layer)
        
        # Output projection: temporal_dim -> output_len
        self.output_proj = nn.Linear(self.temporal_dim, self.output_len)
        
        self.revin = RevIN(self.n_features, affine=True)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to temporal dimension: (batch, n_features, temporal_dim)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.temporal_proj(x)  # (batch, n_features, temporal_dim)
        
        # Channel attention + MLP mixer blocks
        for layer in self.channel_attn_layers:
            # Channel attention: treat features as sequence
            residual = x  # (batch, n_features, temporal_dim)
            x = layer['norm1'](x)  # (batch, n_features, temporal_dim)
            
            # Attention across channels
            x_t = x.permute(0, 2, 1)  # (batch, temporal_dim, n_features)
            qkv = layer['qkv'](x_t)  # (batch, temporal_dim, n_features * 3)
            q, k, v = qkv.chunk(3, dim=-1)  # each (batch, temporal_dim, n_features)
            
            # Scaled dot-product attention
            attn = torch.matmul(q, k.transpose(-2, -1)) / (self.n_features ** 0.5)  # (batch, temporal_dim, temporal_dim)
            attn = F.softmax(attn, dim=-1)  # (batch, temporal_dim, temporal_dim)
            out = torch.matmul(attn, v)  # (batch, temporal_dim, n_features)
            out = layer['proj'](out)  # (batch, temporal_dim, n_features)
            out = out.permute(0, 2, 1)  # (batch, n_features, temporal_dim)
            
            x = residual + out  # (batch, n_features, temporal_dim)
            
            # Temporal MLP
            residual = x  # (batch, n_features, temporal_dim)
            x = layer['norm2'](x)  # (batch, n_features, temporal_dim)
            x = layer['mlp'](x)  # (batch, n_features, temporal_dim)
            x = residual + x  # (batch, n_features, temporal_dim)
        
        # Output: project temporal_dim -> output_len
        out = self.output_proj(x)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

