import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NLinearChannelMixing_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'mixing_dim': [32, 128, 512],
            'n_mixing_layers': [1, 2, 4],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.mixing_dim = self.config.get('mixing_dim', 128)
        self.n_mixing_layers = self.config.get('n_mixing_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Learnable temporal weights: (output_len, input_len)
        self.temporal_weights = nn.Parameter(torch.randn(self.output_len, self.input_len) * 0.02)
        
        # Channel mixing MLP
        self.mixing_layers = nn.ModuleList()
        self.mixing_norms = nn.ModuleList()
        
        in_dim = self.n_features
        for i in range(self.n_mixing_layers):
            out_dim = self.mixing_dim if i < self.n_mixing_layers - 1 else self.n_features
            self.mixing_layers.append(nn.Linear(in_dim, out_dim))
            # LayerNorm on the INPUT to each layer (before linear), so dimension is in_dim
            self.mixing_norms.append(nn.LayerNorm(in_dim))
            in_dim = out_dim
        
        self.mixing_act = nn.GELU()
        self.mixing_drop = nn.Dropout(self.dropout)
        
        # Residual scale
        self.residual_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch, seq_len, n_feat = x.shape
        
        # Normalize: subtract mean per feature across time
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        x_norm = x - x_mean  # (batch, input_len, n_features)
        
        # Temporal aggregation with learnable weights
        x_perm = x_norm.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Apply temporal weights: (output_len, input_len)
        x_agg = torch.einsum('oi,bfi->bfo', self.temporal_weights, x_perm)  # (batch, n_features, output_len)
        x_agg = x_agg.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Channel mixing
        mixed = x_agg  # (batch, output_len, n_features)
        for i, layer in enumerate(self.mixing_layers):
            # Apply norm, activation, dropout to input of layer
            mixed = self.mixing_norms[i](mixed)  # (batch, output_len, in_dim)
            mixed = self.mixing_act(mixed)  # (batch, output_len, in_dim)
            mixed = self.mixing_drop(mixed)  # (batch, output_len, in_dim)
            mixed = layer(mixed)  # (batch, output_len, out_dim)
        
        # Add residual from temporal aggregation (scaled)
        out = mixed + self.residual_scale * x_agg  # (batch, output_len, n_features)
        
        # Add back mean (trend)
        out = out + x_mean  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)

