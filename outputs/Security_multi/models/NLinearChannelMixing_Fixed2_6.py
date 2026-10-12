import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NLinearChannelMixing_Fixed2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'mixing_dim': [64, 256, 1024],
            'n_mixing_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'temporal_agg': ['mean', 'learned']
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.mixing_dim = self.config.get('mixing_dim', 256)
        self.n_mixing_layers = self.config.get('n_mixing_layers', 4)
        self.dropout = self.config.get('dropout', 0.0)
        self.temporal_agg = self.config.get('temporal_agg', 'learned')
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Temporal reduction: project input_len to 1 per feature
        if self.temporal_agg == 'learned':
            self.temporal_proj = nn.Linear(self.input_len, 1)  # (batch, n_features, 1)
        else:
            self.temporal_proj = None
        
        # Input projection from n_features to mixing_dim
        self.input_proj = nn.Linear(self.n_features, self.mixing_dim)  # (batch, 1, mixing_dim)
        self.input_norm = nn.LayerNorm(self.mixing_dim)
        self.input_dropout = nn.Dropout(self.dropout)
        
        # Mixing layers with proper residual connections
        self.mixing_layers = nn.ModuleList()
        self.mixing_norms = nn.ModuleList()
        self.mixing_dropouts = nn.ModuleList()
        self.residual_projs = nn.ModuleList()  # For dimension matching
        
        for i in range(self.n_mixing_layers):
            self.mixing_layers.append(nn.Linear(self.mixing_dim, self.mixing_dim))
            self.mixing_norms.append(nn.LayerNorm(self.mixing_dim))
            self.mixing_dropouts.append(nn.Dropout(self.dropout))
            # No projection needed since dimensions are constant
            self.residual_projs.append(None)
        
        # Output projection: mixing_dim -> n_features, then replicate to output_len
        self.output_proj = nn.Linear(self.mixing_dim, self.n_features)  # (batch, 1, n_features)
        
        # Optional: learned temporal expansion
        self.output_temporal = nn.Linear(1, self.output_len)  # (batch, n_features, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for feature-wise operations: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Temporal aggregation to single value per feature
        if self.temporal_agg == 'learned':
            x = self.temporal_proj(x)  # (batch, n_features, 1)
        else:
            x = x.mean(dim=-1, keepdim=True)  # (batch, n_features, 1)
        
        # Transpose to (batch, 1, n_features) for channel mixing
        x = x.permute(0, 2, 1)  # (batch, 1, n_features)
        
        # Input projection to mixing dimension
        residual = x  # (batch, 1, n_features)
        x = self.input_proj(x)  # (batch, 1, mixing_dim)
        x = self.input_norm(x)
        x = F.relu(x)
        x = self.input_dropout(x)  # (batch, 1, mixing_dim)
        
        # Apply mixing layers with residual connections
        for i, (layer, norm, drop, proj) in enumerate(zip(
            self.mixing_layers, self.mixing_norms, self.mixing_dropouts, self.residual_projs
        )):
            residual_mix = x  # (batch, 1, mixing_dim)
            x = layer(x)  # (batch, 1, mixing_dim)
            x = norm(x)
            x = F.relu(x)
            x = drop(x)
            x = x + residual_mix  # (batch, 1, mixing_dim) - same dim, no projection needed
        
        # Output projection to n_features
        x = self.output_proj(x)  # (batch, 1, n_features)
        
        # Temporal expansion to output_len
        x = x.permute(0, 2, 1)  # (batch, n_features, 1)
        x = self.output_temporal(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x  # (batch, output_len, n_features)
