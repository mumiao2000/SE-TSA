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
            'temporal_agg': ['mean', 'weighted'],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.mixing_dim = self.config.get('mixing_dim', 256)
        self.n_mixing_layers = self.config.get('n_mixing_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.3)
        self.temporal_agg = self.config.get('temporal_agg', 'weighted')
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Temporal reduction: compress input_len to a fixed temporal dimension
        self.temporal_dim = max(1, input_len // 4)
        
        # Learnable temporal weights for aggregation
        if self.temporal_agg == 'weighted':
            self.temporal_weights = nn.Parameter(torch.ones(input_len))  # (input_len,)
        
        # First projection: n_features -> mixing_dim
        self.input_proj = nn.Linear(self.n_features, self.mixing_dim)  # (n_features, mixing_dim)
        self.input_norm = nn.LayerNorm(self.mixing_dim)  # (mixing_dim,)
        self.input_dropout = nn.Dropout(self.dropout_rate)
        
        # Channel mixing layers with proper residuals
        self.mixing_layers = nn.ModuleList()
        self.mixing_norms = nn.ModuleList()
        self.mixing_dropouts = nn.ModuleList()
        self.residual_projs = nn.ModuleList()  # Projections for dimension matching
        
        for i in range(self.n_mixing_layers):
            # Each mixing layer: temporal_dim -> temporal_dim (mixing across time for each channel)
            self.mixing_layers.append(nn.Linear(self.temporal_dim, self.temporal_dim))  # (temporal_dim, temporal_dim)
            self.mixing_norms.append(nn.LayerNorm(self.mixing_dim))  # (mixing_dim,) - norm applied after transpose
            self.mixing_dropouts.append(nn.Dropout(self.dropout_rate))
            # No dimension change in mixing layers, so no projection needed
            self.residual_projs.append(nn.Identity())
        
        # Output projection: mixing_dim -> n_features
        self.output_proj = nn.Linear(self.mixing_dim, self.n_features)  # (mixing_dim, n_features)
        
        # Final temporal expansion: temporal_dim -> output_len
        self.temporal_expand = nn.Linear(self.temporal_dim, self.output_len)  # (temporal_dim, output_len)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _temporal_aggregate(self, x):
        # x: (batch, input_len, n_features)
        if self.temporal_agg == 'weighted':
            # Softmax-normalized weights
            weights = F.softmax(self.temporal_weights, dim=0)  # (input_len,)
            # Weighted average: need to pool to temporal_dim segments
            # Simple approach: use convolution or strided pooling
            # Here we use adaptive average pooling with weights
            x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
            # Apply weighted pooling by reshaping
            batch, n_feat, seq = x.shape
            # Reshape to (batch, n_features, temporal_dim, chunk_size)
            chunk_size = seq // self.temporal_dim
            if chunk_size > 0:
                x = x[:, :, :self.temporal_dim * chunk_size]  # (batch, n_features, temporal_dim * chunk_size)
                x = x.reshape(batch, n_feat, self.temporal_dim, chunk_size)  # (batch, n_features, temporal_dim, chunk_size)
                # Apply weights to each chunk
                x = x.mean(dim=-1)  # (batch, n_features, temporal_dim)
            else:
                # Fallback: interpolate
                x = F.adaptive_avg_pool1d(x, self.temporal_dim)  # (batch, n_features, temporal_dim)
            x = x.permute(0, 2, 1)  # (batch, temporal_dim, n_features)
        else:
            # Mean pooling with adaptive pooling
            x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
            x = F.adaptive_avg_pool1d(x, self.temporal_dim)  # (batch, n_features, temporal_dim)
            x = x.permute(0, 2, 1)  # (batch, temporal_dim, n_features)
        return x  # (batch, temporal_dim, n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Temporal aggregation: reduce time dimension
        x = self._temporal_aggregate(x)  # (batch, temporal_dim, n_features)
        
        # First projection: n_features -> mixing_dim
        residual = x  # (batch, temporal_dim, n_features)
        x = self.input_proj(x)  # (batch, temporal_dim, mixing_dim)
        x = self.input_norm(x)  # (batch, temporal_dim, mixing_dim)
        x = F.gelu(x)  # (batch, temporal_dim, mixing_dim)
        x = self.input_dropout(x)  # (batch, temporal_dim, mixing_dim)
        
        # Need to project residual to match mixing_dim for first layer only
        # But we can't add residual here since shapes differ, so skip input residual
        
        # Channel mixing: operate on transposed tensor (mixing_dim as features, temporal_dim as sequence)
        for i, (layer, norm, dropout, res_proj) in enumerate(zip(
            self.mixing_layers, self.mixing_norms, self.mixing_dropouts, self.residual_projs)):
            
            # Transpose for temporal mixing: (batch, mixing_dim, temporal_dim)
            x_t = x.permute(0, 2, 1)  # (batch, mixing_dim, temporal_dim)
            
            # Apply mixing across temporal dimension for each channel (mixing_dim)
            residual_t = x_t  # (batch, mixing_dim, temporal_dim)
            out_t = layer(x_t)  # (batch, mixing_dim, temporal_dim)
            out_t = out_t + residual_t  # (batch, mixing_dim, temporal_dim) - residual same shape
            out_t = dropout(out_t)  # (batch, mixing_dim, temporal_dim)
            
            # Transpose back
            x = out_t.permute(0, 2, 1)  # (batch, temporal_dim, mixing_dim)
            x = norm(x)  # (batch, temporal_dim, mixing_dim)
            x = F.gelu(x)  # (batch, temporal_dim, mixing_dim)
        
        # Output projection: mixing_dim -> n_features
        x = self.output_proj(x)  # (batch, temporal_dim, n_features)
        
        # Temporal expansion: temporal_dim -> output_len
        x = x.permute(0, 2, 1)  # (batch, n_features, temporal_dim)
        x = self.temporal_expand(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x  # (batch, output_len, n_features)
