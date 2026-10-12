import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HierarchicalDecompositionForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_levels': [2, 4, 6],
            'base_kernel': [2, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        hidden_dim = self.config.get('hidden_dim', 256)
        n_levels = self.config.get('n_levels', 4)
        base_kernel = self.config.get('base_kernel', 4)
        dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        self.n_levels = n_levels
        self.hidden_dim = hidden_dim
        
        # Create hierarchical levels with different pooling scales
        self.level_pools = nn.ModuleList()
        self.level_projs = nn.ModuleList()
        self.level_mlps = nn.ModuleList()
        
        self.level_lengths = []
        for level in range(n_levels):
            kernel_size = base_kernel ** level
            if kernel_size > input_len:
                break
            
            # Average pool to create coarser representation
            self.level_pools.append(nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size))
            
            # Calculate output length of this pooling
            level_len = (input_len + kernel_size - 1) // kernel_size
            self.level_lengths.append(level_len)
            
            # Projection to hidden dim
            self.level_projs.append(nn.Linear(self.n_features, hidden_dim))
            
            # MLP for this level
            self.level_mlps.append(nn.Sequential(
                nn.Linear(level_len, level_len * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(level_len * 2, output_len),
                nn.Dropout(dropout)
            ))
        
        self.actual_n_levels = len(self.level_lengths)
        
        # Gating network to combine levels
        self.gate_net = nn.Sequential(
            nn.Linear(self.actual_n_levels * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.actual_n_levels),
            nn.Softmax(dim=-1)
        )
        
        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.n_features)
        )
        
        # Residual connection
        self.residual_proj = nn.Linear(input_len, output_len)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Compute residual for later
        residual = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        residual = self.residual_proj(residual)  # (batch, n_features, output_len)
        residual = residual.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Process each hierarchical level
        level_outputs = []
        level_features = []
        
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        for i in range(self.actual_n_levels):
            # Pool to coarser scale
            pooled = self.level_pools[i](x_perm)  # (batch, n_features, level_len)
            pooled = pooled.permute(0, 2, 1)  # (batch, level_len, n_features)
            
            # Project to hidden dim
            proj = self.level_projs[i](pooled)  # (batch, level_len, hidden_dim)
            level_features.append(proj.mean(dim=1))  # (batch, hidden_dim)
            
            # Transpose for MLP (operate on temporal dimension)
            proj_temp = proj.permute(0, 2, 1)  # (batch, hidden_dim, level_len)
            
            # Apply level-specific MLP to map to output length
            mlp_out = self.level_mlps[i](proj_temp)  # (batch, hidden_dim, output_len)
            level_outputs.append(mlp_out)  # (batch, hidden_dim, output_len)
        
        # Stack level outputs: (batch, actual_n_levels, hidden_dim, output_len)
        stacked = torch.stack(level_outputs, dim=1)  # (batch, actual_n_levels, hidden_dim, output_len)
        
        # Compute gates from level features
        level_features_cat = torch.cat([f.unsqueeze(1) for f in level_features], dim=1)  # (batch, actual_n_levels, hidden_dim)
        level_features_flat = level_features_cat.reshape(level_features_cat.size(0), -1)  # (batch, actual_n_levels * hidden_dim)
        gates = self.gate_net(level_features_flat)  # (batch, actual_n_levels)
        gates = gates.reshape(gates.size(0), self.actual_n_levels, 1, 1)  # (batch, actual_n_levels, 1, 1)
        
        # Weighted combination
        combined = (stacked * gates).sum(dim=1)  # (batch, hidden_dim, output_len)
        combined = combined.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Final projection
        out = self.output_proj(combined)  # (batch, output_len, n_features)
        
        # Add residual
        out = out + 0.1 * residual  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
