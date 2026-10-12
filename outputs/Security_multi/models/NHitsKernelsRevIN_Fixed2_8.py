import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NHitsKernelsRevIN_Fixed2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_scales': [2, 3, 4],
            'n_basis_kernels': [4, 8, 16],
            'dropout': [0.0, 0.3],
            'n_blocks': [1, 2, 4],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_scales = self.config.get('n_scales', 3)
        self.n_basis_kernels = self.config.get('n_basis_kernels', 8)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.n_blocks = self.config.get('n_blocks', 2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Multi-scale pooling with safe dimension allocation
        self.pool_scales = []
        self.pool_projs = nn.ModuleList()
        
        # Calculate safe pool dimensions - ensure they sum to hidden_dim
        scale_values = []
        for i in range(self.n_scales):
            pool_size = 2 ** i
            # Only use scales where pooling is meaningful
            if self.input_len >= pool_size * 2:
                scale_values.append(pool_size)
        self.n_scales = len(scale_values) if scale_values else 1
        if self.n_scales == 0:
            self.n_scales = 1
            scale_values = [1]
        
        # Distribute hidden_dim across scales evenly, with remainder to last
        base_dim = self.hidden_dim // self.n_scales
        remainder = self.hidden_dim - base_dim * self.n_scales
        self.scale_dims = []
        for i in range(self.n_scales):
            dim = base_dim + (remainder if i == self.n_scales - 1 else 0)
            self.scale_dims.append(dim)
        
        for i, pool_size in enumerate(scale_values):
            self.pool_projs.append(nn.Linear(self.input_len // pool_size if pool_size > 1 else self.input_len, self.scale_dims[i]))
        
        self.scale_values = scale_values
        
        # Learnable basis kernels for temporal modeling
        self.basis_kernels = nn.Parameter(torch.randn(self.n_basis_kernels, self.output_len, self.hidden_dim) * 0.02)
        self.basis_weights = nn.Linear(self.hidden_dim, self.n_basis_kernels)
        
        # Gating network for multi-scale fusion
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        
        # Residual blocks with proper projection
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(self.hidden_dim),
                'mlp': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                    nn.Dropout(self.dropout_rate),
                ),
                'norm2': nn.LayerNorm(self.hidden_dim),
            }))
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        
        # Residual projection if needed
        self.residual_proj = nn.Linear(self.input_len, self.output_len) if self.input_len != self.output_len else None
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save residual for later
        residual = x  # (batch, input_len, n_features)
        
        # Input projection: (batch, input_len, hidden_dim)
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Multi-scale pooling with explicit per-scale handling
        scale_outputs = []
        for i, (pool_size, dim) in enumerate(zip(self.scale_values, self.scale_dims)):
            if pool_size > 1:
                # Adaptive avg pool: (batch, hidden_dim, input_len) -> (batch, hidden_dim, input_len//pool_size)
                x_perm = x.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
                pooled = F.adaptive_avg_pool1d(x_perm, self.input_len // pool_size)  # (batch, hidden_dim, input_len//pool_size)
                pooled = pooled.permute(0, 2, 1)  # (batch, input_len//pool_size, hidden_dim)
            else:
                pooled = x  # (batch, input_len, hidden_dim)
            
            # Project to scale-specific dimension
            # We need to handle the feature dimension carefully
            # pooled: (batch, seq_len, hidden_dim), we want to project the seq_len dimension
            pooled_proj = self.pool_projs[i](pooled.permute(0, 2, 1))  # (batch, hidden_dim, dim)
            pooled_proj = pooled_proj.permute(0, 2, 1)  # (batch, dim, hidden_dim)
            
            # Truncate or pad to match expected dim
            if pooled_proj.size(1) > dim:
                pooled_proj = pooled_proj[:, :dim, :]  # (batch, dim, hidden_dim)
            elif pooled_proj.size(1) < dim:
                pad_size = dim - pooled_proj.size(1)
                pooled_proj = F.pad(pooled_proj, (0, 0, 0, pad_size))  # (batch, dim, hidden_dim)
            
            scale_outputs.append(pooled_proj)
        
        # Concatenate along the sequence dimension: (batch, hidden_dim, hidden_dim)
        multi_scale = torch.cat(scale_outputs, dim=1)  # (batch, sum(scale_dims), hidden_dim)
        
        # Ensure exact hidden_dim in sequence dimension by adaptive pooling
        multi_scale = multi_scale.permute(0, 2, 1)  # (batch, hidden_dim, sum(scale_dims))
        multi_scale = F.adaptive_avg_pool1d(multi_scale, self.hidden_dim)  # (batch, hidden_dim, hidden_dim)
        multi_scale = multi_scale.permute(0, 2, 1)  # (batch, hidden_dim, hidden_dim)
        
        # Take diagonal-like aggregation: average over the second dimension to get (batch, hidden_dim, hidden_dim) -> no
        # Actually we need (batch, hidden_dim, hidden_dim) to become features for each time step
        # Let's use a different approach: treat as (batch, hidden_dim, hidden_dim) and do attention
        
        # Simpler: global average pool to get (batch, hidden_dim)
        pooled_features = multi_scale.mean(dim=1)  # (batch, hidden_dim)
        
        # Expand to output length
        hidden = pooled_features.unsqueeze(1).expand(-1, self.output_len, -1)  # (batch, output_len, hidden_dim)
        
        # Apply basis kernel modulation
        # basis_kernels: (n_basis_kernels, output_len, hidden_dim)
        # We want to compute weighted combination of basis kernels
        basis_weights = self.basis_weights(pooled_features)  # (batch, n_basis_kernels)
        basis_weights = F.softmax(basis_weights, dim=-1)  # (batch, n_basis_kernels)
        
        # Weighted basis: (batch, output_len, hidden_dim)
        weighted_basis = torch.einsum('bk,ktd->btd', basis_weights, self.basis_kernels)  # (batch, output_len, hidden_dim)
        
        # Gated fusion with basis
        gate = self.gate(hidden)  # (batch, output_len, hidden_dim)
        hidden = gate * hidden + (1 - gate) * weighted_basis  # (batch, output_len, hidden_dim)
        
        # Residual blocks
        for block in self.blocks:
            hidden = block['norm1'](hidden)  # (batch, output_len, hidden_dim)
            hidden = hidden + block['mlp'](hidden)  # (batch, output_len, hidden_dim)
            hidden = block['norm2'](hidden)  # (batch, output_len, hidden_dim)
        
        # Output projection
        out = self.output_proj(hidden)  # (batch, output_len, n_features)
        
        # Add residual connection in original space
        if self.residual_proj is not None:
            residual = residual.permute(0, 2, 1)  # (batch, n_features, input_len)
            residual = self.residual_proj(residual)  # (batch, n_features, output_len)
            residual = residual.permute(0, 2, 1)  # (batch, output_len, n_features)
        else:
            residual = residual[:, :self.output_len, :]  # (batch, output_len, n_features)
        
        out = out + residual  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
