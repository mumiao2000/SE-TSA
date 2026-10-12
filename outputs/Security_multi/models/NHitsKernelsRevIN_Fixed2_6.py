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
            'n_blocks': [1, 4, 8],
            'n_pool_scales': [2, 4, 8],
            'dropout': [0.0, 0.3],
            'kernel_hidden': [32, 128, 512],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_blocks = self.config.get('n_blocks', 4)
        self.n_pool_scales = self.config.get('n_pool_scales', 4)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.kernel_hidden = self.config.get('kernel_hidden', 128)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Safe pool dimension allocation: guarantee sum equals hidden_dim
        self.pool_dims = []
        remaining = self.hidden_dim
        for i in range(self.n_pool_scales - 1):
            # Each scale gets at least 1 dimension
            dim = max(1, self.hidden_dim // (2 ** (i + 1)))
            dim = min(dim, remaining - (self.n_pool_scales - i - 1))  # leave enough for remaining
            self.pool_dims.append(dim)
            remaining -= dim
        self.pool_dims.append(remaining)  # last gets remainder
        # Verify
        assert sum(self.pool_dims) == self.hidden_dim, f"Pool dims {self.pool_dims} sum to {sum(self.pool_dims)}, expected {self.hidden_dim}"
        
        # Input projection
        self.input_proj = nn.Linear(self.input_len, self.hidden_dim)
        
        # N-HiTS style blocks with multi-scale pooling
        self.blocks = nn.ModuleList()
        for b in range(self.n_blocks):
            self.blocks.append(self._build_block())
        
        # Output projection: hidden_dim -> output_len per feature
        self.output_proj = nn.Linear(self.hidden_dim, self.output_len)
        
        # Residual connection for blocks
        if self.hidden_dim != self.input_len:
            self.residual_proj = nn.Linear(self.input_len, self.hidden_dim)
        else:
            self.residual_proj = None
            
        self.dropout = nn.Dropout(self.dropout_rate)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _build_block(self):
        """Build a single N-HiTS block with learnable basis kernels."""
        # Learnable temporal basis functions (kernel network)
        kernel_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.kernel_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.kernel_hidden, self.hidden_dim),
        )
        
        # Multi-scale pooling projections
        pool_projs = nn.ModuleList()
        for dim in self.pool_dims:
            pool_projs.append(nn.Linear(1, dim))  # project pooled scalar to dim
        
        # Gate network
        gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        
        return nn.ModuleDict({
            'kernel_net': kernel_net,
            'pool_projs': pool_projs,
            'gate': gate,
            'norm': nn.LayerNorm(self.hidden_dim),
        })
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose to (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        batch_size, n_feat, seq_len = x.shape
        
        # Input projection: (batch, n_features, input_len) -> (batch, n_features, hidden_dim)
        h = self.input_proj(x)  # (batch, n_features, hidden_dim)
        h = F.relu(h)
        h = self.dropout(h)  # (batch, n_features, hidden_dim)
        
        # Residual for first block
        if self.residual_proj is not None:
            residual = self.residual_proj(x)  # (batch, n_features, hidden_dim)
        else:
            residual = h  # (batch, n_features, hidden_dim)
        
        # Process through blocks
        for block in self.blocks:
            # Kernel network
            k = block['kernel_net'](h)  # (batch, n_features, hidden_dim)
            k = F.relu(k)
            
            # Multi-scale pooling with safe dimensions
            pooled_list = []
            for i, (pool_proj, dim) in enumerate(zip(block['pool_projs'], self.pool_dims)):
                pool_size = 2 ** i
                # Pad if necessary
                if seq_len < pool_size:
                    padded = F.pad(x, (0, pool_size - seq_len), mode='replicate')  # (batch, n_features, pool_size)
                else:
                    padded = x  # (batch, n_features, seq_len)
                
                # Adaptive pool to single value per feature
                if padded.shape[-1] >= pool_size:
                    # Use actual pooling
                    kernel_size = padded.shape[-1] // pool_size if padded.shape[-1] // pool_size > 0 else 1
                    pooled = F.avg_pool1d(padded, kernel_size=kernel_size)  # (batch, n_features, pooled_len)
                    # Take mean across pooled dimension
                    pooled = pooled.mean(dim=-1, keepdim=True)  # (batch, n_features, 1)
                else:
                    pooled = padded.mean(dim=-1, keepdim=True)  # (batch, n_features, 1)
                
                # Project to allocated dimension
                projected = pool_proj(pooled)  # (batch, n_features, dim)
                pooled_list.append(projected)
            
            # Concatenate all pooled features - guaranteed to be hidden_dim
            pooled_cat = torch.cat(pooled_list, dim=-1)  # (batch, n_features, hidden_dim)
            
            # Gate mechanism
            gate = block['gate'](h)  # (batch, n_features, hidden_dim)
            
            # Combine: gated kernel + pooled features
            out = gate * k + (1 - gate) * pooled_cat  # (batch, n_features, hidden_dim)
            out = block['norm'](out)  # (batch, n_features, hidden_dim)
            out = F.relu(out)
            out = self.dropout(out)
            
            # Residual connection
            h = out + residual  # (batch, n_features, hidden_dim)
            residual = h  # (batch, n_features, hidden_dim)
        
        # Output projection: (batch, n_features, hidden_dim) -> (batch, n_features, output_len)
        out = self.output_proj(h)  # (batch, n_features, output_len)
        
        # Transpose back: (batch, output_len, n_features)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
