import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class WaveletCrossAttention_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_levels': [2, 4, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_levels = self.config.get('n_levels', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Instance normalization for stability
        self.norm_input = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Learnable wavelet filters (Haar-like)
        self.register_buffer('lo_filter', torch.tensor([0.5, 0.5]))
        self.register_buffer('hi_filter', torch.tensor([0.5, -0.5]))
        
        # Projections for each decomposition level
        self.level_projs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        for level in range(self.n_levels):
            seq_len = self.input_len // (2 ** level)
            if seq_len < 2:
                break
            # Project from n_features to hidden_dim (applied per timestep)
            self.level_projs.append(nn.Linear(self.n_features, self.hidden_dim))
            self.attentions.append(nn.MultiheadAttention(
                self.hidden_dim, self.n_heads, dropout=self.dropout, batch_first=True
            ))
        
        self.n_actual_levels = len(self.level_projs)
        
        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * self.n_actual_levels, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_len * self.n_features)
        )
        
    def dwt(self, x):
        # x: (batch, n_features, seq_len)
        batch, n_feat, seq_len = x.shape
        if seq_len % 2 == 1:
            x = F.pad(x, (0, 1))  # (batch, n_features, seq_len+1)
            seq_len += 1
        
        x = x.reshape(batch * n_feat, 1, seq_len)  # (batch*n_features, 1, seq_len)
        
        # Low pass
        lo = F.conv1d(x, self.lo_filter.view(1, 1, 2), stride=2)  # (batch*n_features, 1, seq_len/2)
        # High pass  
        hi = F.conv1d(x, self.hi_filter.view(1, 1, 2), stride=2)  # (batch*n_features, 1, seq_len/2)
        
        lo = lo.reshape(batch, n_feat, -1)  # (batch, n_features, seq_len/2)
        hi = hi.reshape(batch, n_feat, -1)  # (batch, n_features, seq_len/2)
        
        return lo, hi
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.norm_input(x)  # (batch, n_features, input_len)
        
        # Multi-level wavelet decomposition
        coeffs = []
        current = x  # (batch, n_features, input_len)
        
        for level in range(self.n_levels):
            if current.shape[-1] < 2:
                break
            lo, hi = self.dwt(current)  # lo: (batch, n_features, seq_len/2), hi: same
            coeffs.append((lo, hi))
            current = lo  # (batch, n_features, seq_len/2)
        
        # Process each level with cross-feature attention
        level_outputs = []
        for i, (lo, hi) in enumerate(coeffs[:self.n_actual_levels]):
            # Combine lo and hi
            combined = lo + hi  # (batch, n_features, seq_len/2^i)
            
            # Transpose to (batch, seq_len, n_features) for linear projection
            combined = combined.permute(0, 2, 1)  # (batch, seq_len/2^i, n_features)
            
            # Project from n_features to hidden_dim (applied per timestep)
            proj = self.level_projs[i](combined)  # (batch, seq_len/2^i, hidden_dim)
            
            # Cross-feature attention (treat timesteps as sequence, hidden_dim as features)
            attn_out, _ = self.attentions[i](proj, proj, proj)  # (batch, seq_len/2^i, hidden_dim)
            
            # Global pooling over sequence dimension
            pooled = attn_out.mean(dim=1)  # (batch, hidden_dim)
            level_outputs.append(pooled)  # (batch, hidden_dim)
        
        # Fuse all levels
        fused = torch.cat(level_outputs, dim=-1)  # (batch, hidden_dim * n_actual_levels)
        fused = self.fusion(fused)  # (batch, hidden_dim)
        
        # Output projection to (batch, output_len * n_features)
        output = self.output_proj(fused)  # (batch, output_len * n_features)
        output = output.reshape(-1, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        return output
