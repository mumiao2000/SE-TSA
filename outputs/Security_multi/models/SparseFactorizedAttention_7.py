import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SparseFactorizedAttention(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'local_window': [4, 16, 64],
            'global_stride': [8, 32, 128],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        d_model = self.config.get('d_model', 256)
        n_heads = self.config.get('n_heads', 8)
        n_layers = self.config.get('n_layers', 4)
        local_window = self.config.get('local_window', 16)
        global_stride = self.config.get('global_stride', 32)
        dropout = self.config.get('dropout', 0.1)
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.revin = RevIN(self.n_features, affine=True)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.local_window = local_window
        self.global_stride = global_stride
        
        # Per-channel projections (channel-independent)
        self.input_proj = nn.Linear(self.n_features, d_model * self.n_features)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'q_proj': nn.Linear(d_model, d_model),
                'k_proj': nn.Linear(d_model, d_model),
                'v_proj': nn.Linear(d_model, d_model),
                'out_proj': nn.Linear(d_model, d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'dropout': nn.Dropout(dropout)
            }))
        
        # Output: project each channel's representation back
        self.output_proj = nn.Linear(d_model, 1)
        
        # Direct time mapping
        self.time_mapper = nn.Linear(input_len, output_len)

    def _sparse_attention(self, q, k, v, local_window, global_stride, mask_value=-1e9):
        # q, k, v: (batch, n_heads, seq_len, head_dim)
        batch, n_heads, seq_len, head_dim = q.shape
        
        # Compute full attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # (batch, n_heads, seq_len, seq_len)
        
        # Create sparse mask: local window + global strided
        mask = torch.full((seq_len, seq_len), mask_value, device=q.device, dtype=q.dtype)
        
        # Local attention: each position attends to nearby positions
        for i in range(seq_len):
            start = max(0, i - local_window // 2)
            end = min(seq_len, i + local_window // 2 + 1)
            mask[i, start:end] = 0.0
        
        # Global attention: each position attends to strided positions
        for i in range(seq_len):
            global_indices = torch.arange(0, seq_len, global_stride, device=q.device)
            mask[i, global_indices] = 0.0
        
        # Also always attend to self
        mask = mask + torch.eye(seq_len, device=q.device, dtype=q.dtype) * (-mask_value)
        
        scores = scores + mask.unsqueeze(0).unsqueeze(0)  # (batch, n_heads, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_len, seq_len)
        
        out = torch.matmul(attn, v)  # (batch, n_heads, seq_len, head_dim)
        return out

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-independent processing: process each feature separately
        # Reshape to (batch * n_features, input_len, 1)
        batch_size = x.size(0)
        x_flat = x.permute(0, 2, 1).reshape(batch_size * self.n_features, self.input_len, 1)  # (batch*n_features, input_len, 1)
        
        # Project to d_model for each channel
        x_proj = self.input_proj(x).reshape(batch_size, self.input_len, self.n_features, self.d_model)  # (batch, input_len, n_features, d_model)
        x_proj = x_proj.permute(0, 2, 1, 3).reshape(batch_size * self.n_features, self.input_len, self.d_model)  # (batch*n_features, input_len, d_model)
        
        # Apply attention layers
        for layer in self.layers:
            residual = x_proj  # (batch*n_features, input_len, d_model)
            
            # Self-attention
            q = layer['q_proj'](x_proj)  # (batch*n_features, input_len, d_model)
            k = layer['k_proj'](x_proj)  # (batch*n_features, input_len, d_model)
            v = layer['v_proj'](x_proj)  # (batch*n_features, input_len, d_model)
            
            # Reshape for multi-head
            q = q.reshape(batch_size * self.n_features, self.input_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch*n_features, n_heads, input_len, head_dim)
            k = k.reshape(batch_size * self.n_features, self.input_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch*n_features, n_heads, input_len, head_dim)
            v = v.reshape(batch_size * self.n_features, self.input_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch*n_features, n_heads, input_len, head_dim)
            
            # Sparse attention
            attn_out = self._sparse_attention(q, k, v, self.local_window, self.global_stride)  # (batch*n_features, n_heads, input_len, head_dim)
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(batch_size * self.n_features, self.input_len, self.d_model)  # (batch*n_features, input_len, d_model)
            
            attn_out = layer['out_proj'](attn_out)  # (batch*n_features, input_len, d_model)
            attn_out = layer['dropout'](attn_out)  # (batch*n_features, input_len, d_model)
            x_proj = layer['norm1'](attn_out + residual)  # (batch*n_features, input_len, d_model)
            
            # FFN
            residual = x_proj  # (batch*n_features, input_len, d_model)
            ffn_out = layer['ffn'](x_proj)  # (batch*n_features, input_len, d_model)
            x_proj = layer['norm2'](ffn_out + residual)  # (batch*n_features, input_len, d_model)
        
        # Output projection per channel
        out = self.output_proj(x_proj)  # (batch*n_features, input_len, 1)
        out = out.reshape(batch_size, self.n_features, self.input_len).permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Map to output length
        out = out.permute(0, 2, 1)  # (batch, n_features, input_len)
        out = self.time_mapper(out)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
