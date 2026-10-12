import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttentionMLPv2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'temporal_kernel_sizes': [[3, 5, 7], [3, 7, 11], [5, 9, 13]],
            'mlp_ratio': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.d_model = self.config.get('d_model', 512)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.temporal_kernel_sizes = self.config.get('temporal_kernel_sizes', [3, 5, 7])
        self.mlp_ratio = self.config.get('mlp_ratio', 4)
        
        # Ensure d_model is divisible by n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.d_model)
        
        # Multi-scale temporal convolutions (channel-independent)
        self.temporal_convs = nn.ModuleList()
        for k in self.temporal_kernel_sizes:
            padding = k // 2
            self.temporal_convs.append(
                nn.Conv1d(self.d_model, self.d_model // len(self.temporal_kernel_sizes), 
                         kernel_size=k, padding=padding, groups=1)
            )
        self.temporal_fusion = nn.Linear(
            self.d_model // len(self.temporal_kernel_sizes) * len(self.temporal_kernel_sizes),
            self.d_model
        )
        
        # Cross-channel attention blocks with channel gating
        self.attention_blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attention_blocks.append(
                CrossChannelAttentionBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.dropout,
                    n_features=self.n_features
                )
            )
        
        # Temporal MLP for forecasting
        self.temporal_mlp = nn.Sequential(
            nn.Linear(self.input_len, self.input_len * self.mlp_ratio // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.input_len * self.mlp_ratio // 2, self.output_len),
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.n_features)
        
        # Residual connection scaling
        self.residual_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode="norm")  # (batch, input_len, n_features)
        
        # Input projection: (batch, input_len, d_model)
        h = self.input_proj(x)  # (batch, input_len, d_model)
        
        # Multi-scale temporal convolutions
        h_conv = h.permute(0, 2, 1)  # (batch, d_model, input_len)
        conv_outputs = []
        for conv in self.temporal_convs:
            conv_out = conv(h_conv)  # (batch, d_model//K, input_len)
            conv_outputs.append(conv_out)
        h_multi = torch.cat(conv_outputs, dim=1)  # (batch, d_model, input_len)
        h_multi = h_multi.permute(0, 2, 1)  # (batch, input_len, d_model)
        h_multi = self.temporal_fusion(h_multi)  # (batch, input_len, d_model)
        
        # Residual connection with scaling
        h = h + self.residual_scale * h_multi  # (batch, input_len, d_model)
        
        # Cross-channel attention blocks
        for block in self.attention_blocks:
            h = block(h)  # (batch, input_len, d_model)
        
        # Temporal MLP: transpose to (batch, d_model, input_len) -> (batch, d_model, output_len)
        h = h.permute(0, 2, 1)  # (batch, d_model, input_len)
        h = self.temporal_mlp(h)  # (batch, d_model, output_len)
        h = h.permute(0, 2, 1)  # (batch, output_len, d_model)
        
        # Output projection
        out = self.output_proj(h)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode="denorm")  # (batch, output_len, n_features)
        return out


class CrossChannelAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout, n_features):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Cross-channel attention: attend across feature dimension
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Channel-wise gating (SE-style)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Will be applied on transposed tensor
            nn.Flatten(start_dim=-2),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        
        # Temporal MLP
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout)
        )
        
        # Residual scaling parameter
        self.scale1 = nn.Parameter(torch.ones(1))
        self.scale2 = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # x: (batch, input_len, d_model)
        batch, seq_len, d_model = x.shape
        
        # Cross-channel attention
        h = self.norm1(x)  # (batch, input_len, d_model)
        
        # For cross-channel attention, we treat d_model as the 'channel' dimension
        # and transpose to attend across it: (batch, d_model, input_len)
        h_t = h.permute(0, 2, 1)  # (batch, d_model, input_len)
        
        # Project and reshape for multi-head attention
        # We attend across the d_model dimension (treated as 'heads' grouping)
        q = self.q_proj(h).reshape(batch, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        k = self.k_proj(h).reshape(batch, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        v = self.v_proj(h).reshape(batch, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        
        # Attention: (batch, n_heads, seq_len, seq_len) -> this is temporal attention actually
        # Let me redesign: true cross-channel attention where we attend across d_model dimension
        
        # Reshape to attend across channels (d_model) for each time step
        q = self.q_proj(h).reshape(batch, seq_len, self.n_heads, self.head_dim).permute(0, 1, 2, 3)  # (batch, seq_len, n_heads, head_dim)
        k = self.k_proj(h).reshape(batch, seq_len, self.n_heads, self.head_dim).permute(0, 1, 2, 3)  # (batch, seq_len, n_heads, head_dim)
        v = self.v_proj(h).reshape(batch, seq_len, self.n_heads, self.head_dim).permute(0, 1, 2, 3)  # (batch, seq_len, n_heads, head_dim)
        
        # Cross-channel attention: attend across head_dim dimension for each (batch, seq_len, n_heads)
        # Actually, let's do proper cross-channel: treat each time step independently, attend across d_model
        q = q.reshape(batch * seq_len, self.n_heads, self.head_dim)  # (batch*seq_len, n_heads, head_dim)
        k = k.reshape(batch * seq_len, self.n_heads, self.head_dim)  # (batch*seq_len, n_heads, head_dim)
        v = v.reshape(batch * seq_len, self.n_heads, self.head_dim)  # (batch*seq_len, n_heads, head_dim)
        
        # Scaled dot-product attention across head_dim dimension
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch*seq_len, n_heads, n_heads)
        attn = F.softmax(scores, dim=-1)  # (batch*seq_len, n_heads, n_heads)
        attn = self.dropout1(attn)
        
        out = torch.matmul(attn, v)  # (batch*seq_len, n_heads, head_dim)
        out = out.reshape(batch, seq_len, self.n_heads, self.head_dim).reshape(batch, seq_len, d_model)  # (batch, seq_len, d_model)
        out = self.out_proj(out)  # (batch, seq_len, d_model)
        
        # Residual
        h = x + self.scale1 * out  # (batch, seq_len, d_model)
        
        # Channel-wise gating
        h_t = h.permute(0, 2, 1)  # (batch, d_model, seq_len)
        gate = self.channel_gate(h_t)  # (batch, d_model)
        gate = gate.unsqueeze(-1)  # (batch, d_model, 1)
        h = h + h * gate.permute(0, 2, 1)  # (batch, seq_len, d_model) - skip connection with gating
        
        # Temporal MLP
        h2 = self.norm2(h)  # (batch, seq_len, d_model)
        h2 = self.mlp(h2)  # (batch, seq_len, d_model)
        h = h + self.scale2 * h2  # (batch, seq_len, d_model)
        
        return h
