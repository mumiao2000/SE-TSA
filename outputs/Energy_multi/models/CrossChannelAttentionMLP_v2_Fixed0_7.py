import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttentionMLP_v2_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'temporal_dim': [64, 256, 1024],
            'n_heads': [2, 8, 32],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.hidden_dim = cfg.get('hidden_dim', 512)
        self.n_layers = cfg.get('n_layers', 4)
        self.dropout = cfg.get('dropout', 0.1)
        self.temporal_dim = cfg.get('temporal_dim', 256)
        self.n_heads = cfg.get('n_heads', 8)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to hidden dim
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Hierarchical temporal MLPs: full resolution and half resolution
        self.temporal_mlps_full = nn.ModuleList()
        self.temporal_mlps_half = nn.ModuleList()
        self.gates = nn.ModuleList()
        
        half_len = self.input_len // 2
        
        for _ in range(self.n_layers):
            # Full resolution temporal MLP
            self.temporal_mlps_full.append(nn.Sequential(
                nn.Linear(self.input_len, self.temporal_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.temporal_dim, self.input_len),
                nn.Dropout(self.dropout),
            ))
            
            # Half resolution temporal MLP (with strided pooling)
            self.temporal_mlps_half.append(nn.Sequential(
                nn.Linear(half_len, self.temporal_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.temporal_dim // 2, half_len),
                nn.Dropout(self.dropout),
            ))
            
            # Learnable gate to fuse full and half resolution
            self.gates.append(nn.Linear(self.hidden_dim * 2, self.hidden_dim))
        
        # Cross-channel attention: operates on n_features dimension (O(F^2) complexity)
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=self.n_features,
            num_heads=min(self.n_heads, self.n_features),  # ensure n_heads <= n_features
            dropout=self.dropout,
            batch_first=True,
        )
        
        # Projections between hidden_dim and n_features for attention bottleneck
        self.hidden_to_features = nn.Linear(self.hidden_dim, self.n_features)
        self.features_to_hidden = nn.Linear(self.n_features, self.hidden_dim)
        
        # Pre-norm layer norms for stability
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)
        
        # Channel-wise FFN after attention
        self.channel_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Dropout(self.dropout),
        )
        
        # Strided pooling for half resolution
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Upsample for half resolution back to full
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        
        # Output projection: hidden_dim -> n_features
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        
        # Horizon-aware position embeddings for output modulation
        self.horizon_embed = nn.Parameter(torch.randn(1, self.output_len, self.n_features) * 0.02)
        
        # Learnable horizon scaling: per-position scale and bias
        self.horizon_scale = nn.Parameter(torch.ones(1, self.output_len, self.n_features))
        self.horizon_bias = nn.Parameter(torch.zeros(1, self.output_len, self.n_features))
        
        # Final temporal mapping: input_len -> output_len
        self.temporal_map = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden dimension
        h = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Hierarchical temporal processing with residual gating
        for i in range(self.n_layers):
            h_res = h  # (batch, input_len, hidden_dim)
            
            # Full resolution temporal MLP: operate on time dimension
            h_full = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
            h_full = self.temporal_mlps_full[i](h_full)  # (batch, hidden_dim, input_len)
            h_full = h_full.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
            
            # Half resolution temporal MLP
            h_half = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
            h_half = self.pool(h_half)  # (batch, hidden_dim, input_len//2)
            h_half = self.temporal_mlps_half[i](h_half)  # (batch, hidden_dim, input_len//2)
            h_half = self.upsample(h_half)  # (batch, hidden_dim, input_len)
            # Handle odd length
            if h_half.shape[-1] != h_full.shape[1]:
                h_half = h_half[:, :, :h_full.shape[1]]  # (batch, hidden_dim, input_len)
            h_half = h_half.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
            
            # Concatenate and gate
            h_concat = torch.cat([h_full, h_half], dim=-1)  # (batch, input_len, hidden_dim*2)
            gate = torch.sigmoid(self.gates[i](h_concat))  # (batch, input_len, hidden_dim)
            h = gate * h_full + (1 - gate) * h_half  # (batch, input_len, hidden_dim)
            h = h + h_res  # residual connection  # (batch, input_len, hidden_dim)
        
        # Cross-channel attention: project to n_features space, attend across channels, project back
        h_norm = self.norm1(h)  # (batch, input_len, hidden_dim)
        
        # Project to feature space for cross-channel attention
        h_features = self.hidden_to_features(h_norm)  # (batch, input_len, n_features)
        
        # Apply cross-channel attention: attend across n_features dimension
        # Treat input_len as batch-like: process all time steps together
        # (batch, input_len, n_features) -> self-attention across n_features for each batch and time
        h_attn, _ = self.channel_attn(h_features, h_features, h_features)  # (batch, input_len, n_features)
        
        # Project back to hidden_dim
        h_attn = self.features_to_hidden(h_attn)  # (batch, input_len, hidden_dim)
        
        # Residual connection
        h = h + h_attn  # (batch, input_len, hidden_dim)
        
        # FFN
        h_norm2 = self.norm2(h)  # (batch, input_len, hidden_dim)
        h_ffn = self.channel_ffn(h_norm2)  # (batch, input_len, hidden_dim)
        h = h + h_ffn  # (batch, input_len, hidden_dim)
        
        # Final norm
        h = self.norm3(h)  # (batch, input_len, hidden_dim)
        
        # Project to output features
        out = self.output_proj(h)  # (batch, input_len, n_features)
        
        # Temporal mapping: input_len -> output_len
        out = out.permute(0, 2, 1)  # (batch, n_features, input_len)
        out = self.temporal_map(out)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Horizon-aware modulation
        out = out * self.horizon_scale + self.horizon_bias  # (batch, output_len, n_features)
        out = out + self.horizon_embed  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

