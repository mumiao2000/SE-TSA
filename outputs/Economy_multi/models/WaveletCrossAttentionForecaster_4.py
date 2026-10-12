import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class WaveletCrossAttentionForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_heads': [2, 8, 16],
            'n_levels': [1, 2, 3],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_levels = self.config.get('n_levels', 2)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        assert self.n_heads <= self.hidden_dim, "n_heads must be <= hidden_dim"
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Haar wavelet decomposition (manual, no padding issues)
        # We'll use simple averaging/differencing
        
        # Compute sizes at each level
        self.level_sizes = []
        size = input_len
        for _ in range(self.n_levels):
            size = size // 2
            self.level_sizes.append(size)
        
        # Projections for each wavelet level
        self.approx_projs = nn.ModuleList()
        self.detail_projs = nn.ModuleList()
        for size in self.level_sizes:
            self.approx_projs.append(nn.Linear(size, self.hidden_dim))
            self.detail_projs.append(nn.Linear(size, self.hidden_dim))
        
        # Cross-attention between scales
        self.cross_attn = nn.MultiheadAttention(
            self.hidden_dim, self.n_heads, 
            dropout=self.dropout_rate, batch_first=True
        )
        
        # Feature cross-attention
        self.feat_attn = nn.MultiheadAttention(
            self.hidden_dim, self.n_heads,
            dropout=self.dropout_rate, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # MLP for processing attended features
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * (self.n_levels * 2), self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, output_len)
        
    def haar_decompose(self, x, levels):
        # x: (batch, n_features, seq_len)
        approx = x
        coeffs = []
        for _ in range(levels):
            even = approx[:, :, 0::2]  # (batch, n_features, seq_len//2)
            odd = approx[:, :, 1::2]   # (batch, n_features, seq_len//2)
            approx = (even + odd) / 2.0  # (batch, n_features, seq_len//2)
            detail = (even - odd) / 2.0  # (batch, n_features, seq_len//2)
            coeffs.append((approx, detail))
        return coeffs
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Wavelet decomposition
        coeffs = self.haar_decompose(x, self.n_levels)  # list of (approx, detail) tuples
        
        # Project each coefficient to hidden dim
        level_reprs = []
        for i, (approx, detail) in enumerate(coeffs):
            approx_proj = self.approx_projs[i](approx)  # (batch, n_features, hidden_dim)
            detail_proj = self.detail_projs[i](detail)  # (batch, n_features, hidden_dim)
            level_reprs.append(approx_proj)
            level_reprs.append(detail_proj)
        
        # Stack all level representations: (batch, n_features, n_levels*2, hidden_dim)
        stacked = torch.stack(level_reprs, dim=2)  # (batch, n_features, n_scales, hidden_dim)
        n_scales = stacked.shape[2]
        
        # Cross-attention across scales for each feature
        scale_attended = []
        for f in range(self.n_features):
            feat_scales = stacked[:, f, :, :]  # (batch, n_scales, hidden_dim)
            attn_out, _ = self.cross_attn(
                feat_scales, feat_scales, feat_scales
            )  # (batch, n_scales, hidden_dim)
            attn_out = self.norm1(feat_scales + self.dropout(attn_out))  # (batch, n_scales, hidden_dim)
            scale_attended.append(attn_out)
        
        scale_attended = torch.stack(scale_attended, dim=1)  # (batch, n_features, n_scales, hidden_dim)
        
        # Pool across scales
        pooled = scale_attended.mean(dim=2)  # (batch, n_features, hidden_dim)
        
        # Cross-attention across features
        feat_transposed = pooled.permute(0, 2, 1)  # (batch, hidden_dim, n_features) - wait, need (batch, seq, feat)
        # Actually for attention: query is (batch, seq, hidden_dim)
        # We want to attend across features, so seq = n_features
        feat_pooled = pooled  # (batch, n_features, hidden_dim)
        feat_attn_out, _ = self.feat_attn(
            feat_pooled, feat_pooled, feat_pooled
        )  # (batch, n_features, hidden_dim)
        feat_attn_out = self.norm2(feat_pooled + self.dropout(feat_attn_out))  # (batch, n_features, hidden_dim)
        
        # Flatten scales and process
        scale_attended_flat = scale_attended.reshape(x.shape[0], self.n_features, -1)  # (batch, n_features, n_scales*hidden_dim)
        # But we want to use the feature-attended output
        combined = feat_attn_out  # (batch, n_features, hidden_dim)
        
        # Actually let's use the scale-attended flattened version for richer representation
        # Recompute with proper dimensions
        scale_attended_flat = torch.cat([s for s in scale_attended.unbind(dim=2)], dim=-1)  # (batch, n_features, n_scales*hidden_dim)
        
        processed = self.mlp(scale_attended_flat)  # (batch, n_features, hidden_dim)
        
        out = self.output_proj(processed)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

