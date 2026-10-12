import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttentionV2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_groups': [4, 8, 16],
            'dropout': [0.0, 0.3],
            'decomp_kernel': [3, 13, 25],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 256)
        self.n_groups = cfg.get('n_groups', 8)
        self.dropout = cfg.get('dropout', 0.1)
        self.decomp_kernel = cfg.get('decomp_kernel', 13)
        
        # Ensure n_groups divides n_features
        self.n_groups = min(self.n_groups, self.n_features)
        while self.n_features % self.n_groups != 0 and self.n_groups > 1:
            self.n_groups -= 1
        self.features_per_group = self.n_features // self.n_groups
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: per-group linear
        self.input_proj = nn.ModuleList([
            nn.Linear(self.input_len, self.d_model)
            for _ in range(self.n_groups)
        ])  # (batch, features_per_group, d_model) each
        
        # Cross-channel attention within each group
        self.group_attn = nn.ModuleList([
            nn.MultiheadAttention(self.d_model, num_heads=4, dropout=self.dropout, batch_first=True)
            for _ in range(self.n_groups)
        ])  # (batch, features_per_group, d_model)
        
        self.attn_norm = nn.ModuleList([
            nn.LayerNorm(self.d_model)
            for _ in range(self.n_groups)
        ])  # (batch, features_per_group, d_model)
        
        # Channel-dependent MLP for each group
        self.group_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model // 2, self.output_len)
            )
            for _ in range(self.n_groups)
        ])  # (batch, features_per_group, output_len)
        
        # Residual decomposition branch (DLinear-style)
        # Moving average for trend extraction
        self.decomp = nn.AvgPool1d(kernel_size=self.decomp_kernel, stride=1, padding=self.decomp_kernel // 2)
        
        # Trend and seasonal projections
        self.trend_proj = nn.Linear(self.input_len, self.output_len)
        self.seasonal_proj = nn.Linear(self.input_len, self.output_len)
        
        # Residual weight (learnable blend)
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # --- Decomposition branch ---
        # Transpose for 1D ops: (batch, n_features, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Trend component via moving average
        trend = self.decomp(x_perm)  # (batch, n_features, input_len)
        # Handle potential length mismatch from padding
        if trend.shape[-1] != x_perm.shape[-1]:
            trend = F.interpolate(trend, size=x_perm.shape[-1], mode='linear', align_corners=False)  # (batch, n_features, input_len)
        seasonal = x_perm - trend  # (batch, n_features, input_len)
        
        # Project decomposition components
        trend_out = self.trend_proj(trend)  # (batch, n_features, output_len)
        seasonal_out = self.seasonal_proj(seasonal)  # (batch, n_features, output_len)
        decomp_out = (trend_out + seasonal_out).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # --- Attention branch ---
        # Split into groups: (batch, input_len, n_features) -> list of (batch, input_len, features_per_group)
        group_inputs = torch.split(x, self.features_per_group, dim=-1)  # each: (batch, input_len, features_per_group)
        
        group_outputs = []
        for g in range(self.n_groups):
            # (batch, features_per_group, input_len)
            g_input = group_inputs[g].permute(0, 2, 1)  # (batch, features_per_group, input_len)
            
            # Project to d_model
            g_proj = self.input_proj[g](g_input)  # (batch, features_per_group, d_model)
            
            # Self-attention across channels in group
            g_attn, _ = self.group_attn[g](g_proj, g_proj, g_proj)  # (batch, features_per_group, d_model)
            g_attn = self.attn_norm[g](g_proj + g_attn)  # (batch, features_per_group, d_model), residual
            
            # Channel-dependent MLP
            g_out = self.group_mlps[g](g_attn)  # (batch, features_per_group, output_len)
            group_outputs.append(g_out)
        
        # Concatenate groups: (batch, n_features, output_len)
        attn_out = torch.cat(group_outputs, dim=1)  # (batch, n_features, output_len)
        attn_out = attn_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Blend attention and decomposition with learned weight
        weight = torch.sigmoid(self.residual_weight)  # scalar
        out = weight * attn_out + (1 - weight) * decomp_out  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
