import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttentionV3_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_groups': [4, 8, 16],
            'dropout': [0.0, 0.3],
            'n_scales': [2, 4],
            'temporal_ratio': [2, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_groups = self.config.get('n_groups', 8)
        self.dropout = self.config.get('dropout', 0.3)
        self.n_scales = self.config.get('n_scales', 3)
        self.temporal_ratio = self.config.get('temporal_ratio', 4)
        
        # Ensure n_groups divides n_features
        self.n_groups = min(self.n_groups, self.n_features)
        while self.n_features % self.n_groups != 0 and self.n_groups > 1:
            self.n_groups -= 1
        self.group_size = self.n_features // self.n_groups
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Learnable multi-scale decomposition filters
        self.decomp_scales = nn.ModuleList()
        for i in range(self.n_scales):
            kernel_size = max(3, int(self.input_len / (2 ** i)) | 1)  # odd kernel
            padding = kernel_size // 2
            self.decomp_scales.append(
                nn.Conv1d(self.n_features, self.n_features, kernel_size, padding=padding, groups=self.n_features, bias=False)
            )
        self.decomp_gate = nn.Linear(self.n_features * self.n_scales, self.n_scales)
        
        # Input projection: maps input_len to d_model for each feature
        self.input_proj = nn.Linear(self.input_len, self.d_model)
        
        # Adaptive channel grouping - learnable assignment
        self.group_assign_logits = nn.Parameter(torch.randn(self.n_features, self.n_groups) * 0.1)
        
        # Hierarchical temporal attention: fine and coarse
        self.fine_attn = nn.MultiheadAttention(self.d_model, num_heads=4, dropout=self.dropout, batch_first=True)
        
        # Coarse resolution: project down, pool n_features, attend, upsample back
        self.coarse_down_proj = nn.Linear(self.d_model, self.d_model // self.temporal_ratio)
        coarse_dim = self.d_model // self.temporal_ratio
        self.coarse_pool = nn.AdaptiveAvgPool1d(max(1, coarse_dim))
        self.coarse_attn = nn.MultiheadAttention(coarse_dim, num_heads=2, dropout=self.dropout, batch_first=True)
        self.coarse_upsample = nn.Linear(coarse_dim, self.d_model)
        
        # Cross-channel attention per group
        self.group_queries = nn.Parameter(torch.randn(self.n_groups, self.d_model // 2))
        self.group_keys = nn.Linear(self.d_model, self.d_model // 2)
        self.group_values = nn.Linear(self.d_model, self.d_model // 2)
        self.group_out_proj = nn.Linear(self.d_model // 2, self.d_model)
        
        # Channel-dependent MLP projections
        self.channel_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * 2, self.d_model),
                nn.Dropout(self.dropout)
            ) for _ in range(self.n_groups)
        ])
        
        # Output projection: d_model -> output_len for each feature
        self.output_proj = nn.Linear(self.d_model, self.output_len)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model // self.temporal_ratio)
        self.layer_norm3 = nn.LayerNorm(self.d_model)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Multi-scale decomposition with learnable gating
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        decomp_outputs = []
        for conv in self.decomp_scales:
            decomp_out = conv(x_perm)  # (batch, n_features, input_len)
            decomp_outputs.append(decomp_out.permute(0, 2, 1))  # (batch, input_len, n_features)
        
        # Gated combination of scales
        decomp_stack = torch.stack(decomp_outputs, dim=-1)  # (batch, input_len, n_features, n_scales)
        decomp_flat = decomp_stack.reshape(batch_size, self.input_len, -1)  # (batch, input_len, n_features * n_scales)
        gate_weights = F.softmax(self.decomp_gate(decomp_flat), dim=-1)  # (batch, input_len, n_scales)
        gate_weights = gate_weights.unsqueeze(2)  # (batch, input_len, 1, n_scales)
        x_decomp = (decomp_stack * gate_weights).sum(dim=-1)  # (batch, input_len, n_features)
        
        # Residual: original minus decomposition
        x_residual = x - x_decomp  # (batch, input_len, n_features)
        
        # Combine for processing
        x_combined = x_decomp + 0.5 * x_residual  # (batch, input_len, n_features)
        
        # Input projection: (batch, n_features, input_len) -> (batch, n_features, d_model)
        x_feat = x_combined.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_proj = self.input_proj(x_feat)  # (batch, n_features, d_model)
        
        # Hierarchical attention: fine-grained
        x_fine = x_proj  # (batch, n_features, d_model)
        x_fine = self.layer_norm1(x_fine)
        fine_attn_out, _ = self.fine_attn(x_fine, x_fine, x_fine)  # (batch, n_features, d_model)
        x_fine = x_fine + self.dropout_layer(fine_attn_out)  # (batch, n_features, d_model)
        
        # Coarse-grained attention: project down, pool n_features dimension, attend, upsample
        x_coarse = self.coarse_down_proj(x_fine)  # (batch, n_features, d_model//temporal_ratio)
        x_coarse = x_coarse.permute(0, 2, 1)  # (batch, d_model//temporal_ratio, n_features)
        x_coarse_pooled = self.coarse_pool(x_coarse)  # (batch, d_model//temporal_ratio, d_model//temporal_ratio)
        x_coarse_pooled = x_coarse_pooled.permute(0, 2, 1)  # (batch, d_model//temporal_ratio, d_model//temporal_ratio)
        x_coarse_pooled = self.layer_norm2(x_coarse_pooled)
        coarse_attn_out, _ = self.coarse_attn(x_coarse_pooled, x_coarse_pooled, x_coarse_pooled)  # (batch, d_model//temporal_ratio, d_model//temporal_ratio)
        x_coarse = x_coarse_pooled + coarse_attn_out  # (batch, d_model//temporal_ratio, d_model//temporal_ratio)
        x_coarse = x_coarse.mean(dim=1)  # (batch, d_model//temporal_ratio)
        x_coarse = self.coarse_upsample(x_coarse)  # (batch, d_model)
        x_coarse = x_coarse.unsqueeze(1).expand(-1, self.n_features, -1)  # (batch, n_features, d_model)
        
        # Combine hierarchical representations
        x_hier = x_fine + 0.5 * x_coarse  # (batch, n_features, d_model)
        x_hier = self.layer_norm3(x_hier)
        
        # Adaptive channel grouping with soft assignment
        group_assign = F.softmax(self.group_assign_logits, dim=1)  # (n_features, n_groups)
        
        # For each group, compute weighted aggregation and apply group-specific MLP
        group_outputs = []
        for g in range(self.n_groups):
            # Get weights for this group: (n_features,)
            weights = group_assign[:, g]  # (n_features,)
            weights = weights.view(1, -1, 1)  # (1, n_features, 1)
            
            # Soft-weighted features for this group
            group_features = x_hier * weights  # (batch, n_features, d_model)
            
            # Cross-channel attention with group query
            group_q = self.group_queries[g:g+1].unsqueeze(0).expand(batch_size, -1, -1)  # (batch, 1, d_model//2)
            group_k = self.group_keys(group_features)  # (batch, n_features, d_model//2)
            group_v = self.group_values(group_features)  # (batch, n_features, d_model//2)
            
            # Compute attention
            scores = torch.matmul(group_q, group_k.transpose(-2, -1)) / (group_q.shape[-1] ** 0.5)  # (batch, 1, n_features)
            scores = scores + torch.log(weights.squeeze(-1).unsqueeze(1) + 1e-10)  # bias by group membership
            attn_weights = F.softmax(scores, dim=-1)  # (batch, 1, n_features)
            group_context = torch.matmul(attn_weights, group_v)  # (batch, 1, d_model//2)
            group_context = group_context.squeeze(1)  # (batch, d_model//2)
            group_context = self.group_out_proj(group_context)  # (batch, d_model)
            
            # Apply group-specific MLP
            group_out = self.channel_mlps[g](group_context)  # (batch, d_model)
            group_outputs.append(group_out)
        
        # Stack group outputs and recombine with adaptive weights
        group_stack = torch.stack(group_outputs, dim=1)  # (batch, n_groups, d_model)
        
        # Reconstruct per-feature representation using group assignments
        feature_recon = torch.einsum('ng,bgd->bnd', group_assign, group_stack)  # (batch, n_features, d_model)
        
        # Residual connection
        feature_out = x_hier + feature_recon  # (batch, n_features, d_model)
        
        # Output projection per feature
        output = self.output_proj(feature_out)  # (batch, n_features, output_len)
        output = output.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Denormalize
        output = self.revin(output, mode='denorm')  # (batch, output_len, n_features)
        
        return output

