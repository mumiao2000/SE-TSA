import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveBasisExpansion(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_basis': [16, 64, 256],
            'basis_dim': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3],
            'channel_groups': [1, 4, 41],  # 1=shared, 4=coarse grouping, 41=fully independent
            'coeff_hidden_dim': [64, 256, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.n_basis = self.config.get('n_basis', 64)
        self.basis_dim = self.config.get('basis_dim', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout = self.config.get('dropout', 0.0)
        self.channel_groups = self.config.get('channel_groups', 4)
        self.coeff_hidden_dim = self.config.get('coeff_hidden_dim', 256)
        
        # Ensure channel_groups divides n_features or use n_features for independent
        if self.n_features % self.channel_groups != 0:
            self.channel_groups = self.n_features  # fallback to per-channel
        self.features_per_group = self.n_features // self.channel_groups
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # === Channel-specific basis functions ===
        # Each group has its own set of basis functions
        # basis_functions: (channel_groups, n_basis, basis_dim)
        self.basis_functions = nn.Parameter(
            torch.randn(self.channel_groups, self.n_basis, self.basis_dim) * 0.02
        )
        
        # Basis-to-output projection (shared across groups, but applied per-group)
        # Each basis_dim vector projects to output_len
        self.basis_output_proj = nn.Linear(self.basis_dim, self.output_len, bias=False)
        
        # === Two-stage coefficient predictor ===
        
        # Stage 1: Global temporal encoding with multi-head attention
        self.input_proj = nn.Linear(self.n_features, self.basis_dim)
        self.attn_norm = nn.LayerNorm(self.basis_dim)
        self.attention = nn.MultiheadAttention(
            self.basis_dim, self.n_heads, dropout=self.dropout, batch_first=True
        )
        self.attn_dropout = nn.Dropout(self.dropout)
        
        # Stage 2: Per-channel-group coefficient refinement
        # Global context -> per-group coefficients
        self.global_to_group = nn.Sequential(
            nn.Linear(self.basis_dim, self.coeff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.coeff_hidden_dim, self.channel_groups * self.n_basis)
        )
        
        # Per-group refinement MLP (lightweight, shared parameters across groups)
        self.group_refinement = nn.Sequential(
            nn.Linear(self.n_basis, self.n_basis // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_basis // 2, self.n_basis)
        )
        
        # Optional: direct residual connection from input to output for stability
        self.residual_proj = nn.Linear(self.input_len, self.output_len)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.basis_functions)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.basis_output_proj.weight)
        for m in self.global_to_group.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.group_refinement.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store for residual
        x_input = x  # (batch, input_len, n_features)
        
        # === Stage 1: Global temporal attention ===
        # Project input to basis_dim
        x_proj = self.input_proj(x)  # (batch, input_len, basis_dim)
        
        # Self-attention over time dimension
        x_norm = self.attn_norm(x_proj)  # (batch, input_len, basis_dim)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)  # (batch, input_len, basis_dim)
        attn_out = self.attn_dropout(attn_out)  # (batch, input_len, basis_dim)
        
        # Global pooling over time
        global_context = attn_out.mean(dim=1)  # (batch, basis_dim)
        
        # === Stage 2: Per-group coefficient prediction ===
        # Predict raw coefficients for all groups
        raw_coeffs = self.global_to_group(global_context)  # (batch, channel_groups * n_basis)
        raw_coeffs = raw_coeffs.view(batch_size, self.channel_groups, self.n_basis)  # (batch, channel_groups, n_basis)
        
        # Per-group refinement (shared MLP applied independently to each group)
        # Reshape for group-wise processing: (batch * channel_groups, n_basis)
        coeffs_flat = raw_coeffs.view(batch_size * self.channel_groups, self.n_basis)  # (batch * channel_groups, n_basis)
        refined_flat = self.group_refinement(coeffs_flat)  # (batch * channel_groups, n_basis)
        refined_coeffs = refined_flat.view(batch_size, self.channel_groups, self.n_basis)  # (batch, channel_groups, n_basis)
        
        # Add residual from raw coefficients
        refined_coeffs = refined_coeffs + raw_coeffs  # (batch, channel_groups, n_basis)
        
        # === Basis expansion ===
        # Expand basis functions for each group: apply basis_output_proj to get (channel_groups, n_basis, output_len)
        basis_matrix = self.basis_output_proj(self.basis_functions)  # (channel_groups, n_basis, output_len)
        
        # For each group, compute output: einsum over basis dimension
        # basis_matrix: (channel_groups, n_basis, output_len)
        # refined_coeffs: (batch, channel_groups, n_basis)
        # We want: (batch, channel_groups, output_len)
        group_outputs = torch.einsum('gnf,bgn->bgf', basis_matrix, refined_coeffs)  # (batch, channel_groups, output_len)
        
        # Expand to per-feature outputs
        # Each feature in a group shares the same output
        # group_outputs: (batch, channel_groups, output_len)
        # We need: (batch, output_len, n_features)
        
        # Repeat each group's output for its features
        # (batch, channel_groups, output_len) -> (batch, output_len, channel_groups) -> expand
        group_outputs = group_outputs.permute(0, 2, 1)  # (batch, output_len, channel_groups)
        
        # Expand to all features
        output = group_outputs.repeat_interleave(self.features_per_group, dim=2)  # (batch, output_len, n_features)
        
        # === Residual connection ===
        # Simple linear projection from input time to output time
        residual = self.residual_proj(x_input.permute(0, 2, 1))  # (batch, n_features, output_len)
        residual = residual.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        output = output + residual * 0.1  # scaled residual
        
        # RevIN denormalization
        output = self.revin(output, mode='denorm')  # (batch, output_len, n_features)
        
        return output  # (batch, output_len, n_features)

