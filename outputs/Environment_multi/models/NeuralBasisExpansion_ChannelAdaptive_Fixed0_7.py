import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralBasisExpansion_ChannelAdaptive_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_global_basis': [16, 64, 256],
            'n_channel_basis': [4, 16, 64],
            'basis_dim': [64, 256, 1024],
            'coeff_hidden_dim': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3],
            'temporal_pool_levels': [2, 3, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_global_basis = self.config.get('n_global_basis', 64)
        self.n_channel_basis = self.config.get('n_channel_basis', 16)
        self.n_basis = self.n_global_basis + self.n_channel_basis  # total bases per channel
        self.basis_dim = self.config.get('basis_dim', 256)
        self.coeff_hidden_dim = self.config.get('coeff_hidden_dim', 512)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout_rate = self.config.get('dropout', 0.3)
        self.temporal_pool_levels = self.config.get('temporal_pool_levels', 3)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Global shared basis functions (learnable, output_len x basis_dim)
        self.global_basis = nn.Parameter(torch.randn(self.output_len, self.basis_dim) * 0.02)
        
        # Channel-specific residual basis functions
        self.channel_basis = nn.Parameter(
            torch.randn(self.n_features, self.n_channel_basis, self.output_len, self.basis_dim) * 0.02
        )
        
        # Basis projection network: from basis_dim to scalar output per time step
        self.basis_proj = nn.Sequential(
            nn.Linear(self.basis_dim, self.basis_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.basis_dim // 2, 1)
        )
        
        # Multi-scale temporal encoder for coefficient prediction
        # All layers now output coeff_hidden_dim for consistent dimensions
        self.temporal_convs = nn.ModuleList()
        for level in range(self.temporal_pool_levels):
            kernel_size = 3
            dilation = 2 ** level
            padding = (kernel_size - 1) * dilation // 2
            in_channels = self.n_features if level == 0 else self.coeff_hidden_dim
            self.temporal_convs.append(
                nn.Conv1d(
                    in_channels,
                    self.coeff_hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    groups=1
                )
            )
        
        # Hierarchical pooling for multi-scale representation
        self.pool_weights = nn.Parameter(torch.ones(self.temporal_pool_levels))
        
        # Coefficient prediction network with channel-wise attention
        self.coeff_attn = nn.MultiheadAttention(
            embed_dim=self.coeff_hidden_dim,
            num_heads=self.n_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        self.coeff_norm1 = nn.LayerNorm(self.coeff_hidden_dim)
        self.coeff_norm2 = nn.LayerNorm(self.coeff_hidden_dim)
        
        self.coeff_ffn = nn.Sequential(
            nn.Linear(self.coeff_hidden_dim, self.coeff_hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.coeff_hidden_dim * 2, self.coeff_hidden_dim)
        )
        
        # Final coefficient projection: (batch, n_features, coeff_hidden_dim) -> (batch, n_features, n_basis)
        self.coeff_proj = nn.Sequential(
            nn.Linear(self.coeff_hidden_dim, self.coeff_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.coeff_hidden_dim // 2, self.n_basis)
        )
        
        # Channel-wise gating for combining global and channel-specific contributions
        self.channel_gate = nn.Sequential(
            nn.Linear(self.coeff_hidden_dim, self.n_features),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Multi-scale temporal encoding
        x_conv = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        multi_scale_features = []
        for i, conv in enumerate(self.temporal_convs):
            x_conv = F.gelu(conv(x_conv))  # (batch, coeff_hidden_dim, input_len)
            # Global average pooling over time
            pooled = x_conv.mean(dim=-1)  # (batch, coeff_hidden_dim)
            multi_scale_features.append(pooled)
        
        # Weighted combination of multi-scale features
        weights = F.softmax(self.pool_weights, dim=0)  # (temporal_pool_levels,)
        combined = sum(w * f for w, f in zip(weights, multi_scale_features))  # (batch, coeff_hidden_dim)
        
        # Expand to per-feature representation
        # (batch, coeff_hidden_dim) -> (batch, n_features, coeff_hidden_dim)
        feature_repr = combined.unsqueeze(1).expand(-1, self.n_features, -1)  # (batch, n_features, coeff_hidden_dim)
        
        # Channel-wise self-attention for feature interaction
        attn_out, _ = self.coeff_attn(feature_repr, feature_repr, feature_repr)  # (batch, n_features, coeff_hidden_dim)
        feature_repr = self.coeff_norm1(feature_repr + self.dropout(attn_out))  # (batch, n_features, coeff_hidden_dim)
        
        # FFN
        ffn_out = self.coeff_ffn(feature_repr)  # (batch, n_features, coeff_hidden_dim)
        feature_repr = self.coeff_norm2(feature_repr + self.dropout(ffn_out))  # (batch, n_features, coeff_hidden_dim)
        
        # Predict coefficients: (batch, n_features, n_basis)
        coeffs = self.coeff_proj(feature_repr)  # (batch, n_features, n_basis)
        coeffs = coeffs.permute(0, 2, 1)  # (batch, n_basis, n_features)
        
        # Compute channel-specific basis functions
        # Global basis: (output_len, basis_dim) -> (output_len, n_global_basis, basis_dim)
        global_bases = self.global_basis.unsqueeze(1).expand(-1, self.n_global_basis, -1)  # (output_len, n_global_basis, basis_dim)
        
        # Channel-specific residual bases: (n_features, n_channel_basis, output_len, basis_dim)
        channel_bases = self.channel_basis  # (n_features, n_channel_basis, output_len, basis_dim)
        
        # Project bases to scalar values per time step
        # Global: (output_len, n_global_basis, basis_dim) -> (output_len, n_global_basis)
        global_proj = self.basis_proj(global_bases).squeeze(-1)  # (output_len, n_global_basis)
        
        # Channel-specific: (n_features, n_channel_basis, output_len, basis_dim) -> (n_features, n_channel_basis, output_len)
        channel_proj = self.basis_proj(channel_bases).squeeze(-1)  # (n_features, n_channel_basis, output_len)
        channel_proj = channel_proj.permute(2, 1, 0)  # (output_len, n_channel_basis, n_features)
        
        # Combine: global_proj is (output_len, n_global_basis), need to expand to all features
        global_expanded = global_proj.unsqueeze(-1).expand(-1, -1, self.n_features)  # (output_len, n_global_basis, n_features)
        
        # Concatenate global and channel-specific bases along basis dimension
        all_bases = torch.cat([global_expanded, channel_proj], dim=1)  # (output_len, n_basis, n_features)
        
        # Einsum: (output_len, n_basis, n_features) @ (batch, n_basis, n_features) -> (batch, output_len, n_features)
        # Using correct dimension labels: o=output_len, n=n_basis, f=n_features, b=batch
        out = torch.einsum('onf,bnf->bof', all_bases, coeffs)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out
