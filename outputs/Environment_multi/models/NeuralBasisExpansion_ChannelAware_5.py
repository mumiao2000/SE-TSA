import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralBasisExpansion_ChannelAware(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_basis': [16, 64, 256],
            'basis_dim': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3],
            'channel_hidden': [64, 256, 512],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.n_basis = cfg.get('n_basis', 64)
        self.basis_dim = cfg.get('basis_dim', 256)
        self.n_heads = cfg.get('n_heads', 8)
        self.dropout = cfg.get('dropout', 0.1)
        self.channel_hidden = cfg.get('channel_hidden', 256)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Channel-dependent learnable basis functions: each feature has its own set of basis functions
        # Shape: (n_features, n_basis, basis_dim) -> projected to (n_features, n_basis, output_len)
        self.basis_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.basis_dim, self.basis_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.basis_dim // 2, self.output_len)
            ) for _ in range(self.n_features)
        ])
        
        # Learnable basis parameters per feature
        self.basis_params = nn.Parameter(torch.randn(self.n_features, self.n_basis, self.basis_dim))  # (n_features, n_basis, basis_dim)
        
        # Input projection: encode input sequence per feature
        self.input_encoder = nn.Linear(self.input_len, self.channel_hidden)  # (batch, n_features, input_len) -> (batch, n_features, channel_hidden)
        
        # Cross-channel attention: allow features to exchange information
        self.cross_channel_attn = nn.MultiheadAttention(
            embed_dim=self.channel_hidden,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Layer norm for attention
        self.attn_norm = nn.LayerNorm(self.channel_hidden)
        
        # Hierarchical coefficient predictor
        # First: feature-specific MLP to get initial coefficients
        self.coeff_mlp = nn.Sequential(
            nn.Linear(self.channel_hidden, self.channel_hidden // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.channel_hidden // 2, self.n_basis)
        )  # (batch, n_features, channel_hidden) -> (batch, n_features, n_basis)
        
        # Second: cross-channel mixing for refined coefficients
        self.coeff_mixer = nn.Sequential(
            nn.Linear(self.n_basis, self.n_basis * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_basis * 2, self.n_basis)
        )  # (batch, n_features, n_basis) -> (batch, n_features, n_basis)
        
        # Optional: global feature interaction for coefficient refinement
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Linear(self.n_features, self.n_features)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for feature-wise processing
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Encode input sequence per feature
        feat_encoded = self.input_encoder(x)  # (batch, n_features, channel_hidden)
        feat_encoded = self.dropout_layer(feat_encoded)
        
        # Cross-channel attention: features attend to each other
        attn_out, _ = self.cross_channel_attn(feat_encoded, feat_encoded, feat_encoded)  # (batch, n_features, channel_hidden)
        attn_out = self.attn_norm(feat_encoded + attn_out)  # residual + norm, (batch, n_features, channel_hidden)
        attn_out = self.dropout_layer(attn_out)
        
        # Predict basis coefficients per feature
        coeffs = self.coeff_mlp(attn_out)  # (batch, n_features, n_basis)
        coeffs = self.coeff_mixer(coeffs)  # (batch, n_features, n_basis)
        
        # Add global feature interaction
        # Pool across features, project, and add as residual
        global_feat = self.global_pool(coeffs.transpose(1, 2)).squeeze(-1)  # (batch, n_basis)
        global_feat = self.global_proj(global_feat.unsqueeze(-1).expand(-1, -1, self.n_features))  # (batch, n_basis, n_features)
        global_feat = global_feat.transpose(1, 2)  # (batch, n_features, n_basis)
        coeffs = coeffs + 0.1 * global_feat  # soft global mixing, (batch, n_features, n_basis)
        
        # Generate channel-dependent basis functions
        # For each feature, generate n_basis basis functions of length output_len
        basis_outputs = []
        for f in range(self.n_features):
            # basis_params[f]: (n_basis, basis_dim)
            basis_in = self.basis_params[f]  # (n_basis, basis_dim)
            basis_out = self.basis_generators[f](basis_in)  # (n_basis, output_len)
            basis_outputs.append(basis_out)
        
        # Stack: (n_features, n_basis, output_len)
        basis_matrix = torch.stack(basis_outputs, dim=0)  # (n_features, n_basis, output_len)
        
        # Combine coefficients with basis functions per feature
        # coeffs: (batch, n_features, n_basis)
        # basis_matrix: (n_features, n_basis, output_len)
        # We want: (batch, n_features, output_len)
        
        # Einsum: batch feature basis, feature basis output -> batch feature output
        out = torch.einsum('bfn,fn o->bfo', coeffs, basis_matrix)  # (batch, n_features, output_len)
        
        # Transpose back to standard format
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

