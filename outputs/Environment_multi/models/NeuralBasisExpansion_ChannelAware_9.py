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
            'hidden_dim': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3],
            'gate_dim': [32, 128, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.n_basis = cfg.get('n_basis', 64)
        self.basis_dim = cfg.get('basis_dim', 256)
        self.hidden_dim = cfg.get('hidden_dim', 512)
        self.n_heads = cfg.get('n_heads', 8)
        self.dropout_rate = cfg.get('dropout', 0.1)
        self.gate_dim = cfg.get('gate_dim', 128)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Channel-specific basis functions: each feature has its own basis set
        # basis_matrix shape will be (n_features, output_len, n_basis)
        self.basis_matrix = nn.Parameter(
            torch.randn(self.n_features, self.output_len, self.n_basis) * 0.02
        )
        
        # Per-feature basis refinement network
        self.basis_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.n_basis, self.basis_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.basis_dim, self.n_basis)
            ) for _ in range(self.n_features)
        ])
        
        # Input projection: per-feature encoding
        self.input_proj = nn.Linear(self.input_len, self.hidden_dim)
        
        # Per-feature temporal encoder
        self.temporal_encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=max(1, self.hidden_dim // 64),
                    dim_feedforward=self.hidden_dim * 2,
                    dropout=self.dropout_rate,
                    batch_first=True
                ),
                num_layers=1
            ) for _ in range(self.n_features)
        ])
        
        # Cross-channel fusion attention
        self.cross_channel_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Hierarchical coefficient prediction
        # Stage 1: per-feature coefficient initialization
        self.coef_init = nn.Linear(self.hidden_dim, self.n_basis)
        
        # Stage 2: cross-channel refined coefficients
        self.coef_refine = nn.Sequential(
            nn.Linear(self.n_basis, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.n_basis)
        )
        
        # Channel-wise gating for dynamic basis weighting
        self.gate_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.gate_dim),
            nn.GELU(),
            nn.Linear(self.gate_dim, self.n_basis),
            nn.Sigmoid()
        )
        
        # Output projection per feature
        self.output_proj = nn.Linear(self.n_basis, 1)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Per-feature processing: transpose to (batch, n_features, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Encode each feature independently
        feature_encodings = []
        for f in range(self.n_features):
            feat = x_perm[:, f:f+1, :]  # (batch, 1, input_len)
            feat = feat.squeeze(1)  # (batch, input_len)
            feat = self.input_proj(feat)  # (batch, hidden_dim)
            feat = feat.unsqueeze(1)  # (batch, 1, hidden_dim)
            # Temporal encoding for this feature
            feat = self.temporal_encoders[f](feat)  # (batch, 1, hidden_dim)
            feature_encodings.append(feat)
        
        # Stack: (batch, n_features, hidden_dim)
        feature_stack = torch.cat(feature_encodings, dim=1)  # (batch, n_features, hidden_dim)
        feature_stack = self.layer_norm(feature_stack)
        
        # Cross-channel attention for fusion
        fused, _ = self.cross_channel_attn(
            feature_stack, feature_stack, feature_stack
        )  # (batch, n_features, hidden_dim)
        fused = self.dropout(fused)
        fused = fused + feature_stack  # residual
        fused = self.layer_norm(fused)
        
        # Coefficient prediction per feature
        # Stage 1: initial coefficients
        coefs_init = self.coef_init(fused)  # (batch, n_features, n_basis)
        
        # Stage 2: refined coefficients with cross-channel info
        coefs_refined = self.coef_refine(coefs_init)  # (batch, n_features, n_basis)
        coefs = coefs_init + coefs_refined  # residual refinement
        
        # Channel-wise gating
        gates = self.gate_encoder(fused)  # (batch, n_features, n_basis)
        coefs = coefs * gates  # (batch, n_features, n_basis)
        
        # Prepare for einsum: need (batch, n_basis, n_features)
        coeffs = coefs.permute(0, 2, 1)  # (batch, n_basis, n_features)
        
        # Apply per-feature basis refinement
        refined_bases = []
        for f in range(self.n_features):
            basis_f = self.basis_matrix[f]  # (output_len, n_basis)
            # Refine through MLP
            basis_f_refined = self.basis_mlp[f](basis_f)  # (output_len, n_basis)
            refined_bases.append(basis_f_refined)
        
        # Stack to (n_features, output_len, n_basis)
        basis_stack = torch.stack(refined_bases, dim=0)  # (n_features, output_len, n_basis)
        
        # Einsum: for each feature, combine its basis with its coefficients
        # basis_stack: (n_features, output_len, n_basis)
        # coeffs: (batch, n_basis, n_features)
        # We want: (batch, output_len, n_features)
        # einsum: fnb,bnf->bof where f=feature, o=output, n=basis, b=batch
        out = torch.einsum('fon,bnf->bof', basis_stack, coeffs)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out
