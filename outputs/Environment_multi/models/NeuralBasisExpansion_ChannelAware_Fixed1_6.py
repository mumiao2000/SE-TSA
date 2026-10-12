import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralBasisExpansion_ChannelAware_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_basis': [16, 64, 256],
            'basis_dim': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3],
            'hidden_dim': [128, 512, 2048],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.n_basis = cfg.get('n_basis', 64)
        self.basis_dim = cfg.get('basis_dim', 256)
        self.n_heads = cfg.get('n_heads', 8)
        self.dropout = cfg.get('dropout', 0.3)
        self.hidden_dim = cfg.get('hidden_dim', 512)
        
        # Feature groups: target (1), historical stats (18), SVD embeddings (22)
        self.n_target = 1
        self.n_historical = 18
        self.n_svd = 22
        assert self.n_target + self.n_historical + self.n_svd == self.n_features, "Feature split must sum to n_features"
        
        self.group_sizes = [self.n_target, self.n_historical, self.n_svd]
        self.group_names = ['target', 'historical', 'svd']
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Group-specific learnable basis matrices: directly learn (output_len, n_basis) per group
        self.basis_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(self.output_len, self.n_basis) * 0.02)
            for _ in range(3)
        ])
        
        # Basis refinement MLPs: small network to transform raw basis
        self.basis_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.n_basis, self.basis_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.basis_dim, self.n_basis)
            ) for _ in range(3)
        ])
        
        # Input projection: encode time series to hidden representation
        self.input_projection = nn.Linear(self.n_features, self.hidden_dim)
        
        # Positional encoding for input sequence
        self.input_pos_embed = nn.Parameter(torch.randn(1, self.input_len, self.hidden_dim) * 0.02)
        
        # Multi-head self-attention for global pattern extraction
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Hierarchical coefficient prediction:
        # Level 1: Global coefficient predictor (shared across groups)
        self.global_coeff_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        
        # Level 2: Group-specific coefficient refinement
        self.group_coeff_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 4, self.n_basis * group_size)
            ) for group_size in self.group_sizes
        ])
        
        # Output scaling per group for stable training
        self.output_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, group_size))
            for group_size in self.group_sizes
        ])
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection
        h = self.input_projection(x)  # (batch, input_len, hidden_dim)
        h = h + self.input_pos_embed[:, :self.input_len, :]  # (batch, input_len, hidden_dim)
        
        # Self-attention for global pattern extraction
        h_attn, _ = self.attention(h, h, h)  # (batch, input_len, hidden_dim)
        h = self.norm1(h + self.dropout_layer(h_attn))  # (batch, input_len, hidden_dim)
        
        # FFN
        h_ffn = self.ffn(h)  # (batch, input_len, hidden_dim)
        h = self.norm2(h + self.dropout_layer(h_ffn))  # (batch, input_len, hidden_dim)
        
        # Global pooling for sequence representation
        h_global = h.mean(dim=1)  # (batch, hidden_dim)
        
        # Global coefficient features
        global_features = self.global_coeff_net(h_global)  # (batch, hidden_dim // 2)
        
        # Predict coefficients and generate outputs for each group
        outputs = []
        for i, (group_size, coeff_net, basis_matrix, basis_mlp, scale) in enumerate(
            zip(self.group_sizes, self.group_coeff_nets, self.basis_matrices, self.basis_mlps, self.output_scales)
        ):
            # Group-specific coefficient prediction
            coeffs = coeff_net(global_features)  # (batch, n_basis * group_size)
            coeffs = coeffs.view(batch_size, self.n_basis, group_size)  # (batch, n_basis, group_size)
            
            # Generate basis matrix: raw basis -> MLP refinement -> (output_len, n_basis)
            basis_refined = basis_mlp(basis_matrix)  # (output_len, n_basis)
            
            # Expand basis to all features in group: einsum 'on,bnf->bof'
            # basis_refined: (output_len, n_basis), coeffs: (batch, n_basis, group_size)
            out_g = torch.einsum('on,bnf->bof', basis_refined, coeffs)  # (batch, output_len, group_size)
            
            # Apply learnable scale
            out_g = out_g * scale  # (batch, output_len, group_size)
            
            outputs.append(out_g)
        
        # Concatenate group outputs
        out = torch.cat(outputs, dim=-1)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

