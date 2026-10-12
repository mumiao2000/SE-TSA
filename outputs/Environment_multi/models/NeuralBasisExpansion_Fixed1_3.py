import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralBasisExpansion_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_basis': [16, 64, 256],
            'basis_dim': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.n_basis = self.config.get('n_basis', 64)
        self.basis_dim = self.config.get('basis_dim', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Basis function network: position -> basis value
        # Shared across features, but we have n_basis different functions
        self.basis_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.basis_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.basis_dim, self.basis_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.basis_dim, 1)
            ) for _ in range(self.n_basis)
        ])
        
        # Coefficient network: maps input sequence to basis coefficients
        # Use multi-head attention to pool input
        self.input_proj = nn.Linear(self.n_features, self.basis_dim)
        self.attention = nn.MultiheadAttention(self.basis_dim, self.n_heads, dropout=self.dropout, batch_first=True)
        self.coeff_net = nn.Sequential(
            nn.Linear(self.basis_dim, self.basis_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.basis_dim, self.n_basis * self.n_features)
        )
        
        # Output mixing
        self.output_mix = nn.Linear(self.n_features, self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project input
        h = self.input_proj(x)  # (batch, input_len, basis_dim)
        
        # Self-attention pooling
        h_att, _ = self.attention(h, h, h)  # (batch, input_len, basis_dim)
        h_pooled = h_att.mean(dim=1)  # (batch, basis_dim)
        
        # Compute coefficients: (batch, n_basis * n_features)
        coeffs = self.coeff_net(h_pooled)  # (batch, n_basis * n_features)
        coeffs = coeffs.reshape(-1, self.n_basis, self.n_features)  # (batch, n_basis, n_features)
        
        # Evaluate basis functions at output positions
        positions = torch.linspace(0, 1, self.output_len, device=x.device).unsqueeze(-1)  # (output_len, 1)
        basis_values = []
        for i in range(self.n_basis):
            b = self.basis_nets[i](positions)  # (output_len, 1)
            basis_values.append(b)
        basis_matrix = torch.cat(basis_values, dim=-1)  # (output_len, n_basis)
        
        # Combine: basis_matrix: (output_len, n_basis), coeffs: (batch, n_basis, n_features)
        # We want: (batch, output_len, n_features)
        # Use 'on' for basis_matrix where o=output_len, n=n_basis
        # Use 'bnf' for coeffs where b=batch, n=n_basis, f=n_features
        out = torch.einsum('on,bnf->bof', basis_matrix, coeffs)  # (batch, output_len, n_features)
        
        # Feature mixing
        out = self.output_mix(out)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

