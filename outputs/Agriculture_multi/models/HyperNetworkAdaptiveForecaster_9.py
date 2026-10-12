import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperNetworkAdaptiveForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hyper_dim': [32, 128, 512],
            'embed_dim': [16, 64, 256],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hyper_dim = self.config.get('hyper_dim', 128)
        self.embed_dim = self.config.get('embed_dim', 64)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input statistics encoder (small, fixed)
        self.stats_encoder = nn.Sequential(
            nn.Linear(n_features * 4, self.embed_dim),  # mean, std, min, max per feature
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # Hypernetwork: generates weights for main network
        # Generate a set of basis weights + mixing coefficients
        self.n_basis = 4  # number of weight basis elements
        
        # Basis weights (shared)
        self.basis_weights = nn.ParameterList()
        for i in range(self.n_layers):
            in_d = input_len if i == 0 else self.hyper_dim
            out_d = output_len if i == self.n_layers - 1 else self.hyper_dim
            self.basis_weights.append(nn.Parameter(torch.randn(self.n_basis, in_d, out_d) * 0.1))
            self.basis_weights.append(nn.Parameter(torch.randn(self.n_basis, out_d) * 0.1))
        
        # Hypernetwork generates mixing coefficients
        self.hyper_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.hyper_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hyper_dim, self.n_basis * self.n_layers * 2),
            nn.Softmax(dim=-1)  # softmax over basis dimension
        )
        
        # Feature-specific scaling (learned but small)
        self.feature_scale = nn.Parameter(torch.ones(n_features))
        self.feature_shift = nn.Parameter(torch.zeros(n_features))
        
        # Direct residual
        self.residual = nn.Linear(input_len, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Compute statistics per feature
        x_mean = x.mean(dim=1)  # (batch, n_features)
        x_std = x.std(dim=1) + 1e-5  # (batch, n_features)
        x_min = x.min(dim=1).values  # (batch, n_features)
        x_max = x.max(dim=1).values  # (batch, n_features)
        
        stats = torch.cat([x_mean, x_std, x_min, x_max], dim=-1)  # (batch, n_features * 4)
        
        # Encode statistics
        z = self.stats_encoder(stats)  # (batch, embed_dim)
        
        # Generate mixing coefficients
        mix_all = self.hyper_net(z)  # (batch, n_basis * n_layers * 2)
        mix_all = mix_all.view(-1, self.n_layers, 2, self.n_basis)  # (batch, n_layers, 2, n_basis)
        
        # Process per sample with generated weights
        # For efficiency, process all samples together using einsum
        h = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        for layer_idx in range(self.n_layers):
            # Get mixing for this layer
            mix_w = mix_all[:, layer_idx, 0, :]  # (batch, n_basis)
            mix_b = mix_all[:, layer_idx, 1, :]  # (batch, n_basis)
            
            # Generate weights: weighted sum of basis
            # basis_weights[layer_idx*2]: (n_basis, in_d, out_d)
            W = torch.einsum('bn,nio->bio', mix_w, self.basis_weights[layer_idx * 2])  # (batch, in_d, out_d)
            b = torch.einsum('bn,no->bo', mix_b, self.basis_weights[layer_idx * 2 + 1])  # (batch, out_d)
            
            # Apply linear layer with generated weights
            # h: (batch, n_features, in_d)
            # W: (batch, in_d, out_d)
            h = torch.einsum('bni,bio->bno', h, W) + b.unsqueeze(1)  # (batch, n_features, out_d)
            
            if layer_idx < self.n_layers - 1:
                h = F.gelu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # h: (batch, n_features, output_len)
        out = h.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Apply feature-specific scaling
        scale = self.feature_scale.unsqueeze(0).unsqueeze(0)  # (1, 1, n_features)
        shift = self.feature_shift.unsqueeze(0).unsqueeze(0)  # (1, 1, n_features)
        out = out * scale + shift
        
        # Add residual
        res = self.residual(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, output_len, n_features)
        out = out + res
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

