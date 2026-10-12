import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierKANForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_bases': [8, 32, 128],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_bases = self.config.get('n_bases', 32)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Learnable Fourier frequencies
        self.frequencies = nn.Parameter(torch.randn(self.n_bases) * 0.1)  # (n_bases,)
        
        # KAN layers: each applies learnable basis transformation
        self.kan_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # After _kan_transform, each feature has input_len * 2 * n_bases dimensions
        in_dim = input_len * 2 * self.n_bases
        for i in range(self.n_layers):
            out_dim = output_len if i == self.n_layers - 1 else self.hidden_dim
            # KAN: for each output, learn combination of basis functions of each input
            self.kan_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                self.layer_norms.append(nn.LayerNorm(out_dim))
                self.dropouts.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        
        # Cross-feature mixing
        self.cross_feature_mixer = nn.Sequential(
            nn.Linear(self.n_features, self.n_features * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_features * 2, self.n_features)
        )
        
    def _kan_transform(self, x):
        # x: (batch, n_features, seq_len)
        batch, n_features, seq_len = x.shape
        
        # Apply Fourier basis: cos(freq * t) and sin(freq * t)
        t = torch.arange(seq_len, dtype=x.dtype, device=x.device).view(1, 1, -1, 1)  # (1, 1, seq_len, 1)
        
        # (batch, n_features, seq_len, n_bases)
        phases = t * self.frequencies.view(1, 1, 1, -1)  # (1, 1, seq_len, n_bases)
        
        cos_basis = torch.cos(phases)  # (1, 1, seq_len, n_bases)
        sin_basis = torch.sin(phases)  # (1, 1, seq_len, n_bases)
        
        # Expand x for basis multiplication
        x_expanded = x.unsqueeze(-1)  # (batch, n_features, seq_len, 1)
        
        # Apply basis: each input dimension gets transformed through all bases
        x_cos = x_expanded * cos_basis  # (batch, n_features, seq_len, n_bases)
        x_sin = x_expanded * sin_basis  # (batch, n_features, seq_len, n_bases)
        
        # Concatenate real and imaginary parts
        x_basis = torch.cat([x_cos, x_sin], dim=-1)  # (batch, n_features, seq_len, 2*n_bases)
        
        return x_basis.view(batch, n_features, -1)  # (batch, n_features, seq_len * 2*n_bases)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-independent: process each feature separately
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Apply KAN transformation with Fourier bases
        x = self._kan_transform(x)  # (batch, n_features, input_len * 2*n_bases)
        
        # Pass through KAN layers
        for i in range(self.n_layers):
            x = self.kan_layers[i](x)  # (batch, n_features, hidden_dim) or (batch, n_features, output_len)
            if i < self.n_layers - 1:
                x = self.layer_norms[i](x)  # (batch, n_features, hidden_dim)
                x = F.gelu(x)  # (batch, n_features, hidden_dim)
                x = self.dropouts[i](x)  # (batch, n_features, hidden_dim)
        
        # x: (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Cross-feature mixing
        x = self.cross_feature_mixer(x)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)
