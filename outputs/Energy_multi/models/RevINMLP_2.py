import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class RevINMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'dropout': [0.0, 0.1, 0.3],
            'n_layers': [1, 2, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_layers = self.config.get('n_layers', 2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Build MLP layers channel-independently
        layers = []
        in_dim = self.input_len
        for i in range(self.n_layers):
            out_dim = self.hidden_dim if i < self.n_layers - 1 else self.output_len
            layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Residual projection if dimensions mismatch
        if self.input_len != self.output_len:
            self.residual_proj = nn.Linear(self.input_len, self.output_len)
        else:
            self.residual_proj = None

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode="norm")           # (batch, input_len, n_features)
        
        # Channel-independent: process each feature separately
        x = x.permute(0, 2, 1)                    # (batch, n_features, input_len)
        
        # Store residual for skip connection
        residual = x                              # (batch, n_features, input_len)
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)  # (batch, n_features, output_len)
        else:
            residual = residual                   # (batch, n_features, output_len) if same dim
        
        # Apply MLP
        out = self.mlp(x)                         # (batch, n_features, output_len)
        
        # Add residual connection (trim or pad if needed)
        if residual.shape[-1] == out.shape[-1]:
            out = out + residual                  # (batch, n_features, output_len)
        
        out = out.permute(0, 2, 1)                # (batch, output_len, n_features)
        out = self.revin(out, mode="denorm")      # (batch, output_len, n_features)
        return out

