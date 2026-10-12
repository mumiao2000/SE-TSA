import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class GatedResidualMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.1, 0.3],
            'gate_ratio': [0.5, 1.0, 2.0]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.gate_ratio = self.config.get('gate_ratio', 1.0)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: (batch, input_len, n_features) -> (batch, n_features, hidden_dim)
        self.input_proj = nn.Linear(self.input_len, self.hidden_dim)
        
        # Gated residual blocks
        self.gate_blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.n_layers):
            gate_hidden = int(self.hidden_dim * self.gate_ratio)
            self.gate_blocks.append(nn.ModuleDict({
                'value': nn.Sequential(
                    nn.Linear(self.hidden_dim, gate_hidden),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(gate_hidden, self.hidden_dim)
                ),
                'gate': nn.Sequential(
                    nn.Linear(self.hidden_dim, gate_hidden),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(gate_hidden, self.hidden_dim),
                    nn.Sigmoid()
                )
            }))
            self.norms.append(nn.LayerNorm(self.hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.output_len)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-independent: transpose to (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.input_proj(x)  # (batch, n_features, hidden_dim)
        
        # Gated residual blocks
        for block, norm in zip(self.gate_blocks, self.norms):
            residual = x  # (batch, n_features, hidden_dim)
            v = block['value'](x)  # (batch, n_features, hidden_dim)
            g = block['gate'](x)   # (batch, n_features, hidden_dim)
            x = v * g  # (batch, n_features, hidden_dim)
            x = self.dropout_layer(x)
            x = norm(x + residual)  # (batch, n_features, hidden_dim)
        
        x = self.output_proj(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)   # (batch, output_len, n_features)
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x
