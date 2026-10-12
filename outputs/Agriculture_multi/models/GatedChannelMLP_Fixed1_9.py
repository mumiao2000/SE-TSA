import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class GatedChannelMLP_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_blocks': [1, 4, 8],
            'gate_dim': [16, 64, 256],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_blocks = self.config.get('n_blocks', 4)
        self.gate_dim = self.config.get('gate_dim', 64)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Feature importance embedding (learned)
        self.feature_importance = nn.Parameter(torch.ones(n_features))
        
        # Global gating network: decides layer usage
        self.global_gate = nn.Sequential(
            nn.Linear(input_len * n_features, self.gate_dim),
            nn.GELU(),
            nn.Linear(self.gate_dim, self.n_blocks),
            nn.Sigmoid()
        )
        
        # Initial projection to hidden_dim
        self.input_proj = nn.Linear(input_len, self.hidden_dim)
        
        # Per-channel MLP blocks with gating - all operate on hidden_dim
        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            block = nn.ModuleDict({
                'norm': nn.LayerNorm(self.hidden_dim),
                'up': nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                'gate_act': nn.GELU(),
                'down': nn.Linear(self.hidden_dim, self.hidden_dim),
                'dropout': nn.Dropout(self.dropout)
            })
            self.blocks.append(block)
        
        # Final projection
        self.final_proj = nn.Linear(self.hidden_dim, output_len)
        
        # Direct shortcut
        self.shortcut = nn.Linear(input_len, output_len)
        
        # Feature-wise output scaling
        self.output_scale = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Shortcut
        shortcut = self.shortcut(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Apply feature importance weighting
        importance = torch.softmax(self.feature_importance, dim=0)  # (n_features,)
        x_weighted = x * importance.unsqueeze(0).unsqueeze(0)  # (batch, input_len, n_features)
        
        # Global context for adaptive depth
        flat = x_weighted.reshape(x_weighted.size(0), -1)  # (batch, input_len * n_features)
        layer_gates = self.global_gate(flat)  # (batch, n_blocks)
        
        # Process per channel
        h = x_weighted.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Initial projection to hidden_dim
        h = self.input_proj(h)  # (batch, n_features, hidden_dim)
        
        for i, block in enumerate(self.blocks):
            # Residual
            residual = h  # (batch, n_features, hidden_dim)
            
            # Normalize - LayerNorm on last dim which is hidden_dim
            h_norm = block['norm'](h)  # (batch, n_features, hidden_dim)
            
            # Up-project and gate
            h_up = block['up'](h_norm)  # (batch, n_features, hidden_dim * 2)
            h_gate, h_val = h_up.chunk(2, dim=-1)  # each (batch, n_features, hidden_dim)
            h_gated = h_val * block['gate_act'](h_gate)  # (batch, n_features, hidden_dim)
            
            # Down-project
            h_down = block['down'](h_gated)  # (batch, n_features, hidden_dim)
            h_down = block['dropout'](h_down)
            
            # Apply adaptive gate (per sample, broadcast across features)
            gate = layer_gates[:, i].unsqueeze(1).unsqueeze(2)  # (batch, 1, 1)
            h = residual + h_down * gate  # (batch, n_features, hidden_dim)
        
        # Final projection to output_len
        out = self.final_proj(h)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Feature-wise output scaling
        scale = self.output_scale(out.mean(dim=1))  # (batch, n_features)
        out = out * scale.unsqueeze(1)  # (batch, output_len, n_features)
        
        # Add shortcut
        out = out + shortcut  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

