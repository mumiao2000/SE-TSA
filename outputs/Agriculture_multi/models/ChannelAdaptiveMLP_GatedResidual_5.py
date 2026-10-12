import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveMLP_GatedResidual(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'gate_dim': [32, 128, 512],
            'temporal_mix_dim': [64, 256, 1024],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.gate_dim = self.config.get('gate_dim', 128)
        self.temporal_mix_dim = self.config.get('temporal_mix_dim', 256)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Feature-wise temporal mixing: (batch, n_features, input_len) -> (batch, n_features, temporal_mix_dim)
        self.temporal_mix = nn.Sequential(
            nn.Linear(self.input_len, self.temporal_mix_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.temporal_mix_dim, self.hidden_dim),
        )  # (batch, n_features, hidden_dim)
        
        # Channel-wise adaptive MLP blocks with residual connections
        self.blocks = nn.ModuleList()
        for i in range(self.n_layers):
            self.blocks.append(nn.ModuleDict({
                'mlp': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                ),
                'gate': nn.Sequential(
                    nn.Linear(self.n_features, self.gate_dim),
                    nn.ReLU(),
                    nn.Linear(self.gate_dim, self.n_features),
                    nn.Sigmoid(),
                ),
                'norm': nn.LayerNorm(self.hidden_dim),
            }))
        
        # Temporal decomposition head: level (constant) + trend (linear)
        self.level_proj = nn.Linear(self.hidden_dim, self.output_len)  # (batch, n_features, output_len)
        self.trend_proj = nn.Linear(self.hidden_dim, self.output_len)  # (batch, n_features, output_len)
        
        # Trend coefficient (learnable slope per channel)
        self.trend_init = nn.Parameter(torch.zeros(self.n_features, 1))  # (n_features, 1)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for feature-wise operations
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Feature-wise temporal mixing
        h = self.temporal_mix(x)  # (batch, n_features, hidden_dim)
        
        # Channel-wise adaptive MLP blocks with residual connections
        for block in self.blocks:
            residual = h  # (batch, n_features, hidden_dim)
            
            # MLP on hidden dim
            h_mlp = block['mlp'](h)  # (batch, n_features, hidden_dim)
            h_mlp = self.dropout(h_mlp)  # (batch, n_features, hidden_dim)
            
            # Channel gate: aggregate across hidden dim, compute gate per channel
            h_pooled = h.mean(dim=-1)  # (batch, n_features)
            gate = block['gate'](h_pooled)  # (batch, n_features)
            gate = gate.unsqueeze(-1)  # (batch, n_features, 1)
            
            # Apply gate and residual
            h = block['norm'](residual + gate * h_mlp)  # (batch, n_features, hidden_dim)
        
        # Temporal decomposition: level + trend
        level = self.level_proj(h)  # (batch, n_features, output_len)
        
        # Trend: learnable linear trend per channel
        trend_base = self.trend_proj(h)  # (batch, n_features, output_len)
        # Create time steps [0, 1, ..., output_len-1]
        t = torch.arange(self.output_len, device=h.device, dtype=h.dtype)  # (output_len,)
        t = t.view(1, 1, self.output_len)  # (1, 1, output_len)
        trend_coef = self.trend_init.unsqueeze(0)  # (1, n_features, 1)
        trend = trend_coef * t * trend_base  # (batch, n_features, output_len)
        
        out = level + trend  # (batch, n_features, output_len)
        
        # Transpose back
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)
