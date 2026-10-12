import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NLinearChannelMixing_Improved(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'mixing_dim': [64, 256, 1024],
            'n_mixing_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'temporal_scales': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.mixing_dim = self.config.get('mixing_dim', 256)
        self.n_mixing_layers = self.config.get('n_mixing_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.3)
        self.temporal_scales = self.config.get('temporal_scales', 4)
        
        # RevIN for normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale temporal aggregation: learnable weighted combination of different kernel sizes
        self.temporal_kernels = nn.ModuleList()
        self.temporal_weights = nn.Parameter(torch.ones(self.temporal_scales))
        for i in range(self.temporal_scales):
            kernel_size = 2 * i + 1  # 1, 3, 5, 7, ...
            if kernel_size <= input_len:
                padding = kernel_size // 2
                self.temporal_kernels.append(
                    nn.Conv1d(self.n_features, self.n_features, kernel_size, padding=padding, groups=self.n_features)
                )
            else:
                self.temporal_kernels.append(None)
        
        # Project input to mixing dimension
        self.input_proj = nn.Linear(self.n_features, self.mixing_dim)
        self.input_norm = nn.LayerNorm(self.mixing_dim)
        self.input_dropout = nn.Dropout(self.dropout_rate)
        
        # Gating for input projection residual
        self.input_gate = nn.Sequential(
            nn.Linear(self.n_features + self.mixing_dim, self.mixing_dim),
            nn.Sigmoid()
        )
        
        # Channel mixing layers with gated residuals
        self.mixing_layers = nn.ModuleList()
        for i in range(self.n_mixing_layers):
            self.mixing_layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(self.mixing_dim),
                'mlp': nn.Sequential(
                    nn.Linear(self.mixing_dim, self.mixing_dim * 4),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.mixing_dim * 4, self.mixing_dim),
                    nn.Dropout(self.dropout_rate),
                ),
                'gate': nn.Sequential(
                    nn.Linear(self.mixing_dim * 2, self.mixing_dim),
                    nn.Sigmoid()
                ),
            }))
        
        # Output projection: from mixing_dim back to n_features, then to output_len
        self.output_norm = nn.LayerNorm(self.mixing_dim)
        self.feature_proj = nn.Linear(self.mixing_dim, self.n_features)
        self.temporal_proj = nn.Linear(input_len, output_len)
        
        # Final gate for blending direct and transformed paths
        self.output_gate = nn.Sequential(
            nn.Linear(self.n_features * 2, self.n_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Store original for final residual
        x_orig = x  # (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Multi-scale temporal aggregation
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        temporal_outs = []
        valid_scales = 0
        for i, kernel in enumerate(self.temporal_kernels):
            if kernel is not None:
                out = kernel(x_perm)  # (batch, n_features, input_len)
                temporal_outs.append(out)
                valid_scales += 1
        
        # Weighted combination of temporal scales
        if len(temporal_outs) > 0:
            weights = F.softmax(self.temporal_weights[:valid_scales], dim=0)
            x_temporal = sum(w * out for w, out in zip(weights, temporal_outs))  # (batch, n_features, input_len)
            x = x + x_temporal.permute(0, 2, 1)  # (batch, input_len, n_features), residual connection
        
        # Project to mixing dimension
        x_proj = self.input_proj(x)  # (batch, input_len, mixing_dim)
        x_proj = self.input_norm(x_proj)  # (batch, input_len, mixing_dim)
        x_proj = self.input_dropout(x_proj)  # (batch, input_len, mixing_dim)
        
        # Gated residual for input projection
        gate_input = torch.cat([x, x_proj], dim=-1)  # (batch, input_len, n_features + mixing_dim)
        gate = self.input_gate(gate_input)  # (batch, input_len, mixing_dim)
        # Pad x to match mixing_dim for residual
        x_padded = F.pad(x, (0, self.mixing_dim - self.n_features), 'constant', 0)  # (batch, input_len, mixing_dim)
        x = gate * x_proj + (1 - gate) * x_padded  # (batch, input_len, mixing_dim)
        
        # Channel mixing layers with gated residuals
        for layer in self.mixing_layers:
            residual = x  # (batch, input_len, mixing_dim)
            x = layer['norm1'](x)  # (batch, input_len, mixing_dim)
            x = layer['mlp'](x)  # (batch, input_len, mixing_dim)
            # Gated residual
            gate_input = torch.cat([x, residual], dim=-1)  # (batch, input_len, mixing_dim * 2)
            gate = layer['gate'](gate_input)  # (batch, input_len, mixing_dim)
            x = gate * x + (1 - gate) * residual  # (batch, input_len, mixing_dim)
        
        # Output projection
        x = self.output_norm(x)  # (batch, input_len, mixing_dim)
        x = self.feature_proj(x)  # (batch, input_len, n_features)
        
        # Temporal projection to output_len
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.temporal_proj(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Gated blend with direct temporal projection of original (DeLinformer-style)
        x_direct = x_orig.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_direct = self.temporal_proj(x_direct)  # (batch, n_features, output_len)
        x_direct = x_direct.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        gate_input = torch.cat([x, x_direct], dim=-1)  # (batch, output_len, n_features * 2)
        gate = self.output_gate(gate_input)  # (batch, output_len, n_features)
        x = gate * x + (1 - gate) * x_direct  # (batch, output_len, n_features)
        
        # RevIN denormalization
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x  # (batch, output_len, n_features)
