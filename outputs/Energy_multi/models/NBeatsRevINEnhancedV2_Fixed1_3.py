import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NBeatsRevINEnhancedV2_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_stacks': [2, 4, 8],
            'n_blocks': [1, 3, 6],
            'dropout': [0.0, 0.3],
            'seasonality_harmonics': [8, 16, 64],
            'n_scales': [2, 3, 4],
            'mixing_dim': [64, 256, 512],
            'temporal_dim': [32, 128, 512],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.hidden_dim = cfg.get('hidden_dim', 256)
        self.n_stacks = cfg.get('n_stacks', 4)
        self.n_blocks = cfg.get('n_blocks', 3)
        self.dropout = cfg.get('dropout', 0.3)
        self.seasonality_harmonics = cfg.get('seasonality_harmonics', 16)
        self.n_scales = cfg.get('n_scales', 3)
        self.mixing_dim = cfg.get('mixing_dim', 256)
        self.temporal_dim = cfg.get('temporal_dim', 128)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale seasonality: different harmonic counts per scale
        self.harmonic_scales = []
        for i in range(self.n_scales):
            scale_harmonics = max(1, self.seasonality_harmonics // (2 ** i))
            self.harmonic_scales.append(scale_harmonics)
        
        # Build stacks
        self.stacks = nn.ModuleList()
        for s in range(self.n_stacks):
            is_trend = (s % 2 == 0)
            stack = nn.ModuleList()
            for b in range(self.n_blocks):
                block = NBeatsBlockV2(
                    input_len=self.input_len,
                    output_len=self.output_len,
                    n_features=self.n_features,
                    hidden_dim=self.hidden_dim,
                    dropout=self.dropout,
                    is_trend=is_trend,
                    harmonic_scales=self.harmonic_scales if not is_trend else None,
                    temporal_dim=self.temporal_dim,
                )
                stack.append(block)
            self.stacks.append(stack)
        
        # Stack residual gating weights
        self.stack_gates = nn.Parameter(torch.ones(self.n_stacks))
        
        # Hierarchical feature mixing: local (feature-wise) then global (cross-feature)
        self.local_mixing = nn.Sequential(
            nn.Linear(self.n_features, self.mixing_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mixing_dim, self.n_features),
        )
        
        self.global_mixing = nn.Sequential(
            nn.Linear(self.n_features, self.mixing_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mixing_dim, self.n_features),
        )
        
        # Adaptive blending of local vs global mixing
        self.mix_gate = nn.Linear(self.n_features, self.n_features)
        
        # Temporal attention for lookback weighting
        self.temporal_query = nn.Linear(self.n_features, self.temporal_dim)
        self.temporal_key = nn.Linear(self.n_features, self.temporal_dim)
        self.temporal_value = nn.Linear(self.n_features, self.temporal_dim)
        self.temporal_out = nn.Linear(self.temporal_dim, self.n_features)
        
        # Output projection
        self.output_proj = nn.Linear(self.n_features, self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Temporal attention on input
        # Compute attention weights across time dimension
        q = self.temporal_query(x)  # (batch, input_len, temporal_dim)
        k = self.temporal_key(x)    # (batch, input_len, temporal_dim)
        v = self.temporal_value(x)  # (batch, input_len, temporal_dim)
        
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.temporal_dim ** 0.5)  # (batch, input_len, input_len)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, input_len, input_len)
        attn_out = torch.bmm(attn_weights, v)  # (batch, input_len, temporal_dim)
        attn_out = self.temporal_out(attn_out)  # (batch, input_len, n_features)
        
        # Residual connection with gating
        x = x + 0.1 * attn_out  # (batch, input_len, n_features)
        
        # Process through N-BEATS stacks
        backcast = x  # (batch, input_len, n_features)
        forecast = torch.zeros(x.size(0), self.output_len, self.n_features, device=x.device)  # (batch, output_len, n_features)
        
        for s, stack in enumerate(self.stacks):
            stack_forecast = torch.zeros(x.size(0), self.output_len, self.n_features, device=x.device)  # (batch, output_len, n_features)
            for block in stack:
                b, f = block(backcast)  # b: (batch, input_len, n_features), f: (batch, output_len, n_features)
                backcast = backcast - b  # (batch, input_len, n_features)
                stack_forecast = stack_forecast + f  # (batch, output_len, n_features)
            
            # Gated residual between stacks
            gate = torch.sigmoid(self.stack_gates[s])  # scalar
            forecast = forecast + gate * stack_forecast  # (batch, output_len, n_features)
        
        # Hierarchical feature mixing
        # Local mixing (independent per time step)
        local_out = self.local_mixing(forecast)  # (batch, output_len, n_features)
        
        # Global mixing (cross-feature interactions)
        global_out = self.global_mixing(forecast)  # (batch, output_len, n_features)
        
        # Adaptive blend
        blend_gate = torch.sigmoid(self.mix_gate(forecast))  # (batch, output_len, n_features)
        mixed = blend_gate * local_out + (1 - blend_gate) * global_out  # (batch, output_len, n_features)
        
        # Final projection with residual
        out = self.output_proj(mixed) + forecast  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)


class NBeatsBlockV2(nn.Module):
    def __init__(self, input_len, output_len, n_features, hidden_dim, dropout, is_trend, harmonic_scales, temporal_dim):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.is_trend = is_trend
        self.harmonic_scales = harmonic_scales
        self.temporal_dim = temporal_dim
        
        # FC layers for backcast/forecast generation (channel-independent)
        self.fc = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        if is_trend:
            # Trend: polynomial basis
            self.degree = 3
            self.backcast_basis = nn.Linear(self.degree + 1, input_len, bias=False)
            self.forecast_basis = nn.Linear(self.degree + 1, output_len, bias=False)
            self.coeff_proj = nn.Linear(hidden_dim, self.degree + 1)
        else:
            # Multi-scale seasonality
            self.scale_weights = nn.Parameter(torch.ones(len(harmonic_scales)))
            total_harmonics = sum(harmonic_scales)
            self.backcast_basis = nn.Linear(2 * total_harmonics, input_len, bias=False)
            self.forecast_basis = nn.Linear(2 * total_harmonics, output_len, bias=False)
            self.coeff_proj = nn.Linear(hidden_dim, 2 * total_harmonics)
            
            # Precompute frequency grids
            self.register_buffer('backcast_grid', self._create_grid(input_len, harmonic_scales))
            self.register_buffer('forecast_grid', self._create_grid(output_len, harmonic_scales))
        
    def _create_grid(self, length, scales):
        # Create multi-scale frequency basis
        t = torch.arange(length, dtype=torch.float32) / length  # (length,)
        components = []
        for h in scales:
            for i in range(1, h + 1):
                components.append(torch.sin(2 * torch.pi * i * t))
                components.append(torch.cos(2 * torch.pi * i * t))
        return torch.stack(components, dim=1)  # (length, 2*total_harmonics)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.size(0)
        
        # Channel-independent processing
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_flat = x_perm.reshape(batch_size * self.n_features, self.input_len)  # (batch*n_features, input_len)
        
        h = self.fc(x_flat)  # (batch*n_features, hidden_dim)
        coeffs = self.coeff_proj(h)  # (batch*n_features, num_coeffs)
        
        if self.is_trend:
            # coeffs: (batch*n_features, degree+1)
            backcast = self.backcast_basis(coeffs)  # (batch*n_features, input_len)
            forecast = self.forecast_basis(coeffs)  # (batch*n_features, output_len)
        else:
            # coeffs: (batch*n_features, 2*total_harmonics)
            
            # Apply scale-specific weighting
            scale_norm = F.softmax(self.scale_weights, dim=0)  # (n_scales,)
            
            # Weighted combination through basis matrices
            backcast = self.backcast_basis(coeffs)  # (batch*n_features, input_len)
            forecast = self.forecast_basis(coeffs)  # (batch*n_features, output_len)
        
        backcast = backcast.reshape(batch_size, self.n_features, self.input_len).permute(0, 2, 1)  # (batch, input_len, n_features)
        forecast = forecast.reshape(batch_size, self.n_features, self.output_len).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        return backcast, forecast

