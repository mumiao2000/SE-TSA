import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceGatedMLP_v2_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_state': [16, 64, 256],
            'd_inner': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'dt_rank': [4, 16, 64],
            'channel_mix_ratio': [2, 4, 8],
            'use_horizon_branch': [True, False],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.d_state = self.config.get('d_state', 64)
        self.d_inner = self.config.get('d_inner', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.0)
        self.dt_rank = self.config.get('dt_rank', 16)
        self.channel_mix_ratio = self.config.get('channel_mix_ratio', 4)
        self.use_horizon_branch = self.config.get('use_horizon_branch', True)
        
        # Horizon-adaptive state dimension: shorter horizons need larger states
        if self.use_horizon_branch:
            # Inverse relationship: longer output -> smaller effective state
            self.horizon_scale = max(1, int(256 / max(self.output_len, 1)))
            self.effective_d_state = min(self.d_state, max(16, self.d_state // self.horizon_scale))
        else:
            self.effective_d_state = self.d_state
        
        # REVIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to d_inner (channel-independent: each feature processed separately)
        self.in_proj = nn.Linear(self.input_len, self.d_inner)
        
        # State-space layers with GELU-GLU gating
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            layer = StateSpaceLayer(
                d_inner=self.d_inner,
                dt_rank=self.dt_rank,
                effective_d_state=self.effective_d_state,
                dropout=self.dropout,
            )
            self.layers.append(layer)
        
        # Channel-mixing bottleneck: cross-feature interaction without attention
        # Operates on n_features dimension, so norm should be on n_features
        self.channel_mixer = nn.Sequential(
            nn.Linear(self.n_features, self.n_features * self.channel_mix_ratio),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_features * self.channel_mix_ratio, self.n_features),
            nn.Dropout(self.dropout),
        )
        self.mixer_norm = nn.LayerNorm(self.n_features)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-independent: process each feature separately
        # (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project to d_inner
        x = self.in_proj(x)  # (batch, n_features, d_inner)
        
        # Apply state-space layers
        for layer in self.layers:
            x = layer(x)  # (batch, n_features, d_inner)
        
        # Channel-mixing: allow cross-feature interaction
        # Transpose to (batch, d_inner, n_features) for channel mixing
        x_mixed = x.permute(0, 2, 1)  # (batch, d_inner, n_features)
        # Apply norm on n_features dimension (last dim after transpose)
        x_mixed = self.mixer_norm(x_mixed)  # (batch, d_inner, n_features)
        x_mixed = self.channel_mixer(x_mixed)  # (batch, d_inner, n_features)
        x_mixed = x_mixed.permute(0, 2, 1)  # (batch, n_features, d_inner)
        
        # Residual add with channel mixing
        x = x + x_mixed  # (batch, n_features, d_inner)
        
        # Output projection to forecast horizon
        x = self.out_proj(x)  # (batch, n_features, output_len)
        
        # Restore shape
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Denormalize
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x  # (batch, output_len, n_features)


class StateSpaceLayer(nn.Module):
    def __init__(self, d_inner, dt_rank, effective_d_state, dropout):
        super().__init__()
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.effective_d_state = effective_d_state
        
        self.dt_proj = nn.Linear(d_inner, dt_rank, bias=False)
        self.B_proj = nn.Linear(d_inner, effective_d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, effective_d_state, bias=False)
        self.A_log = nn.Parameter(torch.randn(effective_d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(effective_d_state))
        self.gate = nn.Linear(d_inner, d_inner * 2)
        self.out_proj = nn.Linear(d_inner, d_inner)
        self.norm = nn.LayerNorm(d_inner)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, n_features, d_inner)
        residual = x  # (batch, n_features, d_inner)
        
        # Layer norm
        x = self.norm(x)  # (batch, n_features, d_inner)
        
        # Compute input-dependent parameters for state-space
        dt = F.softplus(self.dt_proj(x))  # (batch, n_features, dt_rank)
        B = self.B_proj(x)  # (batch, n_features, effective_d_state)
        C = self.C_proj(x)  # (batch, n_features, effective_d_state)
        
        # Discretize: simplified diagonal SSM step
        A = -torch.exp(self.A_log)  # (effective_d_state,)
        # Expand A for broadcasting
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, effective_d_state)
        
        # Gate: split into two parts for GELU-GLU
        gate = self.gate(x)  # (batch, n_features, d_inner * 2)
        gate1, gate2 = gate.chunk(2, dim=-1)  # each (batch, n_features, d_inner)
        
        # GELU-GLU gating
        gated = F.gelu(gate1) * gate2  # (batch, n_features, d_inner)
        
        # Apply state-space inspired modulation using B, C, D
        # Simplified: use B and C as gating factors
        state_mod = torch.sigmoid(B.mean(dim=-1, keepdim=True))  # (batch, n_features, 1)
        gated = gated * state_mod  # (batch, n_features, d_inner)
        
        # Output projection with residual
        x = self.out_proj(gated)  # (batch, n_features, d_inner)
        x = self.dropout(x)  # (batch, n_features, d_inner)
        x = x + residual  # (batch, n_features, d_inner) - residual connection
        
        return x  # (batch, n_features, d_inner)
