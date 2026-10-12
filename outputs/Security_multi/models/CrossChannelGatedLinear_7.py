import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelGatedLinear(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_ratio': [2, 8, 32],
            'n_gate_layers': [1, 2, 4],
            'dropout': [0.0, 0.3],
            'use_feature_wise': [True, False]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        hidden_ratio = self.config.get('hidden_ratio', 8)
        n_gate_layers = self.config.get('n_gate_layers', 2)
        dropout = self.config.get('dropout', 0.1)
        use_feature_wise = self.config.get('use_feature_wise', True)
        
        self.revin = RevIN(self.n_features, affine=True)
        self.use_feature_wise = use_feature_wise
        
        hidden_dim = self.n_features * hidden_ratio
        
        # Main temporal linear path (feature-independent)
        self.temporal_proj = nn.Linear(self.input_len, self.output_len)
        
        # Cross-channel gating network
        gate_layers = []
        in_dim = self.n_features
        for i in range(n_gate_layers):
            out_dim = hidden_dim if i < n_gate_layers - 1 else self.n_features
            gate_layers.append(nn.Linear(in_dim, out_dim))
            if i < n_gate_layers - 1:
                gate_layers.append(nn.GELU())
                gate_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        gate_layers.append(nn.Sigmoid())
        self.gate_net = nn.Sequential(*gate_layers)
        
        # Global statistics extractor
        self.stat_extractor = nn.Sequential(
            nn.Linear(self.n_features * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.n_features)
        )
        
        # Feature-wise temporal projections (optional enhancement)
        if use_feature_wise:
            self.feature_temporal = nn.Conv1d(self.n_features, self.n_features, kernel_size=1, groups=self.n_features)
        
        # Cross-feature mixing after temporal projection
        self.cross_mix = nn.Sequential(
            nn.Linear(self.n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_features)
        )
        
        self.cross_norm = nn.LayerNorm(self.n_features)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Compute global statistics for gating
        x_mean = x.mean(dim=1)  # (batch, n_features)
        x_std = x.std(dim=1, unbiased=False) + 1e-5  # (batch, n_features)
        stats = torch.cat([x_mean, x_std], dim=-1)  # (batch, n_features * 2)
        
        # Extract statistics embedding
        stat_embed = self.stat_extractor(stats)  # (batch, n_features)
        
        # Compute gate
        gate = self.gate_net(stat_embed)  # (batch, n_features)
        gate = gate.unsqueeze(1)  # (batch, 1, n_features)
        
        # Main temporal projection (per-feature)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        temporal_out = self.temporal_proj(x_perm)  # (batch, n_features, output_len)
        
        # Optional feature-wise convolution
        if self.use_feature_wise:
            feature_conv = self.feature_temporal(x_perm)  # (batch, n_features, input_len)
            feature_conv = F.interpolate(feature_conv, size=self.output_len, mode='linear', align_corners=False)  # (batch, n_features, output_len)
            temporal_out = temporal_out + 0.1 * feature_conv  # (batch, n_features, output_len)
        
        temporal_out = temporal_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Apply cross-channel gating
        gated = temporal_out * gate  # (batch, output_len, n_features)
        
        # Cross-feature mixing with residual
        mixed = self.cross_mix(gated)  # (batch, output_len, n_features)
        mixed = self.output_dropout(mixed)  # (batch, output_len, n_features)
        out = self.cross_norm(mixed + gated)  # (batch, output_len, n_features)
        
        # Also add direct temporal projection as residual
        direct = temporal_out  # (batch, output_len, n_features)
        out = out + 0.5 * direct  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
