import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NBeatsRevINEnhancedV2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_blocks': [1, 4, 8],
            'n_harmonics': [4, 16, 64],
            'n_polynomials': [2, 4, 8],
            'dropout': [0.0, 0.3],
            'bottleneck_dim': [8, 32, 128],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_blocks = self.config.get('n_blocks', 4)
        self.n_harmonics = self.config.get('n_harmonics', 16)
        self.n_polynomials = self.config.get('n_polynomials', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.bottleneck_dim = self.config.get('bottleneck_dim', 32)
        
        # Multi-scale harmonic frequencies: short, medium, long-term
        self.n_scales = 3
        self.harmonics_per_scale = max(1, self.n_harmonics // self.n_scales)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Build multi-scale N-BEATS blocks for each scale
        self.scale_blocks = nn.ModuleList()
        for scale_idx in range(self.n_scales):
            scale_blocks_list = nn.ModuleList()
            for _ in range(self.n_blocks):
                block = NBeatsBlockMultiScale(
                    input_len=self.input_len,
                    output_len=self.output_len,
                    n_features=self.n_features,
                    hidden_dim=self.hidden_dim,
                    n_harmonics=self.harmonics_per_scale,
                    n_polynomials=self.n_polynomials,
                    dropout=self.dropout,
                    scale_idx=scale_idx,
                    n_scales=self.n_scales
                )
                scale_blocks_list.append(block)
            self.scale_blocks.append(scale_blocks_list)
        
        # Hierarchical gating: fuse predictions across scales with learnable weights
        self.scale_gate = nn.Sequential(
            nn.Linear(self.n_features * self.n_scales, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.n_scales),
            nn.Softmax(dim=-1)
        )
        
        # Cross-feature attention refinement (operates on bottleneck)
        self.feature_compress = nn.Linear(self.n_features, self.bottleneck_dim)
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=self.bottleneck_dim,
            num_heads=max(1, self.bottleneck_dim // 8),
            dropout=self.dropout,
            batch_first=True
        )
        self.feature_expand = nn.Linear(self.bottleneck_dim, self.n_features)
        
        # Residual projection for skip connection
        self.residual_proj = nn.Linear(self.input_len, self.output_len)
        
        # Final output gating
        self.output_gate = nn.Sequential(
            nn.Linear(self.n_features * 2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.n_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x_norm = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Residual connection (direct linear projection)
        x_perm = x_norm.permute(0, 2, 1)  # (batch, n_features, input_len)
        residual = self.residual_proj(x_perm)  # (batch, n_features, output_len)
        residual = residual.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Multi-scale decomposition
        scale_forecasts = []  # List of (batch, output_len, n_features)
        scale_backcasts = []  # List of (batch, input_len, n_features)
        
        for scale_idx in range(self.n_scales):
            scale_input = x_norm  # (batch, input_len, n_features)
            scale_forecast = torch.zeros(batch_size, self.output_len, self.n_features, device=x.device)
            
            for block in self.scale_blocks[scale_idx]:
                backcast, forecast = block(scale_input)  # both (batch, seq, n_features)
                scale_input = scale_input - backcast  # residual connection
                scale_forecast = scale_forecast + forecast
            
            scale_forecasts.append(scale_forecast)
            scale_backcasts.append(scale_input)
        
        # Stack scale forecasts: (batch, output_len, n_features, n_scales)
        stacked_forecasts = torch.stack(scale_forecasts, dim=-1)  # (batch, output_len, n_features, n_scales)
        
        # Hierarchical gating: compute per-timestep, per-feature scale weights
        # Aggregate features for gating decision
        gate_input = stacked_forecasts.permute(0, 1, 3, 2)  # (batch, output_len, n_scales, n_features)
        gate_input = gate_input.reshape(batch_size, self.output_len, self.n_scales * self.n_features)
        # (batch, output_len, n_scales * n_features)
        
        scale_weights = self.scale_gate(gate_input)  # (batch, output_len, n_scales)
        scale_weights = scale_weights.unsqueeze(2)  # (batch, output_len, 1, n_scales)
        
        # Weighted combination across scales
        gated_forecast = (stacked_forecasts * scale_weights).sum(dim=-1)  # (batch, output_len, n_features)
        
        # Cross-feature attention refinement on bottleneck
        # Compress: (batch, output_len, n_features) -> (batch, output_len, bottleneck_dim)
        compressed = self.feature_compress(gated_forecast)  # (batch, output_len, bottleneck_dim)
        
        # Self-attention across time steps (temporal attention on compressed features)
        attn_out, _ = self.feature_attention(compressed, compressed, compressed)
        # (batch, output_len, bottleneck_dim)
        
        # Expand back: (batch, output_len, bottleneck_dim) -> (batch, output_len, n_features)
        cross_feature_refined = self.feature_expand(attn_out)  # (batch, output_len, n_features)
        
        # Output gating: blend gated forecast with cross-feature refinement
        gate_input = torch.cat([gated_forecast, cross_feature_refined], dim=-1)
        # (batch, output_len, n_features * 2)
        output_weight = self.output_gate(gate_input)  # (batch, output_len, n_features)
        
        final_forecast = output_weight * gated_forecast + (1 - output_weight) * cross_feature_refined
        # (batch, output_len, n_features)
        
        # Add residual connection
        final_forecast = final_forecast + residual  # (batch, output_len, n_features)
        
        # Denormalize
        output = self.revin(final_forecast, mode='denorm')  # (batch, output_len, n_features)
        
        return output


class NBeatsBlockMultiScale(nn.Module):
    def __init__(self, input_len, output_len, n_features, hidden_dim, n_harmonics, n_polynomials, dropout, scale_idx, n_scales):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_harmonics = n_harmonics
        self.n_polynomials = n_polynomials
        self.scale_idx = scale_idx
        self.n_scales = n_scales
        
        # Scale-dependent frequency modulation
        self.freq_multiplier = 2.0 ** scale_idx  # Higher scales = higher frequencies
        
        # FC stack for backcast/forecast generation (channel-independent)
        self.fc = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Basis coefficients
        self.backcast_basis_dim = n_harmonics * 2 + n_polynomials
        self.forecast_basis_dim = n_harmonics * 2 + n_polynomials
        
        self.backcast_proj = nn.Linear(hidden_dim, self.backcast_basis_dim)
        self.forecast_proj = nn.Linear(hidden_dim, self.forecast_basis_dim)
        
        # Learnable trend/seasonality gating (per-feature)
        self.trend_gate = nn.Sequential(
            nn.Linear(hidden_dim, n_features),
            nn.Sigmoid()
        )
        self.season_gate = nn.Sequential(
            nn.Linear(hidden_dim, n_features),
            nn.Sigmoid()
        )
        
        # Build basis functions
        self.register_buffer('backcast_basis', self._build_basis(input_len, backcast=True))
        self.register_buffer('forecast_basis', self._build_basis(output_len, backcast=False))
        
    def _build_basis(self, length, backcast=True):
        # Time grid
        t = torch.linspace(0, 1, length).unsqueeze(0)  # (1, length)
        
        basis = []
        
        # Polynomial (trend) basis
        for p in range(self.n_polynomials):
            basis.append(t ** p)  # (1, length)
        
        # Harmonic (seasonality) basis with scale-dependent frequencies
        for h in range(1, self.n_harmonics + 1):
            freq = h * self.freq_multiplier * 2 * 3.141592653589793
            basis.append(torch.sin(freq * t))  # (1, length)
            basis.append(torch.cos(freq * t))  # (1, length)
        
        basis = torch.cat(basis, dim=0)  # (basis_dim, length)
        return basis
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Channel-independent processing: (batch, n_features, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Process each feature independently
        x_flat = x_perm.reshape(batch_size * self.n_features, self.input_len)
        # (batch * n_features, input_len)
        
        h = self.fc(x_flat)  # (batch * n_features, hidden_dim)
        
        # Generate basis coefficients
        backcast_coef = self.backcast_proj(h)  # (batch * n_features, backcast_basis_dim)
        forecast_coef = self.forecast_proj(h)  # (batch * n_features, forecast_basis_dim)
        
        # Gating for trend/seasonality blend
        trend_w = self.trend_gate(h)  # (batch * n_features, n_features)
        # Actually we need per-feature gating, not cross-feature
        # Reshape: take diagonal-like or mean approach
        trend_w = trend_w.mean(dim=-1, keepdim=True)  # (batch * n_features, 1)
        season_w = self.season_gate(h).mean(dim=-1, keepdim=True)  # (batch * n_features, 1)
        
        # Normalize gates
        total = trend_w + season_w + 1e-6
        trend_w = trend_w / total
        season_w = season_w / total
        
        # Apply basis functions
        # Split coefficients into trend and seasonality
        trend_backcast_coef = backcast_coef[:, :self.n_polynomials]  # (batch*n_features, n_polynomials)
        season_backcast_coef = backcast_coef[:, self.n_polynomials:]  # (batch*n_features, n_harmonics*2)
        
        trend_forecast_coef = forecast_coef[:, :self.n_polynomials]
        season_forecast_coef = forecast_coef[:, self.n_polynomials:]
        
        # Trend basis
        trend_backcast_basis = self.backcast_basis[:self.n_polynomials, :]  # (n_polynomials, input_len)
        trend_forecast_basis = self.forecast_basis[:self.n_polynomials, :]  # (n_polynomials, output_len)
        
        # Seasonality basis
        season_backcast_basis = self.backcast_basis[self.n_polynomials:, :]  # (n_harmonics*2, input_len)
        season_forecast_basis = self.forecast_basis[self.n_polynomials:, :]  # (n_harmonics*2, output_len)
        
        # Compute outputs
        trend_backcast = torch.matmul(trend_backcast_coef, trend_backcast_basis)  # (batch*n_features, input_len)
        season_backcast = torch.matmul(season_backcast_coef, season_backcast_basis)  # (batch*n_features, input_len)
        
        trend_forecast = torch.matmul(trend_forecast_coef, trend_forecast_basis)  # (batch*n_features, output_len)
        season_forecast = torch.matmul(season_forecast_coef, season_forecast_basis)  # (batch*n_features, output_len)
        
        # Apply gating
        backcast = trend_w * trend_backcast + season_w * season_backcast  # (batch*n_features, input_len)
        forecast = trend_w * trend_forecast + season_w * season_forecast  # (batch*n_features, output_len)
        
        # Reshape back
        backcast = backcast.reshape(batch_size, self.n_features, self.input_len)
        backcast = backcast.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        forecast = forecast.reshape(batch_size, self.n_features, self.output_len)
        forecast = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        return backcast, forecast
