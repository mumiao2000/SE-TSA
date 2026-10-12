import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class MultiScaleDecompositionForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'trend_degree': [1, 2, 3],
            'n_harmonics': [4, 16, 64],
            'residual_dim': [32, 128, 512],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.trend_degree = self.config.get('trend_degree', 2)
        self.n_harmonics = self.config.get('n_harmonics', 16)
        self.residual_dim = self.config.get('residual_dim', 128)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Decomposition: learned mixing weights
        self.decomp_weights = nn.Parameter(torch.ones(3) / 3)  # trend, seasonal, residual
        
        # Trend: polynomial basis fitting
        self.trend_basis = nn.Parameter(torch.randn(self.trend_degree + 1, n_features), requires_grad=True)
        self.trend_proj = nn.Sequential(
            nn.Linear(input_len, self.trend_degree + 1),
            nn.GELU(),
            nn.Linear(self.trend_degree + 1, output_len)
        )
        
        # Seasonal: Fourier basis with learned frequencies around dominant period
        # Use multiple periods around the dominant one
        self.periods = nn.Parameter(torch.tensor([
            346.0, 173.0, 115.3, 86.5, 69.2  # fundamental + harmonics
        ]), requires_grad=False)
        
        self.seasonal_amplitude = nn.Parameter(torch.randn(len(self.periods), n_features, self.n_harmonics) * 0.1)
        self.seasonal_phase = nn.Parameter(torch.randn(len(self.periods), n_features, self.n_harmonics) * 0.1)
        
        self.seasonal_proj = nn.Sequential(
            nn.Linear(input_len, self.residual_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.residual_dim, output_len)
        )
        
        # Residual: lightweight MLP
        self.residual_encoder = nn.Sequential(
            nn.Linear(input_len, self.residual_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.residual_dim, self.residual_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.residual_dim, output_len)
        )
        
        # Learned combination
        self.combination = nn.Sequential(
            nn.Linear(n_features * 3, n_features * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(n_features * 2, n_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Create time indices
        t_in = torch.arange(self.input_len, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(-1)  # (1, input_len, 1)
        t_out = torch.arange(self.output_len, dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(-1)  # (1, output_len, 1)
        
        # === TREND COMPONENT ===
        # Fit polynomial to input
        # Normalize time for numerical stability
        t_in_norm = t_in / self.input_len  # (1, input_len, 1)
        t_out_norm = t_out / self.input_len  # (1, output_len, 1)
        
        # Polynomial basis: [1, t, t^2, ...]
        trend_basis_in = torch.cat([t_in_norm ** d for d in range(self.trend_degree + 1)], dim=-1)  # (1, input_len, degree+1)
        trend_basis_out = torch.cat([t_out_norm ** d for d in range(self.trend_degree + 1)], dim=-1)  # (1, output_len, degree+1)
        
        # Project input to polynomial coefficients, then evaluate at output times
        # Simplified: use MLP to predict trend
        trend_in = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        trend_out = self.trend_proj(trend_in)  # (batch, n_features, output_len)
        trend_out = trend_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # === SEASONAL COMPONENT ===
        # Extract seasonal using Fourier transform
        x_fft = torch.fft.rfft(x, dim=1)  # (batch, input_len//2+1, n_features)
        
        # Reconstruct with learned harmonics at output times
        seasonal_out = []
        for b in range(x.size(0)):
            batch_seasonal = []
            for feat in range(self.n_features):
                # Sum of sinusoids with learned amplitudes and phases
                feat_seasonal = torch.zeros(self.output_len, device=x.device)
                for p_idx, period in enumerate(self.periods):
                    for h in range(self.n_harmonics):
                        freq = 2 * 3.14159 * (h + 1) / period
                        amp = torch.abs(self.seasonal_amplitude[p_idx, feat, h])
                        phase = self.seasonal_phase[p_idx, feat, h]
                        feat_seasonal = feat_seasonal + amp * torch.sin(freq * t_out.squeeze() + phase)
                batch_seasonal.append(feat_seasonal)
            seasonal_out.append(torch.stack(batch_seasonal, dim=-1))
        seasonal_out = torch.stack(seasonal_out, dim=0)  # (batch, output_len, n_features)
        
        # Also use MLP for seasonal refinement
        seasonal_mlp = self.seasonal_proj(trend_in).permute(0, 2, 1)  # (batch, output_len, n_features)
        seasonal_out = seasonal_out + seasonal_mlp * 0.1  # small MLP correction
        
        # === RESIDUAL COMPONENT ===
        # What's left after removing trend and seasonal (approximate)
        # Use MLP directly
        residual_out = self.residual_encoder(trend_in).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # === COMBINATION ===
        # Normalize decomposition weights
        weights = F.softmax(self.decomp_weights, dim=0)  # (3,)
        
        # Stack components
        combined = torch.cat([
            trend_out * weights[0],
            seasonal_out * weights[1],
            residual_out * weights[2]
        ], dim=-1)  # (batch, output_len, n_features * 3)
        
        # Learned combination
        mix = self.combination(combined)  # (batch, output_len, n_features)
        
        # Weighted sum with learned mixing
        out = trend_out * mix + seasonal_out * (1 - mix) * 0.5 + residual_out * (1 - mix) * 0.5
        # Simplified: direct weighted combination
        out = weights[0] * trend_out + weights[1] * seasonal_out + weights[2] * residual_out
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

