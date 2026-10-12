import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralBasisExpansion_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_bases': [8, 32, 128],
            'hidden_dim': [64, 256, 1024],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_bases = self.config.get('n_bases', 32)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Learnable basis functions: (n_bases, output_len)
        # We parameterize bases as smooth functions using a small MLP evaluated at output positions
        self.basis_freqs = nn.Parameter(torch.randn(self.n_bases) * 0.1)
        self.basis_phases = nn.Parameter(torch.randn(self.n_bases) * 0.1)
        self.basis_amps = nn.Parameter(torch.ones(self.n_bases) * 0.1)
        
        # MLP to predict basis coefficients from input statistics
        # We use summary statistics: mean, std, trend, last values
        self.stat_dim = 4  # mean, std, linear trend slope, last value
        
        # Encoder: process each feature's statistics
        encoder_layers = []
        in_dim = self.stat_dim
        for i in range(self.n_layers):
            out_dim = self.hidden_dim if i < self.n_layers - 1 else self.n_bases
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.coeff_encoder = nn.Sequential(*encoder_layers)
        
        # Feature-specific trend and bias
        self.trend_proj = nn.Linear(self.input_len, 1)
        self.bias = nn.Parameter(torch.zeros(self.n_features))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Compute statistics per feature
        # mean: (batch, n_features)
        x_mean = x.mean(dim=1)  # (batch, n_features)
        x_std = x.std(dim=1, unbiased=False) + 1e-5  # (batch, n_features)
        
        # Linear trend: fit least squares slope
        t = torch.arange(self.input_len, dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(-1)  # (1, input_len, 1)
        t_normalized = (t - self.input_len / 2) / (self.input_len / 2)  # (1, input_len, 1)
        
        # Compute slope for each feature: cov(t, x) / var(t)
        x_centered = x - x_mean.unsqueeze(1)  # (batch, input_len, n_features)
        t_centered = t_normalized.squeeze(-1)  # (1, input_len)
        slope = (x_centered * t_centered.unsqueeze(-1)).sum(dim=1) / (t_centered ** 2).sum()  # (batch, n_features)
        
        # Last value
        x_last = x[:, -1, :]  # (batch, n_features)
        
        # Combine statistics: (batch, n_features, stat_dim)
        stats = torch.stack([x_mean, x_std, slope, x_last], dim=-1)  # (batch, n_features, stat_dim)
        
        # Predict basis coefficients for each feature
        # (batch * n_features, stat_dim) -> (batch * n_features, n_bases)
        batch_size = x.size(0)
        stats_flat = stats.view(batch_size * self.n_features, self.stat_dim)  # (batch * n_features, stat_dim)
        coeffs = self.coeff_encoder(stats_flat)  # (batch * n_features, n_bases)
        coeffs = coeffs.view(batch_size, self.n_features, self.n_bases)  # (batch, n_features, n_bases)
        
        # Generate basis functions at output time points
        # output positions: (output_len,)
        out_t = torch.arange(self.output_len, dtype=x.dtype, device=x.device)  # (output_len,)
        # Normalize to [0, 1] for stability
        out_t_norm = out_t / self.output_len  # (output_len,)
        
        # Basis: sin(2*pi*freq*t + phase) with learnable parameters
        # Expand for broadcasting: (n_bases, output_len)
        bases = self.basis_amps.unsqueeze(-1) * torch.sin(
            2 * 3.14159265 * self.basis_freqs.unsqueeze(-1) * out_t_norm.unsqueeze(0) 
            + self.basis_phases.unsqueeze(-1)
        )  # (n_bases, output_len)
        
        # Add linear trend basis
        trend_basis = out_t_norm.unsqueeze(0)  # (1, output_len)
        bases = torch.cat([bases, trend_basis], dim=0)  # (n_bases + 1, output_len)
        
        # Pad coefficients for trend
        trend_coeff = self.trend_proj(x.permute(0, 2, 1)).squeeze(-1)  # (batch, n_features)
        coeffs = torch.cat([coeffs, trend_coeff.unsqueeze(-1)], dim=-1)  # (batch, n_features, n_bases + 1)
        
        # Combine: (batch, n_features, n_bases + 1) @ (n_bases + 1, output_len)
        # -> (batch, n_features, output_len)
        # Use einsum for proper broadcasting: b=batch, f=feature, k=basis, t=time
        forecast = torch.einsum('bfk,kt->bft', coeffs, bases)  # (batch, n_features, output_len)
        
        # Add bias
        forecast = forecast + self.bias.unsqueeze(0).unsqueeze(-1)  # (batch, n_features, output_len)
        
        # Transpose to (batch, output_len, n_features)
        forecast = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        forecast = self.revin(forecast, mode='denorm')  # (batch, output_len, n_features)
        return forecast

