import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class RevINNBeatsEnhanced(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_blocks': [2, 4, 8],
            'n_layers_per_block': [2, 4, 8],
            'dropout': [0.0, 0.3],
            'trend_poly_degree': [2, 4],
            'n_harmonics': [4, 16, 64]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_blocks = self.config.get('n_blocks', 4)
        self.n_layers_per_block = self.config.get('n_layers_per_block', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.trend_poly_degree = self.config.get('trend_poly_degree', 3)
        self.n_harmonics = self.config.get('n_harmonics', 16)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Direct skip connection for stable training
        self.skip_linear = nn.Linear(self.input_len, self.output_len)
        
        # Stack of N-BEATS blocks with trend/seasonality decomposition
        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            self.blocks.append(
                NBeatsDecompBlock(
                    input_len=self.input_len,
                    output_len=self.output_len,
                    n_features=self.n_features,
                    hidden_dim=self.hidden_dim,
                    n_layers=self.n_layers_per_block,
                    dropout=self.dropout,
                    trend_poly_degree=self.trend_poly_degree,
                    n_harmonics=self.n_harmonics,
                    is_last=(i == self.n_blocks - 1)
                )
            )
        
        # Final projection to combine block outputs with skip
        self.final_proj = nn.Linear(self.n_features * 2, self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Direct skip connection
        skip = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        skip = self.skip_linear(skip)  # (batch, n_features, output_len)
        skip = skip.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Residual through blocks
        residual = x  # (batch, input_len, n_features)
        block_outputs = []
        
        for block in self.blocks:
            backcast, forecast = block(residual)  # both (batch, input_len/output_len, n_features)
            residual = residual - backcast  # (batch, input_len, n_features)
            block_outputs.append(forecast)  # (batch, output_len, n_features)
        
        # Sum all block forecasts
        forecast_sum = torch.stack(block_outputs, dim=0).sum(dim=0)  # (batch, output_len, n_features)
        
        # Combine with skip connection
        combined = torch.cat([forecast_sum, skip], dim=-1)  # (batch, output_len, n_features * 2)
        out = self.final_proj(combined)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class NBeatsDecompBlock(nn.Module):
    def __init__(self, input_len, output_len, n_features, hidden_dim, n_layers, dropout,
                 trend_poly_degree, n_harmonics, is_last):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.trend_poly_degree = trend_poly_degree
        self.n_harmonics = n_harmonics
        self.is_last = is_last
        
        # FC stack for feature extraction (channel-dependent)
        layers = []
        in_dim = input_len * n_features
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.fc_stack = nn.Sequential(*layers)
        
        # Trend branch: polynomial basis
        self.trend_theta = nn.Linear(hidden_dim, (trend_poly_degree + 1) * n_features)
        
        # Seasonality branch: Fourier basis
        self.season_theta = nn.Linear(hidden_dim, 2 * n_harmonics * n_features)
        
        # Build trend basis matrices
        t_in = torch.arange(input_len, dtype=torch.float32) / input_len  # (input_len,)
        t_out = torch.arange(output_len, dtype=torch.float32) / output_len  # (output_len,)
        
        trend_basis_in = torch.stack([t_in ** p for p in range(trend_poly_degree + 1)], dim=1)  # (input_len, trend_poly_degree+1)
        trend_basis_out = torch.stack([t_out ** p for p in range(trend_poly_degree + 1)], dim=1)  # (output_len, trend_poly_degree+1)
        
        self.register_buffer('trend_basis_in', trend_basis_in)
        self.register_buffer('trend_basis_out', trend_basis_out)
        
        # Build seasonality basis matrices (Fourier)
        freq = torch.arange(1, n_harmonics + 1, dtype=torch.float32).unsqueeze(0)  # (1, n_harmonics)
        t_in_expanded = t_in.unsqueeze(1) * 2 * 3.14159  # (input_len, 1)
        t_out_expanded = t_out.unsqueeze(1) * 2 * 3.14159  # (output_len, 1)
        
        season_basis_in = torch.cat([
            torch.sin(t_in_expanded * freq),  # (input_len, n_harmonics)
            torch.cos(t_in_expanded * freq)   # (input_len, n_harmonics)
        ], dim=1)  # (input_len, 2*n_harmonics)
        
        season_basis_out = torch.cat([
            torch.sin(t_out_expanded * freq),  # (output_len, n_harmonics)
            torch.cos(t_out_expanded * freq)   # (output_len, n_harmonics)
        ], dim=1)  # (output_len, 2*n_harmonics)
        
        self.register_buffer('season_basis_in', season_basis_in)
        self.register_buffer('season_basis_out', season_basis_out)
        
        # Backcast projection (reconstruct input)
        self.backcast_proj = nn.Linear(hidden_dim, input_len * n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # (batch, input_len * n_features)
        
        h = self.fc_stack(x_flat)  # (batch, hidden_dim)
        
        # Trend component
        trend_coeffs = self.trend_theta(h)  # (batch, (trend_poly_degree+1) * n_features)
        trend_coeffs = trend_coeffs.reshape(batch_size, self.n_features, self.trend_poly_degree + 1)  # (batch, n_features, trend_poly_degree+1)
        
        trend_backcast = torch.bmm(trend_coeffs, self.trend_basis_in.t().unsqueeze(0).expand(batch_size, -1, -1))  # (batch, n_features, input_len)
        trend_forecast = torch.bmm(trend_coeffs, self.trend_basis_out.t().unsqueeze(0).expand(batch_size, -1, -1))  # (batch, n_features, output_len)
        
        # Seasonality component
        season_coeffs = self.season_theta(h)  # (batch, 2*n_harmonics * n_features)
        season_coeffs = season_coeffs.reshape(batch_size, self.n_features, 2 * self.n_harmonics)  # (batch, n_features, 2*n_harmonics)
        
        season_backcast = torch.bmm(season_coeffs, self.season_basis_in.t().unsqueeze(0).expand(batch_size, -1, -1))  # (batch, n_features, input_len)
        season_forecast = torch.bmm(season_coeffs, self.season_basis_out.t().unsqueeze(0).expand(batch_size, -1, -1))  # (batch, n_features, output_len)
        
        # Combine decompositions
        backcast = trend_backcast + season_backcast  # (batch, n_features, input_len)
        forecast = trend_forecast + season_forecast  # (batch, n_features, output_len)
        
        # Add residual backcast from direct projection for flexibility
        backcast_resid = self.backcast_proj(h).reshape(batch_size, self.n_features, self.input_len)  # (batch, n_features, input_len)
        backcast = backcast + 0.1 * backcast_resid  # (batch, n_features, input_len)
        
        # Transpose to standard shape
        backcast = backcast.permute(0, 2, 1)  # (batch, input_len, n_features)
        forecast = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        return backcast, forecast
