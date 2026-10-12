import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NBeatsRevINHybrid(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_stacks': [2, 4, 8],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_stacks = self.config.get('n_stacks', 4)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Trend stack: polynomial basis
        self.trend_stack = nn.ModuleList()
        for _ in range(self.n_stacks // 2):
            layers = []
            for i in range(self.n_layers):
                in_d = self.n_features * self.input_len if i == 0 else self.hidden_dim
                layers.append(nn.Linear(in_d, self.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.hidden_dim, self.n_features * (self.output_len + 3)))  # 3 polynomial coeffs
            self.trend_stack.append(nn.Sequential(*layers))
        
        # Seasonality stack: Fourier basis with dominant period
        self.seasonality_stack = nn.ModuleList()
        self.n_harmonics = 8
        for _ in range(self.n_stacks // 2):
            layers = []
            for i in range(self.n_layers):
                in_d = self.n_features * self.input_len if i == 0 else self.hidden_dim
                layers.append(nn.Linear(in_d, self.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.hidden_dim, self.n_features * self.n_harmonics * 2))  # sin/cos coeffs
            self.seasonality_stack.append(nn.Sequential(*layers))
        
        # Backcast projection for residual learning
        self.backcast_proj = nn.ModuleList([
            nn.Linear(self.n_features * (self.output_len + 3), self.n_features * self.input_len)
            for _ in range(len(self.trend_stack))
        ] + [
            nn.Linear(self.n_features * self.n_harmonics * 2, self.n_features * self.input_len)
            for _ in range(len(self.seasonality_stack))
        ])
        
        # Forecast projection
        self.forecast_proj = nn.ModuleList([
            nn.Linear(self.n_features * (self.output_len + 3), self.n_features * self.output_len)
            for _ in range(len(self.trend_stack))
        ] + [
            nn.Linear(self.n_features * self.n_harmonics * 2, self.n_features * self.output_len)
            for _ in range(len(self.seasonality_stack))
        ])
        
        # Generate time indices for trend
        self.trend_t = torch.arange(self.output_len, dtype=torch.float32).unsqueeze(0) / self.output_len  # (1, output_len)
        
        # Generate Fourier basis for seasonality
        t = torch.arange(self.output_len, dtype=torch.float32).unsqueeze(0)  # (1, output_len)
        harmonics = torch.arange(1, self.n_harmonics + 1, dtype=torch.float32).unsqueeze(0)  # (1, n_harmonics)
        self.season_freqs = 2 * 3.14159 * harmonics / 103.5  # Use dominant period
        self.season_t = t.unsqueeze(-1) * self.season_freqs.unsqueeze(1)  # (1, output_len, n_harmonics)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # (batch, n_features * input_len)
        
        residual = x_flat
        forecast = torch.zeros(batch_size, self.n_features * self.output_len, device=x.device)  # (batch, n_features * output_len)
        
        # Trend stacks
        for i, stack in enumerate(self.trend_stack):
            stack_out = stack(residual)  # (batch, n_features * (output_len + 3))
            
            # Backcast
            backcast = self.backcast_proj[i](stack_out)  # (batch, n_features * input_len)
            residual = residual - backcast  # (batch, n_features * input_len)
            
            # Forecast: polynomial basis [1, t, t^2, t^3]
            coeffs = stack_out.reshape(batch_size, self.n_features, self.output_len + 3)  # (batch, n_features, output_len + 3)
            trend_basis = torch.stack([
                torch.ones(self.output_len, device=x.device),  # (output_len,)
                self.trend_t.squeeze(0),  # (output_len,)
                self.trend_t.squeeze(0) ** 2,  # (output_len,)
                self.trend_t.squeeze(0) ** 3,  # (output_len,)
            ], dim=0)  # (4, output_len)
            
            # Use first 4 coeffs for trend
            trend_coeffs = coeffs[:, :, :4]  # (batch, n_features, 4)
            trend_forecast = torch.einsum('bnf,ft->bnt', trend_coeffs, trend_basis)  # (batch, n_features, output_len)
            trend_forecast = trend_forecast.reshape(batch_size, -1)  # (batch, n_features * output_len)
            
            forecast = forecast + trend_forecast  # (batch, n_features * output_len)
        
        # Seasonality stacks
        for i, stack in enumerate(self.seasonality_stack):
            stack_idx = len(self.trend_stack) + i
            stack_out = stack(residual)  # (batch, n_features * n_harmonics * 2)
            
            # Backcast
            backcast = self.backcast_proj[stack_idx](stack_out)  # (batch, n_features * input_len)
            residual = residual - backcast  # (batch, n_features * input_len)
            
            # Forecast: Fourier basis
            coeffs = stack_out.reshape(batch_size, self.n_features, self.n_harmonics, 2)  # (batch, n_features, n_harmonics, 2)
            
            # sin/cos basis
            sin_basis = torch.sin(self.season_t).to(x.device)  # (1, output_len, n_harmonics)
            cos_basis = torch.cos(self.season_t).to(x.device)  # (1, output_len, n_harmonics)
            
            season_forecast = torch.einsum('bnh,th->bnt', coeffs[:, :, :, 0], sin_basis.squeeze(0))  # (batch, n_features, output_len)
            season_forecast = season_forecast + torch.einsum('bnh,th->bnt', coeffs[:, :, :, 1], cos_basis.squeeze(0))  # (batch, n_features, output_len)
            season_forecast = season_forecast.reshape(batch_size, -1)  # (batch, n_features * output_len)
            
            forecast = forecast + season_forecast  # (batch, n_features * output_len)
        
        forecast = forecast.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        forecast = self.revin(forecast, mode='denorm')  # (batch, output_len, n_features)
        return forecast  # (batch, output_len, n_features)

