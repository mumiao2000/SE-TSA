import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NBEATSKoopmanHybrid_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'stack_width': [32, 128, 512],
            'n_stacks': [2, 4, 8],
            'koopman_dim': [16, 64, 256],
            'theta_dim': [4, 8, 16]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.stack_width = self.config.get('stack_width', 128)
        self.n_stacks = self.config.get('n_stacks', 4)
        self.koopman_dim = self.config.get('koopman_dim', 64)
        self.theta_dim = self.config.get('theta_dim', 8)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Shared input projection per feature
        self.input_proj = nn.Linear(self.input_len, self.stack_width)
        
        # N-BEATS style stacks with Koopman operator
        # Use custom Module for each stack to properly hold Parameters
        self.stacks = nn.ModuleList()
        for _ in range(self.n_stacks):
            stack = nn.Module()
            stack.backcast_fc = nn.Sequential(
                nn.Linear(self.stack_width, self.stack_width),
                nn.ReLU(),
                nn.Linear(self.stack_width, self.theta_dim)
            )
            stack.forecast_fc = nn.Sequential(
                nn.Linear(self.stack_width, self.stack_width),
                nn.ReLU(),
                nn.Linear(self.stack_width, self.theta_dim)
            )
            stack.koopman_A = nn.Parameter(torch.randn(self.koopman_dim, self.koopman_dim) * 0.01)
            self.stacks.append(stack)
        
        # Trend and seasonality basis functions
        self.trend_basis = nn.Parameter(torch.randn(self.theta_dim // 2, self.output_len), requires_grad=True)
        self.season_basis = nn.Parameter(torch.randn(self.theta_dim // 2, self.output_len), requires_grad=True)
        
        # Koopman embedding network
        self.koopman_encoder = nn.Linear(self.theta_dim, self.koopman_dim)
        self.koopman_decoder = nn.Linear(self.koopman_dim, self.theta_dim)
        
        # Combine stacks
        self.combiner = nn.Linear(self.n_stacks * self.theta_dim, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Process each feature independently
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.input_proj(x)  # (batch, n_features, stack_width)
        
        # Apply stacks per feature
        forecasts = []
        for stack in self.stacks:
            # Backcast and forecast thetas
            backcast_theta = stack.backcast_fc(x)  # (batch, n_features, theta_dim)
            forecast_theta = stack.forecast_fc(x)  # (batch, n_features, theta_dim)
            
            # Apply Koopman operator in latent space
            k_embed = self.koopman_encoder(forecast_theta)  # (batch, n_features, koopman_dim)
            # Koopman evolution: simple linear dynamics
            k_evolved = k_embed @ stack.koopman_A.T  # (batch, n_features, koopman_dim)
            forecast_theta = self.koopman_decoder(k_evolved)  # (batch, n_features, theta_dim)
            
            # Generate forecast using basis
            trend_part = forecast_theta[:, :, :self.theta_dim//2] @ self.trend_basis  # (batch, n_features, output_len)
            season_part = forecast_theta[:, :, self.theta_dim//2:] @ self.season_basis  # (batch, n_features, output_len)
            forecast = trend_part + season_part  # (batch, n_features, output_len)
            forecasts.append(forecast)
        
        # Combine all stack forecasts
        forecast = torch.stack(forecasts, dim=-1)  # (batch, n_features, output_len, n_stacks)
        forecast = forecast.mean(dim=-1)  # (batch, n_features, output_len)
        
        forecast = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
        forecast = self.revin(forecast, mode='denorm')  # (batch, output_len, n_features)
        return forecast

