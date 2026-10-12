import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NBeatsRevIN_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_stacks': [2, 4, 8],
            'n_blocks': [1, 3, 5],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_stacks = self.config.get('n_stacks', 4)
        self.n_blocks = self.config.get('n_blocks', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        self.stacks = nn.ModuleList()
        for s in range(self.n_stacks):
            is_trend = (s % 2 == 0)
            self.stacks.append(
                NBeatsStack(
                    input_len, output_len, self.hidden_dim, 
                    self.n_blocks, is_trend, self.dropout_rate
                )
            )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Process each feature independently (channel-independent N-BEATS)
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = x.reshape(batch_size * self.n_features, self.input_len)  # (batch*n_features, input_len)
        x = x.unsqueeze(-1)  # (batch*n_features, input_len, 1)
        
        backcast = x
        forecast = torch.zeros(batch_size * self.n_features, self.output_len, 1, device=x.device)  # (batch*n_features, output_len, 1)
        
        for stack in self.stacks:
            b, f = stack(backcast)  # b: (batch*n_features, input_len, 1), f: (batch*n_features, output_len, 1)
            backcast = backcast - b
            forecast = forecast + f
        
        forecast = forecast.squeeze(-1)  # (batch*n_features, output_len)
        forecast = forecast.reshape(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        forecast = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        forecast = self.revin(forecast, mode='denorm')  # (batch, output_len, n_features)
        return forecast


class NBeatsBlock(nn.Module):
    def __init__(self, input_len, output_len, hidden_dim, is_trend, dropout_rate):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.is_trend = is_trend
        
        self.fc = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # For trend: polynomial degree, for seasonality: number of harmonics
        if is_trend:
            self.theta_b_dim = 2  # linear trend: intercept and slope
            self.theta_f_dim = 2
        else:
            self.theta_b_dim = input_len // 2  # seasonality harmonics
            self.theta_f_dim = output_len // 2
        
        self.theta_b = nn.Linear(hidden_dim, self.theta_b_dim)
        self.theta_f = nn.Linear(hidden_dim, self.theta_f_dim)
        
        # Create basis matrices
        if is_trend:
            # Trend basis: [1, t, t^2, ...] for polynomial
            backcast_t = torch.linspace(-1, 1, steps=input_len)  # (input_len,)
            forecast_t = torch.linspace(0, 1, steps=output_len)  # (output_len,)
            self.register_buffer('backcast_basis', torch.stack([backcast_t ** i for i in range(self.theta_b_dim)], dim=1))  # (input_len, theta_b_dim)
            self.register_buffer('forecast_basis', torch.stack([forecast_t ** i for i in range(self.theta_f_dim)], dim=1))  # (output_len, theta_f_dim)
        else:
            # Seasonality basis: Fourier harmonics
            backcast_t = torch.linspace(0, 2 * 3.14159 * (input_len - 1) / input_len, steps=input_len)  # (input_len,)
            forecast_t = torch.linspace(0, 2 * 3.14159 * (output_len - 1) / output_len, steps=output_len)  # (output_len,)
            
            backcast_basis_list = []
            forecast_basis_list = []
            for i in range(1, self.theta_b_dim + 1):
                backcast_basis_list.append(torch.cos(i * backcast_t))
                backcast_basis_list.append(torch.sin(i * backcast_t))
            for i in range(1, self.theta_f_dim + 1):
                forecast_basis_list.append(torch.cos(i * forecast_t))
                forecast_basis_list.append(torch.sin(i * forecast_t))
            
            # Trim to exact dimensions
            self.register_buffer('backcast_basis', torch.stack(backcast_basis_list[:self.theta_b_dim], dim=1))  # (input_len, theta_b_dim)
            self.register_buffer('forecast_basis', torch.stack(forecast_basis_list[:self.theta_f_dim], dim=1))  # (output_len, theta_f_dim)
        
    def forward(self, x):
        # x: (batch, input_len, 1)
        x_flat = x.squeeze(-1)  # (batch, input_len)
        h = self.fc(x_flat)  # (batch, hidden_dim)
        
        theta_b = self.theta_b(h)  # (batch, theta_b_dim)
        theta_f = self.theta_f(h)  # (batch, theta_f_dim)
        
        # Apply basis functions: (batch, theta_dim) @ (theta_dim, seq_len) -> (batch, seq_len)
        backcast = torch.matmul(theta_b, self.backcast_basis.t())  # (batch, input_len)
        forecast = torch.matmul(theta_f, self.forecast_basis.t())  # (batch, output_len)
        
        backcast = backcast.unsqueeze(-1)  # (batch, input_len, 1)
        forecast = forecast.unsqueeze(-1)  # (batch, output_len, 1)
        
        return backcast, forecast


class NBeatsStack(nn.Module):
    def __init__(self, input_len, output_len, hidden_dim, n_blocks, is_trend, dropout_rate):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_len, output_len, hidden_dim, is_trend, dropout_rate)
            for _ in range(n_blocks)
        ])
        
    def forward(self, x):
        # x: (batch, input_len, 1)
        backcast = x
        forecast = torch.zeros(x.shape[0], self.blocks[0].output_len, 1, device=x.device)  # (batch, output_len, 1)
        
        for block in self.blocks:
            b, f = block(backcast)
            backcast = backcast - b
            forecast = forecast + f
        
        return backcast, forecast
