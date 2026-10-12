import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NHiTSChannel(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'stack_types': [1, 2, 3],
            'n_blocks': [1, 2, 3],
            'n_pool_kernel_size': [2, 4, 8],
            'n_theta': [64, 256, 512],
            'hidden_units': [64, 128, 256]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_stacks = self.config.get('stack_types', 2)
        self.n_blocks = self.config.get('n_blocks', 2)
        self.n_pool_kernel_size = self.config.get('n_pool_kernel_size', 4)
        self.n_theta = self.config.get('n_theta', 256)
        self.hidden_units = self.config.get('hidden_units', 128)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Build stacks
        self.stacks = nn.ModuleList()
        for i in range(self.n_stacks):
            pool_size = self.n_pool_kernel_size ** i
            blocks = nn.ModuleList()
            for _ in range(self.n_blocks):
                blocks.append(NHiTSBlock(
                    self.input_len, self.output_len, self.n_features,
                    self.n_theta, self.hidden_units, pool_size
                ))
            self.stacks.append(blocks)
        
        # Final residual
        self.residual = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save residual
        x_res = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_res = self.residual(x_res)  # (batch, n_features, output_len)
        x_res = x_res.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Initialize forecast and backcast
        forecast = torch.zeros(x.shape[0], self.output_len, self.n_features, device=x.device)  # (batch, output_len, n_features)
        
        for stack in self.stacks:
            x_stack = x  # (batch, input_len, n_features)
            for block in stack:
                backcast, block_forecast = block(x_stack)  # both (batch, input_len/output_len, n_features)
                x_stack = x_stack - backcast  # (batch, input_len, n_features)
                forecast = forecast + block_forecast  # (batch, output_len, n_features)
        
        # Add residual
        forecast = forecast + x_res  # (batch, output_len, n_features)
        
        forecast = self.revin(forecast, mode='denorm')  # (batch, output_len, n_features)
        return forecast


class NHiTSBlock(nn.Module):
    def __init__(self, input_len, output_len, n_features, n_theta, hidden_units, pool_size):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.pool_size = pool_size
        
        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, ceil_mode=True)
        
        # Calculate pooled length
        self.pooled_len = (input_len + pool_size - 1) // pool_size
        
        # FC layers for theta
        self.fc = nn.Sequential(
            nn.Linear(self.pooled_len, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, n_theta),
        )
        
        # Basis expansion for backcast and forecast
        self.backcast_basis = nn.Linear(n_theta, self.pooled_len)
        self.forecast_basis = nn.Linear(n_theta, output_len)
        
        # Upsample backcast
        self.upsample = nn.Upsample(size=input_len, mode='linear', align_corners=False)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Pool
        x_pooled = self.pool(x_perm)  # (batch, n_features, pooled_len)
        
        # Process each feature independently
        batch_size = x.shape[0]
        
        # Flatten for batch processing
        x_flat = x_pooled.reshape(batch_size * self.n_features, self.pooled_len)  # (batch*n_features, pooled_len)
        
        # Theta
        theta = self.fc(x_flat)  # (batch*n_features, n_theta)
        
        # Backcast and forecast
        backcast = self.backcast_basis(theta)  # (batch*n_features, pooled_len)
        forecast = self.forecast_basis(theta)  # (batch*n_features, output_len)
        
        # Reshape
        backcast = backcast.reshape(batch_size, self.n_features, self.pooled_len)  # (batch, n_features, pooled_len)
        forecast = forecast.reshape(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        
        # Upsample backcast to input_len
        backcast = self.upsample(backcast)  # (batch, n_features, input_len)
        
        # Permute back
        backcast = backcast.permute(0, 2, 1)  # (batch, input_len, n_features)
        forecast = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        return backcast, forecast
