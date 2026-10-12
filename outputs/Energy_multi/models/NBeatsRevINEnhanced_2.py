import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NBeatsRevINEnhanced(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_stacks': [2, 4, 8],
            'n_blocks': [1, 3, 5],
            'dropout': [0.0, 0.15, 0.3],
            'seasonality_harmonics': [4, 8, 16],
            'mixing_dim': [16, 64, 256],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_stacks = self.config.get('n_stacks', 4)
        self.n_blocks = self.config.get('n_blocks', 3)
        self.dropout = self.config.get('dropout', 0.1)
        self.seasonality_harmonics = self.config.get('seasonality_harmonics', 8)
        self.mixing_dim = self.config.get('mixing_dim', 64)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # N-BEATS stacks: alternating trend and seasonality
        self.stacks = nn.ModuleList()
        self.stack_gates = nn.ModuleList()  # Learnable residual gates between stacks
        
        for s in range(self.n_stacks):
            is_trend = (s % 2 == 0)
            stack_blocks = nn.ModuleList()
            for b in range(self.n_blocks):
                stack_blocks.append(
                    NBeatsBlockEnhanced(
                        input_len=self.input_len,
                        output_len=self.output_len,
                        n_features=self.n_features,
                        hidden_dim=self.hidden_dim,
                        dropout=self.dropout,
                        is_trend=is_trend,
                        seasonality_harmonics=self.seasonality_harmonics,
                    )
                )
            self.stacks.append(stack_blocks)
            # Learnable gate for residual connection from this stack
            if s > 0:
                self.stack_gates.append(
                    nn.Sequential(
                        nn.Linear(self.n_features, self.n_features),
                        nn.Sigmoid(),
                    )
                )
            else:
                self.stack_gates.append(None)
        
        # Lightweight cross-feature mixing with bottleneck
        self.mixing = nn.Sequential(
            nn.Linear(self.n_features, self.mixing_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mixing_dim, self.n_features),
        )
        
        # Final residual weight
        self.output_gate = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Process each feature independently through N-BEATS
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Initialize forecast and backcast accumulators
        forecast = torch.zeros(batch_size, self.output_len, self.n_features, device=x.device)  # (batch, output_len, n_features)
        
        for s, (stack_blocks, gate) in enumerate(zip(self.stacks, self.stack_gates)):
            stack_forecast = torch.zeros(batch_size, self.output_len, self.n_features, device=x.device)  # (batch, output_len, n_features)
            
            for block in stack_blocks:
                block_forecast = block(x)  # (batch, output_len, n_features)
                stack_forecast = stack_forecast + block_forecast  # (batch, output_len, n_features)
            
            # Apply learnable residual gate between stacks
            if s > 0 and gate is not None:
                # Compute gate based on current forecast
                gate_val = self.output_gate(forecast.mean(dim=1))  # (batch, n_features)
                gate_val = gate_val.unsqueeze(1)  # (batch, 1, n_features)
                forecast = forecast + gate_val * stack_forecast  # (batch, output_len, n_features)
            else:
                forecast = forecast + stack_forecast  # (batch, output_len, n_features)
        
        # Cross-feature mixing with residual
        mixed = self.mixing(forecast)  # (batch, output_len, n_features)
        mix_gate = self.output_gate(forecast.mean(dim=1)).unsqueeze(1)  # (batch, 1, n_features)
        forecast = forecast + mix_gate * mixed  # (batch, output_len, n_features)
        
        # Denormalize
        forecast = self.revin(forecast, mode='denorm')  # (batch, output_len, n_features)
        return forecast  # (batch, output_len, n_features)


class NBeatsBlockEnhanced(nn.Module):
    def __init__(self, input_len, output_len, n_features, hidden_dim, dropout, is_trend, seasonality_harmonics):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.is_trend = is_trend
        self.seasonality_harmonics = seasonality_harmonics
        
        # Fully connected stack for theta generation
        self.fc = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        if is_trend:
            # Trend: polynomial basis of degree 2
            self.theta_f = nn.Linear(hidden_dim, 2)  # 2 polynomial coefficients
            self.theta_b = nn.Linear(hidden_dim, 2)
            # Precompute trend basis
            t_forecast = torch.linspace(0, 1, output_len)  # (output_len,)
            t_backcast = torch.linspace(0, 1, input_len)   # (input_len,)
            self.register_buffer('t_forecast', t_forecast)
            self.register_buffer('t_backcast', t_backcast)
        else:
            # Seasonality: Fourier basis
            self.theta_f = nn.Linear(hidden_dim, 2 * seasonality_harmonics)
            self.theta_b = nn.Linear(hidden_dim, 2 * seasonality_harmonics)
            # Precompute seasonality basis
            t_forecast = torch.arange(output_len).float()  # (output_len,)
            t_backcast = torch.arange(input_len).float()   # (input_len,)
            # Harmonics: sin(2*pi*k*t/T) and cos(2*pi*k*t/T)
            harmonics_f = []
            harmonics_b = []
            for k in range(1, seasonality_harmonics + 1):
                harmonics_f.append(torch.sin(2 * 3.14159265 * k * t_forecast / output_len))
                harmonics_f.append(torch.cos(2 * 3.14159265 * k * t_forecast / output_len))
                harmonics_b.append(torch.sin(2 * 3.14159265 * k * t_backcast / input_len))
                harmonics_b.append(torch.cos(2 * 3.14159265 * k * t_backcast / input_len))
            self.register_buffer('harmonics_f', torch.stack(harmonics_f, dim=0))  # (2*harmonics, output_len)
            self.register_buffer('harmonics_b', torch.stack(harmonics_b, dim=0))  # (2*harmonics, input_len)
        
        # Learnable gating between forecast and backcast influence
        self.gate_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Process each feature independently
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = x.reshape(batch_size * self.n_features, self.input_len)  # (batch*n_features, input_len)
        
        # Generate hidden representation
        h = self.fc(x)  # (batch*n_features, hidden_dim)
        
        # Generate basis coefficients
        theta_f = self.theta_f(h)  # (batch*n_features, basis_dim)
        theta_b = self.theta_b(h)  # (batch*n_features, basis_dim)
        
        # Generate gate value
        gate = torch.sigmoid(self.gate_fc(h))  # (batch*n_features, 1)
        
        if self.is_trend:
            # Trend basis: [1, t]
            basis_f = torch.stack([
                torch.ones_like(self.t_forecast),
                self.t_forecast,
            ], dim=0)  # (2, output_len)
            basis_b = torch.stack([
                torch.ones_like(self.t_backcast),
                self.t_backcast,
            ], dim=0)  # (2, input_len)
        else:
            # Seasonality basis: precomputed harmonics
            basis_f = self.harmonics_f  # (2*harmonics, output_len)
            basis_b = self.harmonics_b  # (2*harmonics, input_len)
        
        # Compute forecast and backcast
        forecast = torch.matmul(theta_f, basis_f)  # (batch*n_features, output_len)
        backcast = torch.matmul(theta_b, basis_b)  # (batch*n_features, input_len)
        
        # Reshape back
        forecast = forecast.reshape(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        forecast = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Backcast is used implicitly through residual (not returned in this simplified version)
        # The parent model accumulates forecasts; we follow that pattern
        
        return forecast  # (batch, output_len, n_features)
