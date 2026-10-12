import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanOperator(nn.Module):
    """Proper nn.Module container for Koopman operator matrix."""
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        # Initialize near identity for stable dynamics
        self.K = nn.Parameter(torch.eye(latent_dim) * 0.9 + torch.randn(latent_dim, latent_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(latent_dim))
    
    def forward(self, z):
        # z: (batch, latent_dim)
        return torch.matmul(z, self.K.t()) + self.bias  # (batch, latent_dim)


class NBEATSBlock(nn.Module):
    """N-BEATS block with stack type specialization."""
    def __init__(self, input_len, output_len, n_features, hidden_dim, stack_type, koopman_latent_dim):
        super().__init__()
        self.stack_type = stack_type  # 'trend', 'seasonality', or 'koopman'
        self.n_features = n_features
        self.input_len = input_len
        self.output_len = output_len
        self.koopman_latent_dim = koopman_latent_dim
        
        # Common FC layers
        self.fc = nn.Sequential(
            nn.Linear(input_len * n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        if stack_type == 'trend':
            # Polynomial trend: linear + quadratic
            self.degree = 2
            self.theta_f = nn.Linear(hidden_dim, (self.degree + 1) * n_features)
            self.theta_b = nn.Linear(hidden_dim, (self.degree + 1) * n_features)
            
        elif stack_type == 'seasonality':
            # Fourier basis
            self.n_harmonics = min(8, input_len // 2)
            self.theta_f = nn.Linear(hidden_dim, 2 * self.n_harmonics * n_features)
            self.theta_b = nn.Linear(hidden_dim, 2 * self.n_harmonics * n_features)
            
        else:  # koopman
            self.theta_f = nn.Linear(hidden_dim, koopman_latent_dim)
            self.theta_b = nn.Linear(hidden_dim, koopman_latent_dim)
            self.backcast_proj = nn.Linear(koopman_latent_dim, input_len * n_features)
            self.forecast_proj = nn.Linear(koopman_latent_dim, output_len * n_features)
            # Proper Koopman operator as nn.Module
            self.koopman = KoopmanOperator(koopman_latent_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        device = x.device
        x_flat = x.reshape(batch_size, -1)  # (batch, input_len * n_features)
        h = self.fc(x_flat)  # (batch, hidden_dim)
        
        if self.stack_type == 'trend':
            theta_f = self.theta_f(h).reshape(batch_size, self.n_features, self.degree + 1)  # (batch, n_features, degree+1)
            theta_b = self.theta_b(h).reshape(batch_size, self.n_features, self.degree + 1)
            
            # Compute time indices on-the-fly with proper device
            t_in = torch.arange(self.input_len, dtype=torch.float32, device=device) / self.input_len  # (input_len,)
            t_out = torch.arange(self.output_len, dtype=torch.float32, device=device) / self.output_len + 1.0  # (output_len,)
            
            # Polynomial basis matrices
            t_in_powers = torch.stack([t_in ** d for d in range(self.degree + 1)], dim=1)  # (input_len, degree+1)
            t_out_powers = torch.stack([t_out ** d for d in range(self.degree + 1)], dim=1)  # (output_len, degree+1)
            
            backcast = torch.matmul(theta_b, t_in_powers.t())  # (batch, n_features, input_len)
            forecast = torch.matmul(theta_f, t_out_powers.t())  # (batch, n_features, output_len)
            
            backcast = backcast.permute(0, 2, 1)  # (batch, input_len, n_features)
            forecast = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
            
        elif self.stack_type == 'seasonality':
            theta_f = self.theta_f(h).reshape(batch_size, self.n_features, 2 * self.n_harmonics)  # (batch, n_features, 2*H)
            theta_b = self.theta_b(h).reshape(batch_size, self.n_features, 2 * self.n_harmonics)
            
            # Split into cos/sin coefficients
            cos_coeff_f = theta_f[:, :, :self.n_harmonics]  # (batch, n_features, H)
            sin_coeff_f = theta_f[:, :, self.n_harmonics:]
            cos_coeff_b = theta_b[:, :, :self.n_harmonics]
            sin_coeff_b = theta_b[:, :, self.n_harmonics:]
            
            # Compute time indices and frequencies on-the-fly with proper device
            t_in = torch.arange(self.input_len, dtype=torch.float32, device=device)  # (input_len,)
            t_out = torch.arange(self.output_len, dtype=torch.float32, device=device)  # (output_len,)
            freq_in = 2 * torch.pi * torch.arange(1, self.n_harmonics + 1, dtype=torch.float32, device=device) / self.input_len  # (H,)
            freq_out = 2 * torch.pi * torch.arange(1, self.n_harmonics + 1, dtype=torch.float32, device=device) / self.output_len  # (H,)
            
            # Fourier basis
            t_in_expanded = t_in.unsqueeze(1)  # (input_len, 1)
            t_out_expanded = t_out.unsqueeze(1)  # (output_len, 1)
            
            cos_in = torch.cos(freq_in.unsqueeze(0) * t_in_expanded)  # (input_len, H)
            sin_in = torch.sin(freq_in.unsqueeze(0) * t_in_expanded)  # (input_len, H)
            cos_out = torch.cos(freq_out.unsqueeze(0) * t_out_expanded)  # (output_len, H)
            sin_out = torch.sin(freq_out.unsqueeze(0) * t_out_expanded)  # (output_len, H)
            
            backcast = (torch.matmul(cos_coeff_b, cos_in.t()) + 
                       torch.matmul(sin_coeff_b, sin_in.t()))  # (batch, n_features, input_len)
            forecast = (torch.matmul(cos_coeff_f, cos_out.t()) + 
                       torch.matmul(sin_coeff_f, sin_out.t()))  # (batch, n_features, output_len)
            
            backcast = backcast.permute(0, 2, 1)  # (batch, input_len, n_features)
            forecast = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
            
        else:  # koopman
            z = self.theta_f(h)  # (batch, koopman_latent_dim)
            # Apply Koopman operator for forecast
            z_forecast = self.koopman(z)  # (batch, koopman_latent_dim)
            
            backcast = self.backcast_proj(z).reshape(batch_size, -1, self.n_features)  # (batch, input_len, n_features)
            forecast = self.forecast_proj(z_forecast).reshape(batch_size, -1, self.n_features)  # (batch, output_len, n_features)
        
        return backcast, forecast


class ChannelGate(nn.Module):
    """Channel-wise gating for feature interaction."""
    def __init__(self, n_features, reduction=4):
        super().__init__()
        self.n_features = n_features
        self.gate = nn.Sequential(
            nn.Linear(n_features, n_features // reduction),
            nn.ReLU(),
            nn.Linear(n_features // reduction, n_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        # Global average pooling over time
        gap = x.mean(dim=1)  # (batch, n_features)
        gate = self.gate(gap)  # (batch, n_features)
        return x * gate.unsqueeze(1)  # (batch, seq_len, n_features)


class NBEATSKoopmanHybrid_Fixed1_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_stacks': [2, 4, 8],
            'koopman_latent_dim': [16, 64, 256],
            'dropout': [0.0, 0.3],
            'gate_reduction': [2, 4, 8],
        }
    
    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_stacks = self.config.get('n_stacks', 4)
        self.koopman_latent_dim = self.config.get('koopman_latent_dim', 64)
        self.dropout = self.config.get('dropout', 0.1)
        self.gate_reduction = self.config.get('gate_reduction', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Stack types cycle through trend, seasonality, koopman
        stack_types = ['trend', 'seasonality', 'koopman']
        
        # Proper nn.ModuleList of NBEATSBlock modules
        self.blocks = nn.ModuleList()
        for i in range(self.n_stacks):
            st = stack_types[i % 3]
            self.blocks.append(NBEATSBlock(
                input_len=self.input_len,
                output_len=self.output_len,
                n_features=self.n_features,
                hidden_dim=self.hidden_dim,
                stack_type=st,
                koopman_latent_dim=self.koopman_latent_dim
            ))
        
        # Channel-wise gating for feature interaction
        self.channel_gate = ChannelGate(self.n_features, reduction=self.gate_reduction)
        
        # Residual connection projection
        self.residual_proj = nn.Linear(self.input_len, self.output_len)
        
        # Final fusion with gating
        self.fusion = nn.Sequential(
            nn.Linear(self.n_features * 2, self.n_features),
            nn.Dropout(self.dropout),
        )
        
        self.gate_proj = nn.Sequential(
            nn.Linear(self.n_features * 2, self.n_features),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Apply channel gating to input
        x = self.channel_gate(x)  # (batch, input_len, n_features)
        
        # Residual connection
        residual = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        residual = self.residual_proj(residual)  # (batch, n_features, output_len)
        residual = residual.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # N-BEATS forward with doubly residual stacking
        backcast = x
        forecast_sum = 0
        
        for block in self.blocks:
            b, f = block(backcast)  # b: (batch, input_len, n_features), f: (batch, output_len, n_features)
            backcast = backcast - b  # doubly residual: subtract backcast
            forecast_sum = forecast_sum + f  # accumulate forecasts
        
        # Combine with residual and gate
        combined = torch.cat([forecast_sum, residual], dim=-1)  # (batch, output_len, 2*n_features)
        
        gate = self.gate_proj(combined)  # (batch, output_len, n_features)
        fused = self.fusion(combined)  # (batch, output_len, n_features)
        
        output = gate * fused + (1 - gate) * residual  # (batch, output_len, n_features)
        
        output = self.revin(output, mode='denorm')  # (batch, output_len, n_features)
        return output

