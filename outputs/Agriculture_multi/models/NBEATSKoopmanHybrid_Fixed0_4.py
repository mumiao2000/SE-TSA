import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NBEATSKoopmanHybrid_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_blocks': [1, 3, 6],
            'koopman_dim': [32, 128, 512],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_blocks = self.config.get('n_blocks', 3)
        self.koopman_dim = self.config.get('koopman_dim', 128)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        self.backcast_len = self.input_len
        self.forecast_len = self.output_len
        
        # N-BEATS blocks with Koopman operators
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(self._create_block())
        
        # Final projection
        self.final_proj = nn.Linear(self.n_features * self.forecast_len, self.n_features * self.forecast_len)
        
    def _create_block(self):
        total_theta = self.koopman_dim * 2 + self.backcast_len * self.n_features + self.forecast_len * self.n_features
        return nn.ModuleDict({
            'theta_f': nn.Sequential(
                nn.Linear(self.backcast_len * self.n_features, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, total_theta)
            ),
            'koopman_encoder': nn.Linear(self.n_features, self.koopman_dim),
            'koopman_operator': nn.Linear(self.koopman_dim, self.koopman_dim, bias=False),
            'koopman_decoder': nn.Linear(self.koopman_dim, self.n_features),
        })
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Flatten for block processing: (batch, input_len * n_features)
        x_flat = x.view(x.size(0), -1)  # (batch, input_len * n_features)
        
        forecast = torch.zeros(x.size(0), self.forecast_len, self.n_features, device=x.device)  # (batch, output_len, n_features)
        backcast = x_flat  # (batch, input_len * n_features)
        
        for block in self.blocks:
            # Generate parameters
            theta = block['theta_f'](backcast)  # (batch, koopman_dim * 2 + backcast_len * n_features + forecast_len * n_features)
            
            # Split theta into four parts
            k_encode = theta[:, :self.koopman_dim]  # (batch, koopman_dim)
            k_operator = theta[:, self.koopman_dim:2*self.koopman_dim]  # (batch, koopman_dim)
            backcast_basis = theta[:, 2*self.koopman_dim:2*self.koopman_dim + self.backcast_len * self.n_features]  # (batch, input_len * n_features)
            forecast_basis = theta[:, 2*self.koopman_dim + self.backcast_len * self.n_features:]  # (batch, forecast_len * n_features)
            
            # Reshape input for Koopman: (batch, input_len, n_features)
            x_3d = backcast.view(x.size(0), self.backcast_len, self.n_features)  # (batch, input_len, n_features)
            
            # Koopman encoding: average over time, then encode
            x_avg = x_3d.mean(dim=1)  # (batch, n_features)
            z = block['koopman_encoder'](x_avg)  # (batch, koopman_dim)
            z = torch.tanh(z)  # (batch, koopman_dim)
            
            # Apply Koopman operator (diagonal for stability)
            k_diag = torch.sigmoid(k_operator)  # (batch, koopman_dim)
            z_evolved = z * k_diag  # (batch, koopman_dim)
            
            # Decode
            koopman_out = block['koopman_decoder'](z_evolved)  # (batch, n_features)
            
            # Expand to forecast length
            koopman_forecast = koopman_out.unsqueeze(1).expand(-1, self.forecast_len, -1)  # (batch, output_len, n_features)
            
            # Basis forecast
            basis_forecast = forecast_basis.view(x.size(0), self.forecast_len, self.n_features)  # (batch, output_len, n_features)
            
            # Combine
            block_forecast = koopman_forecast + basis_forecast  # (batch, output_len, n_features)
            forecast = forecast + block_forecast  # (batch, output_len, n_features)
            
            # Update backcast with matching dimensions
            backcast = backcast - backcast_basis * 0.1  # (batch, input_len * n_features)
        
        # Final projection
        forecast_flat = forecast.view(forecast.size(0), -1)  # (batch, output_len * n_features)
        forecast_flat = self.final_proj(forecast_flat)  # (batch, output_len * n_features)
        out = forecast_flat.view(x.size(0), self.forecast_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
