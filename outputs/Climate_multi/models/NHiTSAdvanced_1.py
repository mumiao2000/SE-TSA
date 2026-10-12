import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NHiTSAdvanced(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_blocks': [1, 3, 6],
            'n_pool_kernel_size': [2, 4, 8],
            'n_freq_downsample': [2, 4, 12],
            'hidden_size': [64, 256, 1024],
            'n_hidden_layers': [1, 2, 4],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_blocks = self.config.get('n_blocks', 3)
        self.n_pool_kernel_size = self.config.get('n_pool_kernel_size', 4)
        self.n_freq_downsample = self.config.get('n_freq_downsample', 4)
        self.hidden_size = self.config.get('hidden_size', 256)
        self.n_hidden_layers = self.config.get('n_hidden_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Stack of N-HiTS blocks
        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            pool_size = max(self.n_pool_kernel_size // (2 ** i), 1)
            freq_downsample = max(self.n_freq_downsample * (2 ** i), 1)
            
            self.blocks.append(
                NHiTSBlock(
                    input_len=input_len,
                    output_len=output_len,
                    n_features=n_features,
                    hidden_size=self.hidden_size,
                    n_hidden_layers=self.n_hidden_layers,
                    pool_size=pool_size,
                    freq_downsample=freq_downsample,
                    dropout=self.dropout
                )
            )
        
        # Learnable combination gates
        self.gates = nn.Parameter(torch.ones(self.n_blocks))
        
        # Final refinement MLP
        self.refinement = nn.Sequential(
            nn.Linear(n_features, n_features * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(n_features * 2, n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x_raw = x  # (batch, input_len, n_features)
        
        # Instance normalization
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.norm(x)  # (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Get statistics
        mean = x_raw.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        std = x_raw.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        
        # Normalize
        x = (x_raw - mean) / std  # (batch, input_len, n_features)
        
        # Process through blocks
        residuals = x  # (batch, input_len, n_features)
        outputs = []
        
        for i, block in enumerate(self.blocks):
            block_out, residuals = block(residuals)  # both (batch, output_len, n_features)
            outputs.append(block_out)
        
        # Gated combination
        gates = F.softmax(self.gates, dim=0)  # (n_blocks,)
        out = sum(g * o for g, o in zip(gates, outputs))  # (batch, output_len, n_features)
        
        # Refinement
        out = self.refinement(out) + out  # (batch, output_len, n_features)
        
        # Denormalize
        out = out * std + mean  # (batch, output_len, n_features)
        
        return out


class NHiTSBlock(nn.Module):
    def __init__(self, input_len, output_len, n_features, hidden_size, n_hidden_layers, 
                 pool_size, freq_downsample, dropout):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.pool_size = pool_size
        self.freq_downsample = freq_downsample
        
        # Max pooling for input
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0)
        
        # Calculate pooled length
        self.pooled_len = input_len // pool_size
        self.pooled_len = max(self.pooled_len, 1)
        
        # Theta backcast and forecast lengths
        self.n_theta_backcast = max(self.pooled_len // freq_downsample, 1)
        self.n_theta_forecast = max(output_len // freq_downsample, 1)
        
        # Fully connected stack
        layers = []
        in_size = self.pooled_len
        for _ in range(n_hidden_layers):
            layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_size = hidden_size
        
        # Output: backcast theta + forecast theta for each feature
        layers.append(nn.Linear(hidden_size, self.n_theta_backcast + self.n_theta_forecast))
        self.fc = nn.Sequential(*layers)
        
        # Interpolation functions
        self.backcast_interpolation = nn.Linear(self.n_theta_backcast, self.pooled_len)
        self.forecast_interpolation = nn.Linear(self.n_theta_forecast, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # Pool and flatten per feature
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_pooled = self.pool(x)  # (batch, n_features, pooled_len)
        
        # Process each feature
        batch_size, n_feat, pooled_len = x_pooled.shape
        x_flat = x_pooled.reshape(batch_size * n_feat, pooled_len)  # (batch*n_features, pooled_len)
        
        # FC forward
        theta = self.fc(x_flat)  # (batch*n_features, n_theta_backcast + n_theta_forecast)
        
        # Split
        theta_backcast = theta[:, :self.n_theta_backcast]  # (batch*n_features, n_theta_backcast)
        theta_forecast = theta[:, self.n_theta_backcast:]  # (batch*n_features, n_theta_forecast)
        
        # Interpolate
        backcast = self.backcast_interpolation(theta_backcast)  # (batch*n_features, pooled_len)
        forecast = self.forecast_interpolation(theta_forecast)  # (batch*n_features, output_len)
        
        # Reshape
        backcast = backcast.reshape(batch_size, n_feat, self.pooled_len)  # (batch, n_features, pooled_len)
        forecast = forecast.reshape(batch_size, n_feat, self.output_len)  # (batch, n_features, output_len)
        
        # Upsample backcast to original length (simple repeat)
        if self.pooled_len < self.input_len:
            repeat_factor = self.input_len // self.pooled_len
            remainder = self.input_len % self.pooled_len
            backcast = backcast.repeat_interleave(repeat_factor, dim=2)  # (batch, n_features, pooled_len*factor)
            if remainder > 0:
                pad = backcast[:, :, -1:].repeat(1, 1, remainder)  # (batch, n_features, remainder)
                backcast = torch.cat([backcast, pad], dim=2)  # (batch, n_features, input_len)
            backcast = backcast[:, :, :self.input_len]  # (batch, n_features, input_len)
        
        backcast = backcast.permute(0, 2, 1)  # (batch, input_len, n_features)
        forecast = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Return forecast and residual for next block
        residual = x.permute(0, 2, 1) - backcast  # (batch, input_len, n_features)
        
        return forecast, residual

