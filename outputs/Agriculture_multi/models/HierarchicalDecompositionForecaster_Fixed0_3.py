import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HierarchicalDecompositionForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'trend_kernel': [3, 7, 21],
            'n_harmonics': [4, 16, 64],
            'hidden_dim': [64, 256, 512],
            'n_levels': [2, 3, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.trend_kernel = self.config.get('trend_kernel', 7)
        self.n_harmonics = self.config.get('n_harmonics', 16)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_levels = self.config.get('n_levels', 3)
        
        # Learnable moving average for trend extraction
        self.trend_convs = nn.ModuleList()
        for level in range(self.n_levels):
            kernel = self.trend_kernel * (2 ** level)
            # Ensure kernel is odd
            if kernel % 2 == 0:
                kernel += 1
            padding = kernel // 2
            self.trend_convs.append(
                nn.Conv1d(self.n_features, self.n_features, kernel, 
                         padding=padding, groups=self.n_features, bias=False)
            )
        
        # Fourier features for seasonality
        self.frequencies = nn.Parameter(
            torch.randn(self.n_harmonics, self.n_features) * 0.1
        )
        self.phase_shifts = nn.Parameter(torch.zeros(self.n_harmonics, self.n_features))
        
        # Seasonal amplitude prediction
        self.seasonal_amplitude = nn.Sequential(
            nn.Linear(self.input_len * self.n_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_harmonics * self.n_features)
        )
        
        # Residual processor
        self.residual_encoder = nn.LSTM(
            self.n_features, self.hidden_dim, 
            num_layers=2, batch_first=True, bidirectional=True
        )
        self.residual_decoder = nn.LSTM(
            self.hidden_dim * 2, self.hidden_dim,
            num_layers=2, batch_first=True
        )
        self.residual_proj = nn.Linear(self.hidden_dim, self.n_features)
        
        # Trend predictor (simple extrapolation)
        self.trend_predictor = nn.Sequential(
            nn.Linear(self.input_len, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_len)
        )
        
        # Combination weights
        self.combine = nn.Sequential(
            nn.Linear(self.n_features * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),  # weights for trend, seasonal, residual
            nn.Softmax(dim=-1)
        )
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Normalize
        x_perm = x.permute(0, 2, 1).contiguous()  # (batch, n_features, input_len)
        x_norm = self.norm(x_perm)  # (batch, n_features, input_len)
        x_norm = x_norm.permute(0, 2, 1).contiguous()  # (batch, input_len, n_features)
        
        # Store stats for denorm
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        x_std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        
        # Multi-scale trend extraction
        trends = []
        x_conv = x_perm  # (batch, n_features, input_len)
        for conv in self.trend_convs:
            t = conv(x_conv)  # (batch, n_features, input_len)
            trends.append(t)
            # Remove extracted trend for next level
            x_conv = x_conv - t
        
        # Average trends across levels
        trend = sum(trends) / len(trends)  # (batch, n_features, input_len)
        trend = trend.permute(0, 2, 1).contiguous()  # (batch, input_len, n_features)
        
        # Predict trend
        trend_pred = []
        for i in range(self.n_features):
            ti = trend[:, :, i]  # (batch, input_len)
            pi = self.trend_predictor(ti)  # (batch, output_len)
            trend_pred.append(pi)
        trend_pred = torch.stack(trend_pred, dim=-1)  # (batch, output_len, n_features)
        
        # Seasonal component using Fourier series
        time_input = torch.arange(self.input_len, device=x.device).float()
        time_output = torch.arange(self.output_len, device=x.device).float() + self.input_len
        
        # Amplitudes from data
        flat_input = x_norm.reshape(batch_size, -1)  # (batch, input_len * n_features)
        amplitudes = self.seasonal_amplitude(flat_input)  # (batch, n_harmonics * n_features)
        amplitudes = amplitudes.reshape(batch_size, self.n_harmonics, self.n_features)  # (batch, n_harmonics, n_features)
        
        # Generate seasonal predictions
        seasonal_input = torch.zeros(batch_size, self.input_len, self.n_features, device=x.device)
        seasonal_output = torch.zeros(batch_size, self.output_len, self.n_features, device=x.device)
        
        for h in range(self.n_harmonics):
            freq = self.frequencies[h]  # (n_features,)
            phase = self.phase_shifts[h]  # (n_features,)
            amp = amplitudes[:, h, :]  # (batch, n_features)
            
            # Input reconstruction
            sin_input = torch.sin(2 * 3.14159 * freq * time_input.unsqueeze(-1) + phase)  # (input_len, n_features)
            seasonal_input = seasonal_input + amp.unsqueeze(1) * sin_input.unsqueeze(0)  # (batch, input_len, n_features)
            
            # Output prediction
            sin_output = torch.sin(2 * 3.14159 * freq * time_output.unsqueeze(-1) + phase)  # (output_len, n_features)
            seasonal_output = seasonal_output + amp.unsqueeze(1) * sin_output.unsqueeze(0)  # (batch, output_len, n_features)
        
        # Residual component
        residual = x_norm - seasonal_input  # (batch, input_len, n_features)
        
        # Encode residual with BiLSTM
        res_enc, _ = self.residual_encoder(residual)  # (batch, input_len, hidden_dim*2)
        
        # Decode to future
        # Use last hidden state to initialize decoder
        last_hidden = res_enc[:, -1:, :].repeat(1, self.output_len, 1)  # (batch, output_len, hidden_dim*2)
        res_dec, _ = self.residual_decoder(last_hidden)  # (batch, output_len, hidden_dim)
        residual_pred = self.residual_proj(res_dec)  # (batch, output_len, n_features)
        
        # Combine predictions with learned weights
        combined_input = torch.cat([
            trend_pred, seasonal_output, residual_pred
        ], dim=-1)  # (batch, output_len, n_features*3)
        
        weights = self.combine(combined_input)  # (batch, output_len, 3)
        
        out = (weights[:, :, 0:1] * trend_pred + 
               weights[:, :, 1:2] * seasonal_output + 
               weights[:, :, 2:3] * residual_pred)  # (batch, output_len, n_features)
        
        # Denormalize
        out = out * x_std + x_mean  # (batch, output_len, n_features)
        
        return out

