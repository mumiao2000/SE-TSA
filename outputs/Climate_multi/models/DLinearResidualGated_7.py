import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DLinearResidualGated(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'gate_dim': [16, 32, 64],
            'dropout': [0.0, 0.1, 0.3],
            'decomp_kernel': [13, 25, 51]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config parameters with defaults
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.gate_dim = self.config.get('gate_dim', 32)
        self.dropout = self.config.get('dropout', 0.1)
        self.decomp_kernel = self.config.get('decomp_kernel', 25)
        
        # Ensure odd kernel size for symmetric moving average
        if self.decomp_kernel % 2 == 0:
            self.decomp_kernel += 1
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Decomposition: moving average for trend, residual for seasonality
        self.moving_avg = nn.AvgPool1d(kernel_size=self.decomp_kernel, stride=1, padding=self.decomp_kernel // 2)
        
        # Learnable decomposition weights (per channel)
        self.decomp_weight = nn.Parameter(torch.ones(self.n_features) * 0.5)  # (n_features,)
        
        # Trend branch: linear mapping from input_len to output_len
        self.trend_linear = nn.Linear(self.input_len, self.output_len)  # (input_len, output_len)
        
        # Seasonal branch: deeper with hidden layer
        self.seasonal_fc1 = nn.Linear(self.input_len, self.hidden_dim)  # (input_len, hidden_dim)
        self.seasonal_dropout = nn.Dropout(self.dropout)
        self.seasonal_fc2 = nn.Linear(self.hidden_dim, self.output_len)  # (hidden_dim, output_len)
        
        # Channel-adaptive gating for trend branch
        self.trend_gate_down = nn.Linear(self.n_features, self.gate_dim)  # (n_features, gate_dim)
        self.trend_gate_up = nn.Linear(self.gate_dim, self.n_features)  # (gate_dim, n_features)
        
        # Channel-adaptive gating for seasonal branch
        self.seasonal_gate_down = nn.Linear(self.n_features, self.gate_dim)  # (n_features, gate_dim)
        self.seasonal_gate_up = nn.Linear(self.gate_dim, self.n_features)  # (gate_dim, n_features)
        
        # Fusion gate: learnable combination of trend and seasonal per channel
        self.fusion_gate_down = nn.Linear(self.n_features * 2, self.gate_dim)  # (2*n_features, gate_dim)
        self.fusion_gate_up = nn.Linear(self.gate_dim, self.n_features)  # (gate_dim, n_features)
        
        # Residual projection for skip connection
        self.residual_linear = nn.Linear(self.input_len, self.output_len)  # (input_len, output_len)
        
        # Layer norms for stability
        self.trend_norm = nn.LayerNorm(self.n_features)
        self.seasonal_norm = nn.LayerNorm(self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store residual for skip connection
        residual = x  # (batch, input_len, n_features)
        
        # Decomposition: trend via moving average
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        trend = self.moving_avg(x_perm)  # (batch, n_features, input_len)
        trend = trend.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Seasonal = original - trend, with learnable weight
        decomp_w = torch.sigmoid(self.decomp_weight)  # (n_features,)
        seasonal = x - decomp_w.view(1, 1, -1) * trend  # (batch, input_len, n_features)
        
        # Trend branch
        trend_out = self.trend_linear(trend.permute(0, 2, 1))  # (batch, n_features, output_len)
        trend_out = trend_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Channel-adaptive gating for trend
        trend_gate = torch.sigmoid(self.trend_gate_up(torch.relu(self.trend_gate_down(trend_out))))  # (batch, output_len, n_features)
        trend_out = trend_out * trend_gate  # (batch, output_len, n_features)
        trend_out = self.trend_norm(trend_out)  # (batch, output_len, n_features)
        
        # Seasonal branch
        seasonal_perm = seasonal.permute(0, 2, 1)  # (batch, n_features, input_len)
        seasonal_out = torch.relu(self.seasonal_fc1(seasonal_perm))  # (batch, n_features, hidden_dim)
        seasonal_out = self.seasonal_dropout(seasonal_out)  # (batch, n_features, hidden_dim)
        seasonal_out = self.seasonal_fc2(seasonal_out)  # (batch, n_features, output_len)
        seasonal_out = seasonal_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Channel-adaptive gating for seasonal
        seasonal_gate = torch.sigmoid(self.seasonal_gate_up(torch.relu(self.seasonal_gate_down(seasonal_out))))  # (batch, output_len, n_features)
        seasonal_out = seasonal_out * seasonal_gate  # (batch, output_len, n_features)
        seasonal_out = self.seasonal_norm(seasonal_out)  # (batch, output_len, n_features)
        
        # Fusion with learnable gating
        combined = torch.cat([trend_out, seasonal_out], dim=-1)  # (batch, output_len, 2*n_features)
        fusion_gate = torch.sigmoid(self.fusion_gate_up(torch.relu(self.fusion_gate_down(combined))))  # (batch, output_len, n_features)
        
        # Weighted fusion
        out = fusion_gate * trend_out + (1 - fusion_gate) * seasonal_out  # (batch, output_len, n_features)
        
        # Residual connection
        residual_out = self.residual_linear(residual.permute(0, 2, 1))  # (batch, n_features, output_len)
        residual_out = residual_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        out = out + residual_out  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
