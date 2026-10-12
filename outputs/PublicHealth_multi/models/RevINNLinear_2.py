import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class RevINNLinear(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_ratio': [1, 2, 4],
            'use_bias': [True, False],
            'skip_weight': [0.0, 0.5, 1.0]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_ratio = self.config.get('hidden_ratio', 2)
        self.use_bias = self.config.get('use_bias', True)
        self.skip_weight = self.config.get('skip_weight', 0.5)
        
        # NLinear decomposition: subtract last value, predict residual
        self.linear = nn.Linear(input_len, output_len, bias=self.use_bias)
        
        # Optional hidden expansion for channel mixing
        hidden_dim = self.n_features * self.hidden_ratio
        self.channel_mix = nn.Sequential(
            nn.Linear(self.n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_features)
        ) if self.hidden_ratio > 1 else nn.Identity()
        
        # Learnable skip weight
        self.skip_alpha = nn.Parameter(torch.tensor(self.skip_weight))
        
        # RevIN for distribution normalization
        self.revin = RevIN(self.n_features, affine=True)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # NLinear decomposition: subtract last timestep value
        last = x[:, -1:, :]  # (batch, 1, n_features)
        x_decomp = x - last  # (batch, input_len, n_features)
        
        # Linear mapping on decomposed input
        x_decomp = x_decomp.permute(0, 2, 1)  # (batch, n_features, input_len)
        out = self.linear(x_decomp)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Add back the last value (trend continuation)
        last_expanded = last.expand(-1, self.output_len, -1)  # (batch, output_len, n_features)
        out = out + last_expanded  # (batch, output_len, n_features)
        
        # Optional channel mixing
        out = self.channel_mix(out)  # (batch, output_len, n_features)
        
        # Learnable skip: blend with naive last-value repeat
        skip = last_expanded  # (batch, output_len, n_features)
        alpha = torch.sigmoid(self.skip_alpha)  # scalar
        out = alpha * out + (1 - alpha) * skip  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
