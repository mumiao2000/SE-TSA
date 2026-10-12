import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NLinearChannelMixing(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.3],
            'use_channel_mixing': [True, False]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.use_channel_mixing = self.config.get('use_channel_mixing', True)
        
        # NLinear normalization: subtract last value
        # Channel mixing
        if self.use_channel_mixing:
            self.channel_mix = nn.Linear(n_features, n_features)
        
        # Build MLP layers
        layers = []
        in_dim = input_len
        for i in range(self.n_layers):
            out_dim = self.hidden_dim if i < self.n_layers - 1 else output_len
            layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Feature-wise projection
        self.feature_proj = nn.ModuleList([
            nn.Linear(input_len, output_len) for _ in range(n_features)
        ])
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # NLinear normalization: subtract the last time step
        last = x[:, -1:, :]  # (batch, 1, n_features)
        x = x - last  # (batch, input_len, n_features)
        
        # Channel mixing
        if self.use_channel_mixing:
            x = self.channel_mix(x)  # (batch, input_len, n_features)
        
        # Feature-independent linear layers with residual from last value
        outputs = []
        for i in range(self.n_features):
            feat = x[:, :, i]  # (batch, input_len)
            out = self.feature_proj[i](feat)  # (batch, output_len)
            outputs.append(out)
        
        x = torch.stack(outputs, dim=-1)  # (batch, output_len, n_features)
        
        # Add back the last value (de-normalization)
        x = x + last  # (batch, output_len, n_features)
        
        return x
