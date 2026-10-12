import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NLinearChannelMixRevIN(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'channel_mix_dim': [32, 128, 512],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.channel_mix_dim = self.config.get('channel_mix_dim', 128)
        self.dropout_p = self.config.get('dropout', 0.1)
        
        # Temporal linear: predict future from past (per-channel)
        self.temporal_linear = nn.Linear(input_len, output_len)
        
        # Channel mixing MLP
        self.channel_mixer = nn.Sequential(
            nn.Linear(n_features, self.channel_mix_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.channel_mix_dim, n_features)
        )
        
        self.revin = RevIN(n_features, affine=True)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # NLinear: subtract last value as trend
        seq_last = x[:, -1:, :].detach()  # (batch, 1, n_features)
        x = x - seq_last  # (batch, input_len, n_features)
        
        # Temporal projection
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.temporal_linear(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Channel mixing
        x = self.channel_mixer(x)  # (batch, output_len, n_features)
        x = self.dropout(x)  # (batch, output_len, n_features)
        
        # Add back trend
        x = x + seq_last  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)

