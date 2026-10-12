import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NLinearChannelMix(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'mix_dim': [32, 128, 512],
            'dropout': [0.0, 0.2]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.mix_dim = self.config.get('mix_dim', 128)
        self.dropout = self.config.get('dropout', 0.0)
        
        # Per-channel linear: maps input_len -> output_len for each feature
        self.channel_linears = nn.ModuleList([
            nn.Linear(self.input_len, self.output_len) for _ in range(self.n_features)
        ])
        
        # Channel mixing bottleneck: n_features -> mix_dim -> n_features
        self.mix_down = nn.Linear(self.n_features, self.mix_dim)
        self.mix_up = nn.Linear(self.mix_dim, self.n_features)
        self.mix_act = nn.GELU()
        self.mix_dropout = nn.Dropout(self.dropout)
        
        # Learnable residual weight: how much to trust channel-mixed vs channel-independent
        self.residual_gate = nn.Parameter(torch.zeros(1))
        
        self.revin = RevIN(self.n_features, affine=True)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-independent predictions
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        channel_outs = []
        for i, lin in enumerate(self.channel_linears):
            ci = lin(x_perm[:, i, :])  # (batch, output_len)
            channel_outs.append(ci)
        channel_pred = torch.stack(channel_outs, dim=-1)  # (batch, output_len, n_features)
        
        # Channel mixing on predictions
        mixed = self.mix_down(channel_pred)  # (batch, output_len, mix_dim)
        mixed = self.mix_act(mixed)  # (batch, output_len, mix_dim)
        mixed = self.mix_dropout(mixed)  # (batch, output_len, mix_dim)
        mixed = self.mix_up(mixed)  # (batch, output_len, n_features)
        
        # Adaptive blending with sigmoid-gated residual
        gate = torch.sigmoid(self.residual_gate)  # scalar
        out = (1 - gate) * channel_pred + gate * mixed  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

