import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'channel_hidden': [32, 128, 512],
            'cross_hidden': [64, 256, 1024],
            'n_blocks': [1, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.channel_hidden = self.config.get('channel_hidden', 128)
        self.cross_hidden = self.config.get('cross_hidden', 256)
        self.n_blocks = self.config.get('n_blocks', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Instance normalization per feature
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Channel-wise MLP: each feature gets its own network
        self.channel_mlps = nn.ModuleList()
        for _ in range(self.n_features):
            layers = []
            layers.append(nn.Linear(self.input_len, self.channel_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.channel_hidden, self.output_len))
            self.channel_mlps.append(nn.Sequential(*layers))
        
        # Cross-channel gating network
        self.gate_encoder = nn.Sequential(
            nn.Linear(self.n_features, self.cross_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.cross_hidden, self.n_features),
            nn.Sigmoid()
        )
        
        # Residual blocks for cross-channel mixing
        self.cross_blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.cross_blocks.append(nn.Sequential(
                nn.Linear(self.n_features, self.cross_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.cross_hidden, self.n_features)
            ))
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.n_features, self.cross_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.cross_hidden, self.n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        # Instance norm: normalize per feature across time
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_norm = self.norm(x_perm)  # (batch, n_features, input_len)
        x_norm = x_norm.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Channel-wise processing
        channel_outputs = []
        for i in range(self.n_features):
            feat = x_norm[:, :, i]  # (batch, input_len)
            out = self.channel_mlps[i](feat)  # (batch, output_len)
            channel_outputs.append(out)
        
        # Stack: (batch, output_len, n_features)
        channel_out = torch.stack(channel_outputs, dim=-1)  # (batch, output_len, n_features)
        
        # Cross-channel gating
        # Compute gate from mean of input features
        gate_input = x_norm.mean(dim=1)  # (batch, n_features)
        gate = self.gate_encoder(gate_input)  # (batch, n_features)
        gate = gate.unsqueeze(1)  # (batch, 1, n_features)
        
        # Apply gated cross-channel mixing with residual blocks
        out = channel_out  # (batch, output_len, n_features)
        for block in self.cross_blocks:
            residual = out  # (batch, output_len, n_features)
            out = block(out)  # (batch, output_len, n_features)
            out = out * gate + residual  # (batch, output_len, n_features)
        
        # Final projection
        out = self.output_proj(out)  # (batch, output_len, n_features)
        
        # Residual from channel outputs
        out = out + channel_out  # (batch, output_len, n_features)
        
        return out
