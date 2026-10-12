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
            'temporal_hidden': [64, 256, 1024],
            'n_interaction_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.channel_hidden = self.config.get('channel_hidden', 128)
        self.temporal_hidden = self.config.get('temporal_hidden', 256)
        self.n_interaction_layers = self.config.get('n_interaction_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Per-channel instance normalization with learnable affine
        self.channel_norms = nn.ModuleList([
            nn.InstanceNorm1d(1, affine=True) for _ in range(self.n_features)
        ])
        
        # Channel interaction: compute attention weights between channels
        self.channel_query = nn.Linear(self.input_len, self.channel_hidden)
        self.channel_key = nn.Linear(self.input_len, self.channel_hidden)
        self.channel_value = nn.Linear(self.input_len, self.channel_hidden)
        
        # Cross-channel mixing layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(self.n_interaction_layers):
            self.interaction_layers.append(nn.Sequential(
                nn.Linear(self.n_features * self.channel_hidden, self.temporal_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.temporal_hidden, self.n_features * self.channel_hidden)
            ))
        
        # Channel-specific temporal MLPs
        self.temporal_mlps = nn.ModuleList()
        for _ in range(self.n_features):
            self.temporal_mlps.append(nn.Sequential(
                nn.Linear(self.input_len, self.temporal_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.temporal_hidden, self.temporal_hidden // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.temporal_hidden // 2, self.output_len)
            ))
        
        # Adaptive gating for each channel
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_len, self.channel_hidden),
                nn.Sigmoid()
            ) for _ in range(self.n_features)
        ])
        
        # Output scaling
        self.output_scale = nn.Parameter(torch.ones(self.n_features))
        self.output_shift = nn.Parameter(torch.zeros(self.n_features))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Per-channel normalization
        x_norm = []
        for i in range(self.n_features):
            xi = x[:, :, i:i+1]  # (batch, input_len, 1)
            xi = xi.permute(0, 2, 1)  # (batch, 1, input_len)
            xi = self.channel_norms[i](xi)  # (batch, 1, input_len)
            xi = xi.permute(0, 2, 1)  # (batch, input_len, 1)
            x_norm.append(xi)
        x_norm = torch.cat(x_norm, dim=-1)  # (batch, input_len, n_features)
        
        # Store original scale for denormalization
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        x_std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        
        # Channel interaction via attention
        # Reshape for attention: (batch, n_features, input_len)
        x_perm = x_norm.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        Q = self.channel_query(x_perm)  # (batch, n_features, channel_hidden)
        K = self.channel_key(x_perm)    # (batch, n_features, channel_hidden)
        V = self.channel_value(x_perm)  # (batch, n_features, channel_hidden)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.channel_hidden ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, n_features, n_features)
        
        # Apply attention
        channel_repr = torch.matmul(attn_weights, V)  # (batch, n_features, channel_hidden)
        
        # Flatten and apply interaction MLPs
        channel_flat = channel_repr.view(batch_size, -1)  # (batch, n_features * channel_hidden)
        
        for layer in self.interaction_layers:
            residual = channel_flat
            channel_flat = channel_flat + layer(channel_flat)  # (batch, n_features * channel_hidden)
        
        # Reshape back
        channel_enhanced = channel_flat.view(batch_size, self.n_features, self.channel_hidden)
        
        # Channel-specific temporal prediction with gating
        outputs = []
        for i in range(self.n_features):
            xi = x_perm[:, i, :]  # (batch, input_len)
            gate = self.gates[i](xi)  # (batch, channel_hidden)
            
            # Temporal prediction
            pred = self.temporal_mlps[i](xi)  # (batch, output_len)
            outputs.append(pred)
        
        out = torch.stack(outputs, dim=-1)  # (batch, output_len, n_features)
        
        # Denormalize using original statistics
        out = out * x_std + x_mean  # (batch, output_len, n_features)
        
        # Apply learned output scaling
        out = out * self.output_scale + self.output_shift  # (batch, output_len, n_features)
        
        return out
