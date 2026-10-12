import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelIndependentMixer_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'expansion_factor': [2, 4, 8],
            'n_blocks': [2, 4, 8],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.expansion_factor = self.config.get('expansion_factor', 4)
        self.n_blocks = self.config.get('n_blocks', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding: treat each time point as a token with n_features channels
        # We'll process channel-independently: each channel gets its own MLP
        self.input_embed = nn.Linear(self.input_len, self.hidden_dim)
        
        # MLP-Mixer blocks
        self.mixer_blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.mixer_blocks.append(nn.ModuleDict({
                # Temporal mixing: MLP across time dimension (now hidden_dim)
                'temporal_norm': nn.LayerNorm(self.hidden_dim),
                'temporal_mlp': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * self.expansion_factor),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim * self.expansion_factor, self.hidden_dim),
                    nn.Dropout(self.dropout),
                ),
                # Channel mixing: MLP across channels
                'channel_norm': nn.LayerNorm(self.n_features),
                'channel_mlp': nn.Sequential(
                    nn.Linear(self.n_features, self.n_features * self.expansion_factor),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.n_features * self.expansion_factor, self.n_features),
                    nn.Dropout(self.dropout),
                ),
            }))
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose to (batch, n_features, input_len) for channel-independent processing
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Embed each channel's time series to hidden dimension
        # (batch, n_features, input_len) -> (batch, n_features, hidden_dim)
        h = self.input_embed(x)  # (batch, n_features, hidden_dim)
        
        # Apply mixer blocks
        for block in self.mixer_blocks:
            # Temporal mixing: apply MLP to each channel's hidden representation
            # h: (batch, n_features, hidden_dim)
            h_res = h
            h = block['temporal_norm'](h)  # (batch, n_features, hidden_dim)
            h = block['temporal_mlp'](h)  # (batch, n_features, hidden_dim)
            h = h + h_res  # (batch, n_features, hidden_dim)
            
            # Channel mixing: apply MLP across channels at each hidden position
            # Transpose: (batch, hidden_dim, n_features)
            h = h.permute(0, 2, 1)  # (batch, hidden_dim, n_features)
            h_res = h
            h = block['channel_norm'](h)  # (batch, hidden_dim, n_features)
            h = block['channel_mlp'](h)  # (batch, hidden_dim, n_features)
            h = h + h_res  # (batch, hidden_dim, n_features)
            
            # Transpose back
            h = h.permute(0, 2, 1)  # (batch, n_features, hidden_dim)
        
        # Project to output length: (batch, n_features, hidden_dim) -> (batch, n_features, output_len)
        out = self.output_proj(h)  # (batch, n_features, output_len)
        
        # Transpose back to (batch, output_len, n_features)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

