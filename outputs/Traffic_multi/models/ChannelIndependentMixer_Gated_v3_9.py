import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelIndependentMixer_Gated_v3(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_mixer_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'expansion_factor': [2, 4, 8],
            'use_residual_bypass': [True, False],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_mixer_layers = self.config.get('n_mixer_layers', 4)
        self.dropout = self.config.get('dropout', 0.3)
        self.expansion_factor = self.config.get('expansion_factor', 4)
        self.use_residual_bypass = self.config.get('use_residual_bypass', True)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: (batch, input_len, n_features) -> (batch, input_len, hidden_dim)
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)  # (batch, input_len, hidden_dim)
        
        # Mixer layers
        self.mixer_layers = nn.ModuleList()
        for _ in range(self.n_mixer_layers):
            self.mixer_layers.append(
                GatedMixerBlock(
                    seq_len=self.input_len,
                    hidden_dim=self.hidden_dim,
                    n_features=self.n_features,
                    expansion_factor=self.expansion_factor,
                    dropout=self.dropout,
                )
            )
        
        # Optional residual bypass for low-frequency preservation
        if self.use_residual_bypass:
            self.bypass_proj = nn.Sequential(
                nn.Linear(self.n_features, self.hidden_dim),  # (batch, input_len, hidden_dim)
                nn.Dropout(self.dropout),
            )
        
        # Output projection: (batch, input_len, hidden_dim) -> (batch, output_len, n_features)
        self.temporal_proj = nn.Linear(self.input_len, self.output_len)  # operates after transpose
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)  # (batch, output_len, n_features)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store residual for bypass
        x_residual = x  # (batch, input_len, n_features)
        
        # Input projection
        h = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Apply mixer layers
        for mixer_layer in self.mixer_layers:
            h = mixer_layer(h)  # (batch, input_len, hidden_dim)
        
        # Add bypass connection if enabled
        if self.use_residual_bypass:
            h_bypass = self.bypass_proj(x_residual)  # (batch, input_len, hidden_dim)
            h = h + h_bypass  # (batch, input_len, hidden_dim)
        
        # Temporal projection: (batch, hidden_dim, input_len) -> (batch, hidden_dim, output_len)
        h = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        h = self.temporal_proj(h)  # (batch, hidden_dim, output_len)
        h = h.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Feature projection
        out = self.output_proj(h)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class GatedMixerBlock(nn.Module):
    def __init__(self, seq_len, hidden_dim, n_features, expansion_factor, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.expansion_factor = expansion_factor
        
        # Temporal mixing: operates on (batch, hidden_dim, seq_len)
        self.temporal_norm = nn.LayerNorm(self.hidden_dim)  # normalize over hidden_dim
        
        temporal_inner = hidden_dim * expansion_factor
        self.temporal_fc1 = nn.Linear(seq_len, temporal_inner)  # (batch, hidden_dim, temporal_inner)
        self.temporal_gate = nn.Linear(seq_len, temporal_inner)  # (batch, hidden_dim, temporal_inner)
        self.temporal_fc2 = nn.Linear(temporal_inner, seq_len)  # (batch, hidden_dim, seq_len)
        self.temporal_dropout = nn.Dropout(dropout)
        
        # Channel mixing: operates on (batch, seq_len, hidden_dim)
        self.channel_norm = nn.LayerNorm(self.hidden_dim)  # normalize over hidden_dim (last dim)
        
        channel_inner = hidden_dim * expansion_factor
        self.channel_fc1 = nn.Linear(hidden_dim, channel_inner)  # (batch, seq_len, channel_inner)
        self.channel_gate = nn.Linear(hidden_dim, channel_inner)  # (batch, seq_len, channel_inner)
        self.channel_fc2 = nn.Linear(channel_inner, hidden_dim)  # (batch, seq_len, hidden_dim)
        self.channel_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        
        # Temporal mixing
        h = x.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
        h = self.temporal_norm(h.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, hidden_dim, seq_len) - norm over hidden_dim via permute trick
        
        # Actually, let's be cleaner: norm over the hidden_dim dimension
        h = h.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)
        h = self.temporal_norm(h)  # (batch, seq_len, hidden_dim) - norm over hidden_dim
        h = h.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
        
        # GELU-GLU gating for temporal
        h_gate = self.temporal_gate(h)  # (batch, hidden_dim, temporal_inner)
        h_val = self.temporal_fc1(h)  # (batch, hidden_dim, temporal_inner)
        h = F.gelu(h_gate) * h_val  # (batch, hidden_dim, temporal_inner)
        h = self.temporal_dropout(h)  # (batch, hidden_dim, temporal_inner)
        h = self.temporal_fc2(h)  # (batch, hidden_dim, seq_len)
        h = h.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)
        x = x + h  # (batch, seq_len, hidden_dim)
        
        # Channel mixing
        h = self.channel_norm(x)  # (batch, seq_len, hidden_dim) - norm over hidden_dim
        
        # GELU-GLU gating for channel
        h_gate = self.channel_gate(h)  # (batch, seq_len, channel_inner)
        h_val = self.channel_fc1(h)  # (batch, seq_len, channel_inner)
        h = F.gelu(h_gate) * h_val  # (batch, seq_len, channel_inner)
        h = self.channel_dropout(h)  # (batch, seq_len, channel_inner)
        h = self.channel_fc2(h)  # (batch, seq_len, hidden_dim)
        x = x + h  # (batch, seq_len, hidden_dim)
        
        return x  # (batch, seq_len, hidden_dim)
