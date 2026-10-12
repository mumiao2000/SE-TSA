import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class RevINMLPMixer(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'expansion_factor': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.expansion_factor = self.config.get('expansion_factor', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Project to hidden dimension
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # MLP-Mixer blocks
        self.mixer_blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.mixer_blocks.append(
                MLPMixerBlock(
                    seq_len=self.input_len,
                    hidden_dim=self.hidden_dim,
                    expansion_factor=self.expansion_factor,
                    dropout=self.dropout
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        
        # Temporal decoder: project from input_len to output_len
        self.temporal_decoder = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden dimension
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Apply mixer blocks
        for block in self.mixer_blocks:
            x = block(x)  # (batch, input_len, hidden_dim)
        
        # Project back to feature dimension
        x = self.output_proj(x)  # (batch, input_len, n_features)
        
        # Temporal decoding: mix across time
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.temporal_decoder(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class MLPMixerBlock(nn.Module):
    def __init__(self, seq_len, hidden_dim, expansion_factor, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Token mixing (across time)
        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * expansion_factor, seq_len),
            nn.Dropout(dropout)
        )
        
        # Channel mixing (across features)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        # Token mixing
        residual = x
        x = self.norm1(x)  # (batch, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
        x = self.token_mlp(x)  # (batch, hidden_dim, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)
        x = x + residual  # (batch, seq_len, hidden_dim)
        
        # Channel mixing
        residual = x
        x = self.norm2(x)  # (batch, seq_len, hidden_dim)
        x = self.channel_mlp(x)  # (batch, seq_len, hidden_dim)
        x = x + residual  # (batch, seq_len, hidden_dim)
        
        return x
