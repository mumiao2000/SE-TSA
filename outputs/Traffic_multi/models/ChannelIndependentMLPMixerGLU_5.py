import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelIndependentMLPMixerGLU(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'expansion_factor': [4, 8, 16],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.2, 0.4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.expansion_factor = self.config.get('expansion_factor', 8)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout_rate = self.config.get('dropout', 0.2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Per-channel projection to hidden dim
        self.input_proj = nn.ModuleList([
            nn.Linear(self.input_len, self.hidden_dim) for _ in range(self.n_features)
        ])  # (batch, hidden_dim) per channel
        
        # MLP-Mixer blocks with GELU-GLU gating
        self.mixer_blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            # Temporal mixing: mix across time dimension (hidden_dim represents time tokens)
            self.mixer_blocks.append(nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim * self.expansion_factor),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim * self.expansion_factor, self.hidden_dim),
                nn.Dropout(self.dropout_rate)
            ))
            # Feature mixing within each token position
            self.mixer_blocks.append(nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim * self.expansion_factor),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim * self.expansion_factor, self.hidden_dim),
                nn.Dropout(self.dropout_rate)
            ))
        
        # Output projection per channel
        self.output_proj = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.output_len) for _ in range(self.n_features)
        ])  # (batch, output_len) per channel
        
        self.norm = nn.LayerNorm(self.hidden_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Process each channel independently
        outputs = []
        for i in range(self.n_features):
            # Extract channel: (batch, input_len)
            xi = x[:, :, i]  # (batch, input_len)
            
            # Project to hidden dimension: (batch, hidden_dim)
            h = self.input_proj[i](xi)  # (batch, hidden_dim)
            
            # Apply MLP-Mixer blocks
            for j, block in enumerate(self.mixer_blocks):
                if j % 2 == 0:
                    # Temporal mixing with residual
                    h = h + block(h)  # (batch, hidden_dim)
                else:
                    # Feature mixing with residual (here it's within-token mixing)
                    h = h + block(h)  # (batch, hidden_dim)
            
            h = self.norm(h)  # (batch, hidden_dim)
            
            # Project to output length
            out = self.output_proj[i](h)  # (batch, output_len)
            outputs.append(out)
        
        # Stack channels: list of (batch, output_len) -> (batch, output_len, n_features)
        x = torch.stack(outputs, dim=-1)  # (batch, output_len, n_features)
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x
