import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class RevINMLPMixer_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'patch_size': [2, 8, 24],
            'hidden_dim': [64, 256, 1024],
            'expansion_factor': [2, 4, 8],
            'n_mixer_layers': [1, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.patch_size = self.config.get('patch_size', 8)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.expansion_factor = self.config.get('expansion_factor', 4)
        self.n_mixer_layers = self.config.get('n_mixer_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Ensure patch size divides input_len
        self.n_patches = self.input_len // self.patch_size
        assert self.n_patches * self.patch_size == self.input_len, f"input_len {self.input_len} must be divisible by patch_size {self.patch_size}"
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding: (batch, n_patches, patch_size * n_features) -> (batch, n_patches, hidden_dim)
        self.patch_embed = nn.Linear(self.patch_size * self.n_features, self.hidden_dim)
        
        # MLP-Mixer blocks
        self.mixer_blocks = nn.ModuleList()
        for _ in range(self.n_mixer_layers):
            self.mixer_blocks.append(
                MLPMixerBlock(self.hidden_dim, self.n_patches, self.expansion_factor, self.dropout)
            )
        
        # Head: flatten mixer output and project to forecast
        self.head = nn.Sequential(
            nn.Linear(self.n_patches * self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 4, self.output_len * self.n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Patchify: (batch, n_patches, patch_size * n_features)
        B = x.shape[0]
        x = x.reshape(B, self.n_patches, self.patch_size * self.n_features)  # (batch, n_patches, patch_size * n_features)
        
        # Embed patches
        x = self.patch_embed(x)  # (batch, n_patches, hidden_dim)
        
        # Apply mixer blocks
        for block in self.mixer_blocks:
            x = block(x)  # (batch, n_patches, hidden_dim)
        
        # Flatten and project to output
        x = x.reshape(B, self.n_patches * self.hidden_dim)  # (batch, n_patches * hidden_dim)
        x = self.head(x)  # (batch, output_len * n_features)
        x = x.reshape(B, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class MLPMixerBlock(nn.Module):
    def __init__(self, hidden_dim, n_patches, expansion_factor, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Temporal mixing: mix across patches (treat hidden_dim as channels)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(n_patches, n_patches * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_patches * expansion_factor, n_patches)
        )
        
        # Channel mixing: mix across hidden_dim (treat n_patches as channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim)
        )
    
    def forward(self, x):
        # x: (batch, n_patches, hidden_dim)
        # Temporal mixing
        residual = x
        x = self.norm1(x)  # (batch, n_patches, hidden_dim)
        x = x.transpose(1, 2)  # (batch, hidden_dim, n_patches)
        x = self.temporal_mlp(x)  # (batch, hidden_dim, n_patches)
        x = x.transpose(1, 2)  # (batch, n_patches, hidden_dim)
        x = x + residual
        
        # Channel mixing
        residual = x
        x = self.norm2(x)  # (batch, n_patches, hidden_dim)
        x = self.channel_mlp(x)  # (batch, n_patches, hidden_dim)
        x = x + residual
        
        return x
