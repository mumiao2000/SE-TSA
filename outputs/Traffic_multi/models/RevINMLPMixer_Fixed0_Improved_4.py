import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class RevINMLPMixer_Fixed0_Improved(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'expansion_factor': [2, 4, 8],
            'n_blocks': [2, 4, 8],
            'dropout': [0.0, 0.2],
            'patch_size': [4, 8, 16],
            'gate_dim': [32, 128, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.expansion_factor = self.config.get('expansion_factor', 4)
        self.n_blocks = self.config.get('n_blocks', 4)
        self.dropout = self.config.get('dropout', 0.2)
        self.patch_size = self.config.get('patch_size', 8)
        self.gate_dim = self.config.get('gate_dim', 128)
        
        # Ensure patch size divides input_len
        self.patch_size = min(self.patch_size, self.input_len)
        while self.input_len % self.patch_size != 0 and self.patch_size > 1:
            self.patch_size -= 1
        
        self.n_patches = self.input_len // self.patch_size
        self.patch_dim = self.patch_size * self.n_features
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding: flatten each patch across all features
        self.patch_embed = nn.Linear(self.patch_dim, self.hidden_dim)
        
        # MLP-Mixer blocks with dual mixing
        self.blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.blocks.append(nn.ModuleDict({
                'temporal_norm': nn.LayerNorm(self.hidden_dim),
                'temporal_mlp': nn.Sequential(
                    nn.Linear(self.n_patches, self.n_patches * self.expansion_factor),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.n_patches * self.expansion_factor, self.n_patches),
                    nn.Dropout(self.dropout)
                ),
                'channel_norm': nn.LayerNorm(self.hidden_dim),
                'channel_mlp': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * self.expansion_factor),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim * self.expansion_factor, self.hidden_dim),
                    nn.Dropout(self.dropout)
                ),
                'gate_norm': nn.LayerNorm(self.hidden_dim),
                'gate_mlp': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.gate_dim),
                    nn.GELU(),
                    nn.Linear(self.gate_dim, 2),
                    nn.Sigmoid()
                )
            }))
        
        # Final norm
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        
        # Output head: map from patches to output sequence
        # Flatten patches and project to output_len * n_features
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim * self.n_patches, self.hidden_dim * self.n_patches // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * self.n_patches // 2, self.output_len * self.n_features)
        )
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch = x.shape[0]
        
        # RevIN normalize
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Patch: (batch, n_patches, patch_size, n_features) -> (batch, n_patches, patch_dim)
        x = x.reshape(batch, self.n_patches, self.patch_size, self.n_features)
        x = x.reshape(batch, self.n_patches, self.patch_dim)  # (batch, n_patches, patch_dim)
        
        # Patch embedding: (batch, n_patches, hidden_dim)
        x = self.patch_embed(x)  # (batch, n_patches, hidden_dim)
        
        # Transpose for temporal mixing: (batch, hidden_dim, n_patches)
        # Keep original for channel mixing: (batch, n_patches, hidden_dim)
        
        for block in self.blocks:
            residual = x  # (batch, n_patches, hidden_dim)
            
            # Temporal mixing: operate on patch dimension
            h_temp = block['temporal_norm'](x)  # (batch, n_patches, hidden_dim)
            h_temp = h_temp.permute(0, 2, 1)  # (batch, hidden_dim, n_patches)
            h_temp = block['temporal_mlp'](h_temp)  # (batch, hidden_dim, n_patches)
            h_temp = h_temp.permute(0, 2, 1)  # (batch, n_patches, hidden_dim)
            
            # Channel mixing: operate on hidden dimension
            h_chan = block['channel_norm'](x)  # (batch, n_patches, hidden_dim)
            h_chan = block['channel_mlp'](h_chan)  # (batch, n_patches, hidden_dim)
            
            # Gating: learn to combine temporal and channel
            gate_input = block['gate_norm'](x)  # (batch, n_patches, hidden_dim)
            gates = block['gate_mlp'](gate_input)  # (batch, n_patches, 2)
            gate_temp = gates[:, :, 0:1]  # (batch, n_patches, 1)
            gate_chan = gates[:, :, 1:2]  # (batch, n_patches, 1)
            
            # Combine with gates
            x = residual + gate_temp * h_temp + gate_chan * h_chan  # (batch, n_patches, hidden_dim)
        
        # Final norm
        x = self.final_norm(x)  # (batch, n_patches, hidden_dim)
        
        # Flatten and project to output
        x = x.reshape(batch, self.n_patches * self.hidden_dim)  # (batch, n_patches * hidden_dim)
        x = self.head(x)  # (batch, output_len * n_features)
        x = x.reshape(batch, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # RevIN denormalize
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x
