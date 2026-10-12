import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class PatchTSTChannelMix_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'patch_size': [2, 8, 24],
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.2, 0.4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.patch_size = self.config.get('patch_size', 8)
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout_rate = self.config.get('dropout', 0.2)
        
        # Ensure patch size divides input length
        self.n_patches = self.input_len // self.patch_size
        self.effective_input_len = self.n_patches * self.patch_size
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding: per-channel patch embedding
        self.patch_embed = nn.Linear(self.patch_size, self.d_model)  # (batch*n_features, n_patches, d_model)
        
        # Positional encoding for patches
        self.pos_enc = nn.Parameter(torch.randn(1, self.n_patches, self.d_model) * 0.02)
        
        # Channel grouping: separate high-scale and low-scale features
        self.n_high_scale = min(5, self.n_features)
        self.n_low_scale = self.n_features - self.n_high_scale
        
        # Channel-specific transformers
        self.high_scale_transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=self.n_heads, 
                dim_feedforward=self.d_model * 4, dropout=self.dropout_rate,
                batch_first=True, activation='gelu'
            ) for _ in range(self.n_layers)
        ])
        
        self.low_scale_transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=self.n_heads,
                dim_feedforward=self.d_model * 4, dropout=self.dropout_rate,
                batch_first=True, activation='gelu'
            ) for _ in range(self.n_layers)
        ])
        
        # Cross-channel mixing after transformer layers
        self.channel_mixer = nn.Sequential(
            nn.Linear(self.d_model * self.n_features, self.d_model * 4),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model * 4, self.d_model)
        )
        
        # Output projection: from flattened patches to output time steps and features
        self.fc_out = nn.Linear(self.n_patches * self.d_model, self.output_len * self.n_features)
        
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Trim to effective input length
        if self.effective_input_len < self.input_len:
            x = x[:, :self.effective_input_len, :]  # (batch, effective_input_len, n_features)
        
        # Reshape to patches: (batch, n_patches, patch_size, n_features)
        x = x.reshape(x.size(0), self.n_patches, self.patch_size, self.n_features)
        # (batch, n_patches, patch_size, n_features)
        
        # Process each channel
        outputs = []
        for i in range(self.n_features):
            # Extract channel patches: (batch, n_patches, patch_size)
            xi = x[:, :, :, i]  # (batch, n_patches, patch_size)
            
            # Embed patches: (batch, n_patches, d_model)
            xi = self.patch_embed(xi)  # (batch, n_patches, d_model)
            xi = xi + self.pos_enc  # (batch, n_patches, d_model)
            xi = self.dropout(xi)  # (batch, n_patches, d_model)
            
            # Apply appropriate transformer
            if i < self.n_high_scale:
                for layer in self.high_scale_transformer:
                    xi = layer(xi)  # (batch, n_patches, d_model)
            else:
                for layer in self.low_scale_transformer:
                    xi = layer(xi)  # (batch, n_patches, d_model)
            
            outputs.append(xi)  # (batch, n_patches, d_model)
        
        # Stack channels: (batch, n_patches, n_features, d_model)
        h = torch.stack(outputs, dim=2)  # (batch, n_patches, n_features, d_model)
        
        # Cross-channel mixing per patch
        h = h.reshape(h.size(0), self.n_patches, -1)  # (batch, n_patches, n_features * d_model)
        h = self.channel_mixer(h)  # (batch, n_patches, d_model)
        
        # Flatten all patches and project to output
        h_flat = h.reshape(h.size(0), -1)  # (batch, n_patches * d_model)
        out = self.fc_out(h_flat)  # (batch, output_len * n_features)
        out = out.view(-1, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

