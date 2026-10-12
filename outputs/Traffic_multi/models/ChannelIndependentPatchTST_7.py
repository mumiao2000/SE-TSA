import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelIndependentPatchTST(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'patch_size': [2, 8, 24],
            'ffn_ratio': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 256)
        self.n_heads = cfg.get('n_heads', 8)
        self.n_layers = cfg.get('n_layers', 4)
        self.dropout = cfg.get('dropout', 0.1)
        self.patch_size = cfg.get('patch_size', 8)
        self.ffn_ratio = cfg.get('ffn_ratio', 4)
        
        # Ensure patch_size divides input_len
        self.patch_size = min(self.patch_size, input_len)
        while input_len % self.patch_size != 0 and self.patch_size > 1:
            self.patch_size -= 1
        self.n_patches = input_len // self.patch_size
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding: linear projection of each patch
        self.patch_embed = nn.Linear(self.patch_size, self.d_model)  # (batch*n_features, n_patches, d_model)
        
        # Positional encoding for patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, self.d_model) * 0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * self.ffn_ratio,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Decoder: predict output patches then flatten
        self.output_patch_size = max(1, output_len // self.n_patches)
        self.n_output_patches = (output_len + self.output_patch_size - 1) // self.output_patch_size
        self.decoder = nn.Linear(self.d_model, self.output_patch_size)
        
        # Final projection to exact output_len
        self.final_proj = nn.Linear(self.n_output_patches * self.output_patch_size, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-independent: process each feature separately
        batch_size = x.shape[0]
        
        # Reshape to (batch * n_features, input_len)
        x = x.permute(0, 2, 1).reshape(batch_size * self.n_features, self.input_len)  # (batch*n_features, input_len)
        
        # Create patches: (batch*n_features, n_patches, patch_size)
        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)  # (batch*n_features, n_patches, patch_size)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch*n_features, n_patches, d_model)
        x = x + self.pos_embed  # (batch*n_features, n_patches, d_model)
        
        # Transformer encoding
        x = self.encoder(x)  # (batch*n_features, n_patches, d_model)
        
        # Decode to output patches
        x = self.decoder(x)  # (batch*n_features, n_patches, output_patch_size)
        
        # Flatten patches and project to exact output length
        x = x.reshape(batch_size * self.n_features, -1)  # (batch*n_features, n_patches * output_patch_size)
        x = self.final_proj(x)  # (batch*n_features, output_len)
        
        # Reshape back: (batch, n_features, output_len) -> (batch, output_len, n_features)
        x = x.reshape(batch_size, self.n_features, self.output_len).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x

