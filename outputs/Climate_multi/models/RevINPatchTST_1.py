import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class RevINPatchTST(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'patch_size': [2, 8, 24],
            'ff_dim_ratio': [2, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 512)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.patch_size = self.config.get('patch_size', 8)
        self.ff_dim = self.d_model * self.config.get('ff_dim_ratio', 4)
        
        # Ensure patch_size divides input_len reasonably
        self.patch_size = min(self.patch_size, input_len // 2)
        self.patch_size = max(self.patch_size, 1)
        self.n_patches = input_len // self.patch_size
        self.patch_stride = self.patch_size  # non-overlapping patches
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding: (batch, n_features, input_len) -> (batch, n_features, n_patches, d_model)
        self.patch_embed = nn.Linear(self.patch_size, self.d_model)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.n_patches, self.d_model) * 0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Prediction head: predict all output patches at once
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, output_len)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-independent: process each feature separately
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Create patches: (batch, n_features, n_patches, patch_size)
        batch_size, n_feat, seq_len = x.shape
        x = x.unfold(dimension=2, size=self.patch_size, step=self.patch_stride)  # (batch, n_features, n_patches, patch_size)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, n_features, n_patches, d_model)
        x = x + self.pos_embed  # (batch, n_features, n_patches, d_model)
        
        # Reshape for transformer: (batch * n_features, n_patches, d_model)
        x = x.reshape(batch_size * n_feat, self.n_patches, self.d_model)  # (batch * n_features, n_patches, d_model)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch * n_features, n_patches, d_model)
        
        # Global average pooling over patches
        x = x.mean(dim=1)  # (batch * n_features, d_model)
        
        # Prediction head
        x = self.head(x)  # (batch * n_features, output_len)
        
        # Reshape back
        x = x.reshape(batch_size, n_feat, self.output_len)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x

