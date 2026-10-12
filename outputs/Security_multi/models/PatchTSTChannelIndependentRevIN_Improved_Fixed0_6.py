import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class PatchTSTChannelIndependentRevIN_Improved_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [32, 64, 128],
            'n_heads': [1, 2, 4],
            'dropout': [0.0, 0.1, 0.3],
            'ffn_ratio': [1, 2, 4],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Hyperparameters with wide ranges
        self.d_model = self.config.get('d_model', 64)
        self.n_heads = self.config.get('n_heads', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.ffn_ratio = self.config.get('ffn_ratio', 2)
        
        # Fixed small patch size based on validated principle
        self.patch_size = 2
        self.stride = 2
        
        # Calculate number of patches
        self.n_patches = (input_len - self.patch_size) // self.stride + 1
        
        # Ensure d_model is divisible by n_heads
        assert self.d_model % self.n_heads == 0, f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding: linear projection of each patch to d_model
        # Channel-independent: each feature has its own embedding
        self.patch_embedding = nn.Linear(self.patch_size, self.d_model)
        
        # Positional encoding for patches - fixed shape to be 3D for proper broadcasting
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, self.d_model) * 0.02)
        
        # Single transformer layer (minimal depth)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * self.ffn_ratio),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * self.ffn_ratio, self.d_model),
            nn.Dropout(self.dropout)
        )
        
        # Output projection: from patch representations to forecast
        # Flatten patches and project to output_len
        self.head = nn.Linear(self.n_patches * self.d_model, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for channel-independent processing: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Create patches: (batch, n_features, n_patches, patch_size)
        patches = x.unfold(dimension=2, size=self.patch_size, step=self.stride)  # (batch, n_features, n_patches, patch_size)
        
        # Patch embedding: apply same linear to each channel independently
        # Reshape for linear: (batch * n_features, n_patches, patch_size)
        patches_reshaped = patches.reshape(batch_size * self.n_features, self.n_patches, self.patch_size)  # (batch * n_features, n_patches, patch_size)
        
        # Embed patches
        embedded = self.patch_embedding(patches_reshaped)  # (batch * n_features, n_patches, d_model)
        
        # Add positional encoding - now both are 3D and broadcast correctly
        embedded = embedded + self.pos_encoding  # (batch * n_features, n_patches, d_model)
        
        # Transformer layer with pre-norm
        # Attention
        embedded_norm = self.norm1(embedded)  # (batch * n_features, n_patches, d_model)
        attn_out, _ = self.attn(embedded_norm, embedded_norm, embedded_norm, need_weights=False)  # (batch * n_features, n_patches, d_model)
        embedded = embedded + attn_out  # (batch * n_features, n_patches, d_model)
        
        # FFN
        embedded_norm = self.norm2(embedded)  # (batch * n_features, n_patches, d_model)
        ffn_out = self.ffn(embedded_norm)  # (batch * n_features, n_patches, d_model)
        embedded = embedded + ffn_out  # (batch * n_features, n_patches, d_model)
        
        # Flatten patches and project to output
        embedded_flat = embedded.reshape(batch_size * self.n_features, self.n_patches * self.d_model)  # (batch * n_features, n_patches * d_model)
        
        # Output projection
        out = self.head(embedded_flat)  # (batch * n_features, output_len)
        
        # Reshape back: (batch, n_features, output_len)
        out = out.reshape(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        
        # Transpose to (batch, output_len, n_features)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out
