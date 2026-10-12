import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class PatchTSTChannelIndependent_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.1, 0.3],
            'ffn_ratio': [1, 2, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.patch_size = 2
        self.stride = 2
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout = self.config.get('dropout', 0.1)
        self.ffn_ratio = self.config.get('ffn_ratio', 2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Calculate number of patches
        self.n_patches = (input_len - self.patch_size) // self.stride + 1
        
        # Patch embedding: per-channel
        self.patch_embed = nn.Linear(self.patch_size, self.d_model)  # Shared across channels
        
        # Single transformer encoder layer
        self.attn = nn.MultiheadAttention(self.d_model, self.n_heads, dropout=self.dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * self.ffn_ratio),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * self.ffn_ratio, self.d_model),
            nn.Dropout(self.dropout)
        )
        self.norm2 = nn.LayerNorm(self.d_model)
        
        # Flatten and project to output
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),  # Flatten patches: (batch * n_features, n_patches * d_model)
            nn.Linear(self.d_model * self.n_patches, output_len)
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, self.d_model) * 0.02)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Channel-independent: process each feature separately
        # Reshape to (batch * n_features, input_len, 1)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = x.reshape(batch_size * self.n_features, self.input_len, 1)  # (batch * n_features, input_len, 1)
        
        # Create patches
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            patch = x[:, start:start + self.patch_size, :]  # (batch * n_features, patch_size, 1)
            patch = patch.squeeze(-1)  # (batch * n_features, patch_size)
            patches.append(patch)
        
        x_patched = torch.stack(patches, dim=1)  # (batch * n_features, n_patches, patch_size)
        
        # Embed patches
        x_embed = self.patch_embed(x_patched)  # (batch * n_features, n_patches, d_model)
        x_embed = x_embed + self.pos_embed  # (batch * n_features, n_patches, d_model)
        
        # Single attention layer
        attn_out, _ = self.attn(x_embed, x_embed, x_embed)  # (batch * n_features, n_patches, d_model)
        x_embed = self.norm1(x_embed + attn_out)  # (batch * n_features, n_patches, d_model)
        
        # FFN
        ffn_out = self.ffn(x_embed)  # (batch * n_features, n_patches, d_model)
        x_embed = self.norm2(x_embed + ffn_out)  # (batch * n_features, n_patches, d_model)
        
        # Head: project to output_len
        out = self.head(x_embed)  # (batch * n_features, output_len)
        
        # Reshape back
        out = out.reshape(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

