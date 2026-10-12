import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class PatchTSTChannelIndependentRevIN_Lite_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 128, 256],
            'n_heads': [2, 4, 8],
            'dropout': [0.0, 0.2, 0.4],
            'drop_path': [0.0, 0.1, 0.2],
            'patch_size': [8, 16, 24],
            'ffn_ratio': [1, 2, 4],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with wide ranges
        self.d_model = self.config.get('d_model', 128)
        self.n_heads = self.config.get('n_heads', 4)
        self.dropout = self.config.get('dropout', 0.2)
        self.drop_path_prob = self.config.get('drop_path', 0.1)
        self.patch_size = self.config.get('patch_size', 16)
        self.ffn_ratio = self.config.get('ffn_ratio', 2)
        
        # Ensure patch_size divides input_len reasonably
        self.n_patches = max(1, input_len // self.patch_size)
        self.effective_input_len = self.n_patches * self.patch_size
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding: linear projection of flattened patches
        self.patch_embed = nn.Linear(self.patch_size, self.d_model)
        
        # Positional encoding (learnable, shared across channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Single transformer encoder layer (reduced from deep stack)
        self.attn = nn.MultiheadAttention(
            self.d_model, self.n_heads, dropout=self.dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        
        # FFN
        ffn_dim = self.d_model * self.ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(ffn_dim, self.d_model),
            nn.Dropout(self.dropout),
        )
        
        # Flatten and project to output
        self.flatten_dim = self.n_patches * self.d_model
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, output_len)
        )
        
    def apply_drop_path(self, x, drop_prob=0.0, training=False):
        # (batch, seq, d_model)
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0], 1, 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output  # (batch, seq, d_model)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalize
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Trim or pad to effective_input_len
        if x.shape[1] > self.effective_input_len:
            x = x[:, :self.effective_input_len, :]  # (batch, effective_input_len, n_features)
        elif x.shape[1] < self.effective_input_len:
            pad_len = self.effective_input_len - x.shape[1]
            x = F.pad(x, (0, 0, 0, pad_len))  # (batch, effective_input_len, n_features)
        
        # Channel-independent processing: process each feature separately
        # Reshape to (batch * n_features, n_patches, patch_size)
        x = x.permute(0, 2, 1)  # (batch, n_features, effective_input_len)
        x = x.reshape(batch_size * self.n_features, self.n_patches, self.patch_size)  # (batch*n_features, n_patches, patch_size)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch*n_features, n_patches, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed  # (batch*n_features, n_patches, d_model)
        
        # Transformer with residual and DropPath
        # Self-attention
        attn_out, _ = self.attn(x, x, x)  # (batch*n_features, n_patches, d_model)
        x = self.norm1(x + self.apply_drop_path(attn_out, self.drop_path_prob, self.training))  # (batch*n_features, n_patches, d_model)
        
        # FFN
        ffn_out = self.ffn(x)  # (batch*n_features, n_patches, d_model)
        x = self.norm2(x + self.apply_drop_path(ffn_out, self.drop_path_prob, self.training))  # (batch*n_features, n_patches, d_model)
        
        # Flatten and project to output length
        x = x.reshape(batch_size * self.n_features, -1)  # (batch*n_features, n_patches * d_model)
        x = self.head(x)  # (batch*n_features, output_len)
        
        # Reshape back: (batch, n_features, output_len) -> (batch, output_len, n_features)
        x = x.reshape(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalize
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x  # (batch, output_len, n_features)

