import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class PatchTSTChannelIndependentRevIN_v2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 128, 256],
            'n_layers': [1, 2, 4],
            'n_heads': [2, 4, 8],
            'dropout': [0.0, 0.1, 0.3],
            'patch_size': [4, 8, 16],
            'ffn_ratio': [2, 4],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with conservative defaults for small dataset
        self.d_model = self.config.get('d_model', 128)
        self.n_layers = self.config.get('n_layers', 2)
        self.n_heads = self.config.get('n_heads', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.patch_size = self.config.get('patch_size', 8)
        self.ffn_ratio = self.config.get('ffn_ratio', 2)
        
        # Ensure d_model divisible by n_heads
        assert self.d_model % self.n_heads == 0, f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        
        # Calculate number of patches
        self.n_patches = max(1, self.input_len // self.patch_size)
        self.actual_patch_size = self.input_len // self.n_patches  # Adjust to fit exactly
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding: linear projection of each patch for each channel independently
        self.patch_embed = nn.Linear(self.actual_patch_size, self.d_model)  # (batch*n_features, n_patches, d_model)
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, self.d_model) * 0.02)
        
        # Layer scale for stable deep training (small init)
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, self.d_model) * 0.01) for _ in range(self.n_layers)
        ])
        
        # Transformer encoder layers with gated residual connections
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            # Self-attention
            attn = nn.MultiheadAttention(
                self.d_model, self.n_heads, dropout=self.dropout, batch_first=True
            )
            # FFN
            ffn = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * self.ffn_ratio),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * self.ffn_ratio, self.d_model),
                nn.Dropout(self.dropout),
            )
            # Layer norms
            norm1 = nn.LayerNorm(self.d_model)
            norm2 = nn.LayerNorm(self.d_model)
            # Gating parameters
            gate_attn = nn.Linear(self.d_model, self.d_model)
            gate_ffn = nn.Linear(self.d_model, self.d_model)
            
            self.layers.append(nn.ModuleDict({
                'attn': attn,
                'ffn': ffn,
                'norm1': norm1,
                'norm2': norm2,
                'gate_attn': gate_attn,
                'gate_ffn': gate_ffn,
            }))
        
        # Final norm
        self.final_norm = nn.LayerNorm(self.d_model)
        
        # Prediction head: flatten patches and project to output
        self.head = nn.Sequential(
            nn.Linear(self.n_patches * self.d_model, self.output_len),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel independence: process each feature independently
        # (batch, input_len, n_features) -> (batch * n_features, input_len, 1)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = x.reshape(batch_size * self.n_features, self.input_len)  # (batch*n_features, input_len)
        
        # Create patches
        # Pad if necessary to make divisible
        if self.input_len % self.n_patches != 0:
            pad_len = self.n_patches * self.actual_patch_size - self.input_len
            x = F.pad(x, (0, pad_len))  # (batch*n_features, padded_len)
        
        # Reshape to patches
        x = x.reshape(batch_size * self.n_features, self.n_patches, self.actual_patch_size)  # (batch*n_features, n_patches, patch_size)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch*n_features, n_patches, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed  # (batch*n_features, n_patches, d_model)
        
        # Transformer layers with gated residuals
        for i, layer in enumerate(self.layers):
            # Self-attention with gating
            residual = x  # (batch*n_features, n_patches, d_model)
            x = layer['norm1'](x)  # (batch*n_features, n_patches, d_model)
            attn_out, _ = layer['attn'](x, x, x)  # (batch*n_features, n_patches, d_model)
            gate = torch.sigmoid(layer['gate_attn'](x))  # (batch*n_features, n_patches, d_model)
            x = residual + self.layer_scales[i] * gate * attn_out  # (batch*n_features, n_patches, d_model)
            
            # FFN with gating
            residual = x  # (batch*n_features, n_patches, d_model)
            x = layer['norm2'](x)  # (batch*n_features, n_patches, d_model)
            ffn_out = layer['ffn'](x)  # (batch*n_features, n_patches, d_model)
            gate = torch.sigmoid(layer['gate_ffn'](x))  # (batch*n_features, n_patches, d_model)
            x = residual + self.layer_scales[i] * gate * ffn_out  # (batch*n_features, n_patches, d_model)
        
        # Final norm
        x = self.final_norm(x)  # (batch*n_features, n_patches, d_model)
        
        # Flatten patches and predict
        x = x.reshape(batch_size * self.n_features, self.n_patches * self.d_model)  # (batch*n_features, n_patches*d_model)
        x = self.head(x)  # (batch*n_features, output_len)
        
        # Reshape back to channel-mixed format
        x = x.reshape(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x
