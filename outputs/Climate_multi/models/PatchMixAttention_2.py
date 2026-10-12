import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class PatchMixAttention(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'patch_sizes': [[2, 4, 8], [4, 8, 16], [8, 16, 32]],
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.patch_sizes = self.config.get('patch_sizes', [4, 8, 16])
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Ensure d_model divisible by n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads
        
        # RevIN for normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding: project (patch_size, n_features) to d_model
        self.patch_embeds = nn.ModuleList()
        for ps in self.patch_sizes:
            # We use 1D conv for efficient patch embedding
            embed = nn.Conv1d(
                self.n_features, self.d_model,
                kernel_size=ps, stride=ps
            )
            self.patch_embeds.append(embed)
        
        # Positional encoding (learned)
        max_patches = max(self.input_len // min(self.patch_sizes), self.output_len) + 10
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, self.d_model) * 0.02)
        
        # Cross-patch attention (lightweight, within each scale)
        self.scale_attns = nn.ModuleList()
        for _ in self.patch_sizes:
            self.scale_attns.append(nn.MultiheadAttention(
                self.d_model, self.n_heads, 
                dropout=self.dropout, 
                batch_first=True
            ))
        
        # Cross-scale fusion: attend across different patch scales
        self.scale_fusion = nn.MultiheadAttention(
            self.d_model, self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Channel mixing MLP
        self.channel_mix = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 4, self.d_model)
        )
        
        # Output decoder: predict output_len steps
        # We use a query-based decoder
        self.output_queries = nn.Parameter(torch.randn(1, self.output_len, self.d_model) * 0.02)
        
        self.decoder = nn.MultiheadAttention(
            self.d_model, self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Final projection to features
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.n_features)
        )
        
        # Direct linear forecast for residual
        self.direct = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode="norm")  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Multi-scale patch processing
        scale_outputs = []
        for i, (ps, embed, attn) in enumerate(zip(self.patch_sizes, self.patch_embeds, self.scale_attns)):
            # Pad if needed
            pad_len = (ps - x.shape[1] % ps) % ps
            x_padded = F.pad(x.permute(0, 2, 1), (0, pad_len), mode='replicate')  # (batch, n_features, padded_len)
            
            # Embed patches
            patches = embed(x_padded)  # (batch, d_model, n_patches)
            patches = patches.permute(0, 2, 1)  # (batch, n_patches, d_model)
            
            # Add positional encoding
            n_patches = patches.shape[1]
            patches = patches + self.pos_embed[:, :n_patches, :]  # (batch, n_patches, d_model)
            
            # Self-attention on patches
            attn_out, _ = attn(patches, patches, patches)  # (batch, n_patches, d_model)
            attn_out = attn_out + patches  # residual
            
            # Channel mixing
            mixed = self.channel_mix(attn_out)  # (batch, n_patches, d_model)
            mixed = mixed + attn_out  # residual
            
            # Interpolate to common length for fusion
            target_len = self.input_len // min(self.patch_sizes)
            if n_patches != target_len:
                mixed = mixed.permute(0, 2, 1)  # (batch, d_model, n_patches)
                mixed = F.interpolate(mixed, size=target_len, mode='linear', align_corners=False)
                mixed = mixed.permute(0, 2, 1)  # (batch, target_len, d_model)
            
            scale_outputs.append(mixed)
        
        # Stack scales: (batch, n_scales, target_len, d_model)
        scale_stack = torch.stack(scale_outputs, dim=1)
        
        # Flatten for cross-scale attention
        scale_flat = scale_stack.reshape(batch_size * len(self.patch_sizes), target_len, self.d_model)
        
        # Cross-scale fusion
        fused, _ = self.scale_fusion(scale_flat, scale_flat, scale_flat)  # (batch*n_scales, target_len, d_model)
        fused = fused + scale_flat  # residual
        
        # Reshape back and pool across scales
        fused = fused.reshape(batch_size, len(self.patch_sizes), target_len, self.d_model)
        fused = fused.mean(dim=1)  # (batch, target_len, d_model)
        
        # Decode with queries
        queries = self.output_queries.expand(batch_size, -1, -1)  # (batch, output_len, d_model)
        decoded, _ = self.decoder(queries, fused, fused)  # (batch, output_len, d_model)
        
        # Final projection
        out = self.output_proj(decoded)  # (batch, output_len, n_features)
        
        # Direct residual
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        direct_out = self.direct(x_perm)  # (batch, n_features, output_len)
        direct_out = direct_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = out + direct_out  # (batch, output_len, n_features)
        
        out = self.revin(out, mode="denorm")  # (batch, output_len, n_features)
        return out
