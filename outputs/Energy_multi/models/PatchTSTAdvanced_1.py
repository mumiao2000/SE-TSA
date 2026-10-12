import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class PatchTSTAdvanced(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'patch_size': [2, 8, 24],
            'dropout': [0.0, 0.3],
            'd_ff': [256, 1024, 4096]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 512)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.patch_sizes = [self.config.get('patch_size', 8)]
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.d_ff = self.config.get('d_ff', 2048)
        
        # Ensure patch sizes divide input_len reasonably
        self.patch_sizes = [p for p in [2, 4, 8, 12, 16, 24] if p <= input_len // 2]
        if not self.patch_sizes:
            self.patch_sizes = [max(1, input_len // 4)]
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale patch embedding
        self.patch_embeds = nn.ModuleList()
        self.patch_norms = nn.ModuleList()
        for ps in self.patch_sizes:
            n_patches = input_len // ps
            # Channel-mixing: each patch embeds all features
            self.patch_embeds.append(nn.Linear(ps * n_features, self.d_model))
            self.patch_norms.append(nn.LayerNorm(self.d_model))
        
        # Positional encoding
        max_patches = max([input_len // ps for ps in self.patch_sizes])
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, self.d_model) * 0.02)
        
        # Transformer encoder with channel-mixing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Multi-scale fusion
        self.scale_fusion = nn.Linear(len(self.patch_sizes) * self.d_model, self.d_model)
        
        # Forecast head
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_ff, output_len * n_features)
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Multi-scale patch processing
        scale_outputs = []
        for ps, embed, norm in zip(self.patch_sizes, self.patch_embeds, self.patch_norms):
            n_patches = self.input_len // ps
            # Pad if needed
            if self.input_len % ps != 0:
                pad_len = ps - (self.input_len % ps)
                x_padded = F.pad(x, (0, 0, 0, pad_len))  # (batch, input_len+pad, n_features)
            else:
                x_padded = x
            
            # Create patches: (batch, n_patches, patch_size, n_features)
            x_patches = x_padded[:, :n_patches * ps, :].reshape(
                batch_size, n_patches, ps, self.n_features
            )  # (batch, n_patches, patch_size, n_features)
            
            # Flatten patch features: (batch, n_patches, patch_size * n_features)
            x_patches = x_patches.reshape(batch_size, n_patches, -1)
            
            # Embed to d_model
            x_embed = embed(x_patches)  # (batch, n_patches, d_model)
            x_embed = norm(x_embed)
            x_embed = x_embed + self.pos_embed[:, :n_patches, :]  # (batch, n_patches, d_model)
            x_embed = self.dropout(x_embed)
            
            # Transformer encoding
            x_encoded = self.transformer(x_embed)  # (batch, n_patches, d_model)
            
            # Global average pooling over patches
            x_pooled = x_encoded.mean(dim=1)  # (batch, d_model)
            scale_outputs.append(x_pooled)
        
        # Fuse multi-scale representations
        if len(scale_outputs) > 1:
            x_fused = torch.cat(scale_outputs, dim=-1)  # (batch, len(patch_sizes) * d_model)
            x_fused = self.scale_fusion(x_fused)  # (batch, d_model)
        else:
            x_fused = scale_outputs[0]  # (batch, d_model)
        
        # Generate forecast
        out = self.head(x_fused)  # (batch, output_len * n_features)
        out = out.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
