import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class PatchTSTChannelIndependentRevIN(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'patch_size': [2, 8, 24],
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.patch_size = self.config.get('patch_size', 8)
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_p = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(n_features, affine=True)
        
        # Ensure patch_size divides input_len reasonably
        self.n_patches = input_len // self.patch_size
        self.remainder = input_len % self.patch_size
        
        # Patch embedding (shared across channels)
        self.patch_embed = nn.Linear(self.patch_size, self.d_model)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, self.d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout_p,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Prediction head: predict future patches
        self.n_pred_patches = (output_len + self.patch_size - 1) // self.patch_size
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_model, self.patch_size * self.n_pred_patches)
        )

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Channel independence: process each channel separately
        # Reshape to (batch * n_features, input_len, 1)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = x.reshape(batch_size * self.n_features, self.input_len, 1)  # (batch*n_features, input_len, 1)
        
        # Pad if needed
        if self.remainder > 0:
            pad_len = self.patch_size - self.remainder
            x = F.pad(x, (0, 0, 0, pad_len))  # (batch*n_features, padded_len, 1)
            actual_n_patches = (self.input_len + pad_len) // self.patch_size
        else:
            actual_n_patches = self.n_patches
        
        # Create patches: (batch*n_features, n_patches, patch_size)
        x = x[:, :actual_n_patches * self.patch_size, :]  # ensure clean division
        x = x.reshape(batch_size * self.n_features, actual_n_patches, self.patch_size)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch*n_features, n_patches, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size * self.n_features, -1, -1)  # (batch*n_features, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch*n_features, n_patches+1, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed[:, :x.shape[1], :]  # (batch*n_features, n_patches+1, d_model)
        
        # Transformer
        x = self.transformer(x)  # (batch*n_features, n_patches+1, d_model)
        
        # Use CLS token for prediction
        cls_out = x[:, 0, :]  # (batch*n_features, d_model)
        
        # Predict future
        pred = self.head(cls_out)  # (batch*n_features, patch_size * n_pred_patches)
        pred = pred.reshape(batch_size * self.n_features, self.n_pred_patches, self.patch_size)
        pred = pred.reshape(batch_size * self.n_features, -1)  # flatten patches
        pred = pred[:, :self.output_len]  # truncate to output_len
        pred = pred.unsqueeze(-1)  # (batch*n_features, output_len, 1)
        
        # Reshape back: (batch, n_features, output_len) -> (batch, output_len, n_features)
        pred = pred.reshape(batch_size, self.n_features, self.output_len)
        pred = pred.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Add last value as residual
        last_val = x[:, -1, :] if x.dim() == 3 else torch.zeros(batch_size, self.n_features, device=x.device)
        # Actually last_val from original x before reshape
        # We need to recalculate - use original normalized input
        # This is approximate; better to store before
        
        x = self.revin(pred, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)

