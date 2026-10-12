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
            'ffn_ratio': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 512)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.patch_size = self.config.get('patch_size', 8)
        self.ffn_ratio = self.config.get('ffn_ratio', 4)
        
        # Ensure patch_size divides input_len
        self.patch_size = min(self.patch_size, input_len)
        while input_len % self.patch_size != 0 and self.patch_size > 1:
            self.patch_size -= 1
        
        self.n_patches = input_len // self.patch_size
        self.d_ff = self.d_model * self.ffn_ratio
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding: project each patch to d_model
        self.patch_embed = nn.Linear(self.patch_size, self.d_model)
        
        # Positional encoding for patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, self.d_model) * 0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Prediction head: predict output patches
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.patch_size)
        )
        
        # Calculate how many patches to predict
        self.out_patches = (output_len + self.patch_size - 1) // self.patch_size
        self.patch_predictor = nn.Linear(self.n_patches, self.out_patches)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-independent: process each feature separately
        batch_size = x.shape[0]
        
        # Reshape to (batch * n_features, input_len)
        x = x.permute(0, 2, 1).reshape(batch_size * self.n_features, self.input_len)  # (batch * n_features, input_len)
        
        # Create patches: (batch * n_features, n_patches, patch_size)
        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)  # (batch * n_features, n_patches, patch_size)
        
        # Embed patches
        x = self.patch_embed(x)  # (batch * n_features, n_patches, d_model)
        x = x + self.pos_embed  # (batch * n_features, n_patches, d_model)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch * n_features, n_patches, d_model)
        
        # Predict output patches: transpose for linear projection across patches
        x = x.permute(0, 2, 1)  # (batch * n_features, d_model, n_patches)
        x = self.patch_predictor(x)  # (batch * n_features, d_model, out_patches)
        x = x.permute(0, 2, 1)  # (batch * n_features, out_patches, d_model)
        
        # Map to patch values
        x = self.head(x)  # (batch * n_features, out_patches, patch_size)
        
        # Reshape to sequence
        x = x.reshape(batch_size * self.n_features, -1)  # (batch * n_features, out_patches * patch_size)
        x = x[:, :self.output_len]  # (batch * n_features, output_len)
        
        # Reshape back
        x = x.reshape(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x
