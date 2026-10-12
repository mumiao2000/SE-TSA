import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttention(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 512],
            'n_heads': [2, 4, 11],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.1, 0.3],
            'kernel_size': [3, 7, 15]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 4)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.kernel_size = self.config.get('kernel_size', 7)
        
        # Ensure n_heads divides d_model
        while self.d_model % self.n_heads != 0 and self.n_heads > 1:
            self.n_heads -= 1
        if self.d_model % self.n_heads != 0:
            self.d_model = (self.d_model // self.n_heads + 1) * self.n_heads
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Temporal embedding: conv along time
        self.temporal_embed = nn.Conv1d(
            self.n_features, self.d_model, 
            kernel_size=self.kernel_size, 
            padding=self.kernel_size // 2
        )
        
        # Cross-channel attention blocks
        self.attn_layers = nn.ModuleList([
            CrossChannelAttnLayer(self.d_model, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Temporal reduction to output_len
        self.temporal_agg = nn.Sequential(
            nn.AdaptiveAvgPool1d(self.output_len),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.n_features)
        
        # Direct mapping
        self.direct = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save for direct connection
        x_direct = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_direct = self.direct(x_direct)  # (batch, n_features, output_len)
        x_direct = x_direct.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Temporal convolution embedding
        x_conv = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_embed = self.temporal_embed(x_conv)  # (batch, d_model, input_len)
        
        # Cross-channel attention: treat time as batch, channels as sequence
        x_attn = x_embed.permute(2, 0, 1)  # (input_len, batch, d_model)
        
        for layer in self.attn_layers:
            x_attn = layer(x_attn)  # (input_len, batch, d_model)
        
        x_attn = x_attn.permute(1, 2, 0)  # (batch, d_model, input_len)
        
        # Temporal aggregation to output length
        x_out = self.temporal_agg(x_attn)  # (batch, d_model, output_len)
        x_out = x_out.permute(0, 2, 1)  # (batch, output_len, d_model)
        
        # Project to features
        x_out = self.output_proj(x_out)  # (batch, output_len, n_features)
        
        # Add direct connection
        x_out = x_out + x_direct  # (batch, output_len, n_features)
        
        x_out = self.revin(x_out, mode='denorm')  # (batch, output_len, n_features)
        return x_out


class CrossChannelAttnLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (seq_len, batch, d_model) where seq_len is time steps
        # Self-attention across the sequence dimension (time)
        attn_out, _ = self.attn(x, x, x)  # (seq_len, batch, d_model)
        x = self.norm1(x + attn_out)  # (seq_len, batch, d_model)
        
        # FFN
        ffn_out = self.ffn(x)  # (seq_len, batch, d_model)
        x = self.norm2(x + ffn_out)  # (seq_len, batch, d_model)
        
        return x
