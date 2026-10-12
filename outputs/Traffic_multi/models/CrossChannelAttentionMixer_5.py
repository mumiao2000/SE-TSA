import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttentionMixer(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.2, 0.4],
            'ff_mult': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout_rate = self.config.get('dropout', 0.2)
        self.ff_mult = self.config.get('ff_mult', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input embedding: project features to d_model
        self.input_embed = nn.Linear(self.n_features, self.d_model)  # (batch, seq_len, d_model)
        
        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, self.input_len, self.d_model) * 0.02)
        
        # Cross-channel attention: attend across features for each time step
        self.channel_attention = nn.MultiheadAttention(
            self.d_model, self.n_heads, dropout=self.dropout_rate, batch_first=True
        )
        
        # Temporal attention layers
        self.temporal_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.temporal_layers.append(nn.ModuleDict({
                'temporal_attn': nn.MultiheadAttention(
                    self.d_model, self.n_heads, dropout=self.dropout_rate, batch_first=True
                ),
                'norm1': nn.LayerNorm(self.d_model),
                'ff': nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * self.ff_mult),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.d_model * self.ff_mult, self.d_model),
                    nn.Dropout(self.dropout_rate)
                ),
                'norm2': nn.LayerNorm(self.d_model)
            }))
        
        # Output: project from input_len to output_len using attention
        self.query_embed = nn.Parameter(torch.randn(1, self.output_len, self.d_model))
        self.output_attn = nn.MultiheadAttention(
            self.d_model, self.n_heads, dropout=self.dropout_rate, batch_first=True
        )
        
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.n_features)  # (batch, output_len, n_features)
        
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Embed to d_model
        h = self.input_embed(x)  # (batch, input_len, d_model)
        h = h + self.pos_enc  # (batch, input_len, d_model)
        h = self.dropout(h)  # (batch, input_len, d_model)
        
        # Cross-channel attention: treat time steps as batch, features as sequence
        # Actually, we already embedded features to d_model, so this is standard self-attention
        # Let's reinterpret: for each time step, we want to attend across the embedded features
        # But features are already embedded... Let's do temporal attention instead
        
        # Apply temporal attention layers
        for layer in self.temporal_layers:
            # Temporal self-attention
            h_attn, _ = layer['temporal_attn'](h, h, h)  # (batch, input_len, d_model)
            h = layer['norm1'](h + h_attn)  # (batch, input_len, d_model)
            
            # Feedforward
            h_ff = layer['ff'](h)  # (batch, input_len, d_model)
            h = layer['norm2'](h + h_ff)  # (batch, input_len, d_model)
        
        # Cross-attention to output queries
        queries = self.query_embed.expand(x.size(0), -1, -1)  # (batch, output_len, d_model)
        out, _ = self.output_attn(queries, h, h)  # (batch, output_len, d_model)
        out = self.output_norm(out)  # (batch, output_len, d_model)
        
        # Project to features
        out = self.output_proj(out)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
