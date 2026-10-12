import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttentionForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3],
            'bottleneck_ratio': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.bottleneck_ratio = self.config.get('bottleneck_ratio', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Ensure d_model divisible by n_heads
        assert self.d_model % self.n_heads == 0, f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        self.head_dim = self.d_model // self.n_heads
        
        # Input embedding: map scalar value at each time step to d_model
        self.input_embed = nn.Linear(1, self.d_model)
        
        # Relative positional encodings
        self.rel_pos_time = nn.Embedding(2 * input_len - 1, self.head_dim)
        self.rel_pos_channel = nn.Embedding(2 * n_features - 1, self.head_dim)
        
        # Factorized attention layers
        self.time_attn_layers = nn.ModuleList()
        self.channel_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        
        bottleneck_dim = max(self.d_model // self.bottleneck_ratio, 16)
        
        for _ in range(self.n_layers):
            # Time-wise attention (batch*n_features as batch)
            self.time_attn_layers.append(nn.MultiheadAttention(
                self.d_model, self.n_heads, dropout=self.dropout_rate, batch_first=True
            ))
            
            # Channel-wise attention (batch*input_len as batch)
            self.channel_attn_layers.append(nn.MultiheadAttention(
                self.d_model, self.n_heads, dropout=self.dropout_rate, batch_first=True
            ))
            
            # FFN with bottleneck
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(self.d_model, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(bottleneck_dim, self.d_model),
                nn.Dropout(self.dropout_rate)
            ))
        
        # Layer norms
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, output_len)
        )
        
    def _get_rel_pos(self, seq_len, rel_pos_embed):
        # Create relative position matrix
        positions = torch.arange(seq_len, device=rel_pos_embed.weight.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
        rel_pos = rel_pos + seq_len - 1  # shift to positive indices
        return rel_pos_embed(rel_pos)  # (seq_len, seq_len, head_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Embed: treat features as separate sequences
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Reshape for embedding: each feature's time series becomes a batch item
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size * self.n_features, self.input_len)  # (batch*n_features, input_len)
        x_flat = x_flat.unsqueeze(-1)  # (batch*n_features, input_len, 1)
        h = self.input_embed(x_flat)  # (batch*n_features, input_len, d_model)
        
        # Apply layers
        for i, (time_attn, channel_attn, ffn) in enumerate(
            zip(self.time_attn_layers, self.channel_attn_layers, self.ffn_layers)
        ):
            # Time-wise attention
            h_norm = self.norm1(h)  # (batch*n_features, input_len, d_model)
            h_time, _ = time_attn(h_norm, h_norm, h_norm)  # (batch*n_features, input_len, d_model)
            h = h + h_time  # (batch*n_features, input_len, d_model)
            
            # Reshape for channel-wise attention
            # (batch*n_features, input_len, d_model) -> (batch, n_features, input_len, d_model)
            h_4d = h.view(batch_size, self.n_features, self.input_len, self.d_model)  # (batch, n_features, input_len, d_model)
            # -> (batch*input_len, n_features, d_model)
            h_channel_in = h_4d.permute(0, 2, 1, 3).reshape(batch_size * self.input_len, self.n_features, self.d_model)  # (batch*input_len, n_features, d_model)
            
            h_norm = self.norm2(h_channel_in)  # (batch*input_len, n_features, d_model)
            h_channel, _ = channel_attn(h_norm, h_norm, h_norm)  # (batch*input_len, n_features, d_model)
            h_channel = h_channel + h_channel_in  # (batch*input_len, n_features, d_model)
            
            # Reshape back
            h = h_channel.view(batch_size, self.input_len, self.n_features, self.d_model)  # (batch, input_len, n_features, d_model)
            h = h.permute(0, 2, 1, 3).reshape(batch_size * self.n_features, self.input_len, self.d_model)  # (batch*n_features, input_len, d_model)
            
            # FFN
            h_norm = self.norm3(h)  # (batch*n_features, input_len, d_model)
            h = h + ffn(h_norm)  # (batch*n_features, input_len, d_model)
        
        # Output projection: aggregate over time, project to output_len
        # Use last time step representation
        h_last = h[:, -1, :]  # (batch*n_features, d_model)
        out = self.output_proj(h_last)  # (batch*n_features, output_len)
        
        # Reshape to (batch, output_len, n_features)
        out = out.view(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

