import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossAttentionForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'ff_mult': [2, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.ff_mult = self.config.get('ff_mult', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.d_model)
        
        # Learnable query tokens for each output step
        self.query_tokens = nn.Parameter(torch.randn(self.output_len, self.d_model))
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(self.d_model, self.n_heads, dropout=self.dropout, batch_first=True)
            )
            self.norm1_layers.append(nn.LayerNorm(self.d_model))
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(self.d_model, self.d_model * self.ff_mult),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * self.ff_mult, self.d_model),
                nn.Dropout(self.dropout)
            ))
            self.norm2_layers.append(nn.LayerNorm(self.d_model))
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project input to d_model
        x = self.input_proj(x)  # (batch, input_len, d_model)
        
        # Expand query tokens for batch
        batch_size = x.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, output_len, d_model)
        
        # Cross-attention layers
        for i in range(self.n_layers):
            # Cross-attention: queries attend to input keys/values
            attn_out, _ = self.cross_attn_layers[i](queries, x, x)  # (batch, output_len, d_model)
            queries = self.norm1_layers[i](queries + attn_out)  # (batch, output_len, d_model)
            
            # Feed-forward
            ff_out = self.ffn_layers[i](queries)  # (batch, output_len, d_model)
            queries = self.norm2_layers[i](queries + ff_out)  # (batch, output_len, d_model)
        
        # Project to output features
        out = self.output_proj(queries)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

