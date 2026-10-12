import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HybridTCNAttention(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_channels': [64, 256, 512],
            'n_tcn_layers': [2, 4, 8],
            'n_attn_layers': [1, 2, 4],
            'kernel_size': [3, 7],
            'dropout': [0.0, 0.3],
            'd_model': [64, 256, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_channels = self.config.get('hidden_channels', 256)
        self.n_tcn_layers = self.config.get('n_tcn_layers', 4)
        self.n_attn_layers = self.config.get('n_attn_layers', 2)
        self.kernel_size = self.config.get('kernel_size', 3)
        self.dropout = self.config.get('dropout', 0.1)
        self.d_model = self.config.get('d_model', 256)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_channels)
        
        # TCN layers with increasing dilation
        self.tcn_convs = nn.ModuleList()
        self.tcn_norms = nn.ModuleList()
        self.tcn_dropouts = nn.ModuleList()
        
        for i in range(self.n_tcn_layers):
            dilation = 2 ** i
            padding = (self.kernel_size - 1) * dilation // 2
            self.tcn_convs.append(
                nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 
                         padding=padding, dilation=dilation)
            )
            self.tcn_norms.append(nn.LayerNorm(self.hidden_channels))
            self.tcn_dropouts.append(nn.Dropout(self.dropout))
        
        # TCN to attention projection
        self.tcn_to_attn = nn.Linear(self.hidden_channels, self.d_model)
        
        # Learnable future queries
        self.future_queries = nn.Parameter(torch.randn(self.output_len, self.d_model))
        
        # Attention layers for decoding
        self.attn_layers = nn.ModuleList()
        self.attn_norms1 = nn.ModuleList()
        self.attn_ffns = nn.ModuleList()
        self.attn_norms2 = nn.ModuleList()
        
        for _ in range(self.n_attn_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(self.d_model, num_heads=8, dropout=self.dropout, batch_first=True)
            )
            self.attn_norms1.append(nn.LayerNorm(self.d_model))
            self.attn_ffns.append(nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 4),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * 4, self.d_model),
                nn.Dropout(self.dropout)
            ))
            self.attn_norms2.append(nn.LayerNorm(self.d_model))
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden channels
        x = self.input_proj(x)  # (batch, input_len, hidden_channels)
        
        # TCN forward (need channels first for conv1d)
        x = x.permute(0, 2, 1)  # (batch, hidden_channels, input_len)
        
        for i in range(self.n_tcn_layers):
            residual = x  # (batch, hidden_channels, input_len)
            x = self.tcn_convs[i](x)  # (batch, hidden_channels, input_len)
            x = x.permute(0, 2, 1)  # (batch, input_len, hidden_channels)
            x = self.tcn_norms[i](x)  # (batch, input_len, hidden_channels)
            x = F.gelu(x)  # (batch, input_len, hidden_channels)
            x = x.permute(0, 2, 1)  # (batch, hidden_channels, input_len)
            x = self.tcn_dropouts[i](x)  # (batch, hidden_channels, input_len)
            x = x + residual  # (batch, hidden_channels, input_len)
        
        # Back to (batch, seq, features)
        x = x.permute(0, 2, 1)  # (batch, input_len, hidden_channels)
        
        # Project to d_model for attention
        x = self.tcn_to_attn(x)  # (batch, input_len, d_model)
        
        # Prepare future queries
        batch_size = x.shape[0]
        queries = self.future_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, output_len, d_model)
        
        # Cross-attention decoding
        for i in range(self.n_attn_layers):
            attn_out, _ = self.attn_layers[i](queries, x, x)  # (batch, output_len, d_model)
            queries = self.attn_norms1[i](queries + attn_out)  # (batch, output_len, d_model)
            ff_out = self.attn_ffns[i](queries)  # (batch, output_len, d_model)
            queries = self.attn_norms2[i](queries + ff_out)  # (batch, output_len, d_model)
        
        # Project to output
        out = self.output_proj(queries)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

