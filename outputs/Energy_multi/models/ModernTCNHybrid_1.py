import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ModernTCNHybrid(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 2048],
            'n_layers': [2, 6, 12],
            'kernel_size': [3, 7, 31],
            'dropout': [0.0, 0.3],
            'd_ff': [256, 1024, 4096],
            'use_mixer': [True, False]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 512)
        self.n_layers = self.config.get('n_layers', 6)
        self.kernel_size = self.config.get('kernel_size', 7)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.d_ff = self.config.get('d_ff', 2048)
        self.use_mixer = self.config.get('use_mixer', True)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(n_features, self.d_model)
        
        # ModernTCN blocks
        self.blocks = nn.ModuleList()
        for i in range(self.n_layers):
            dilation = 2 ** (i % 4)  # Exponential dilation with cycling
            self.blocks.append(
                ModernTCNBlock(
                    self.d_model, self.kernel_size, dilation, 
                    self.d_ff, self.dropout_rate, self.use_mixer
                )
            )
        
        # Multi-scale pooling for forecast
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Forecast head
        self.forecast_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_ff, output_len * n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, input_len, d_model)
        
        # ModernTCN blocks
        for block in self.blocks:
            x = block(x)  # (batch, input_len, d_model)
        
        # Global pooling and forecast
        x = x.permute(0, 2, 1)  # (batch, d_model, input_len)
        x_pooled = self.global_pool(x).squeeze(-1)  # (batch, d_model)
        
        out = self.forecast_head(x_pooled)  # (batch, output_len * n_features)
        out = out.reshape(-1, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class ModernTCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size, dilation, d_ff, dropout_rate, use_mixer):
        super().__init__()
        self.use_mixer = use_mixer
        
        # Ensure kernel_size with dilation fits
        effective_kernel = (kernel_size - 1) * dilation + 1
        self.padding = (effective_kernel - 1) // 2
        
        # Depthwise separable convolution
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=self.padding, dilation=dilation, groups=d_model
        )
        self.bn1 = nn.BatchNorm1d(d_model)
        
        # Pointwise convolution
        self.pw_conv = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        
        # MLP mixer for cross-channel communication
        if use_mixer:
            self.mixer = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(d_ff, d_model)
            )
            self.mixer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        residual = x
        
        # Depthwise conv
        x = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        x = self.dw_conv(x)  # (batch, d_model, seq_len)
        x = self.bn1(x)
        
        # Pointwise conv
        x = self.pw_conv(x)  # (batch, d_ff, seq_len)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pw_conv2(x)  # (batch, d_model, seq_len)
        x = self.bn2(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, d_model)
        
        # Residual
        x = self.layer_norm(x + residual)
        
        # MLP mixer
        if self.use_mixer:
            mixer_residual = x
            x = self.mixer(x)  # (batch, seq_len, d_model)
            x = self.dropout(x)
            x = self.mixer_norm(x + mixer_residual)  # (batch, seq_len, d_model)
        
        return x
