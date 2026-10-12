import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class MultiScaleDecompAttentionRevIN_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_scales': [2, 3, 4],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_scales = self.config.get('n_scales', 3)
        self.dropout_p = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Scales for decomposition (kernel sizes for moving average)
        self.scales = [2 ** i + 1 for i in range(self.n_scales)]  # [3, 5, 9, ...]
        
        # Input projection per scale
        self.input_projs = nn.ModuleList([
            nn.Linear(self.n_features, self.d_model) for _ in range(self.n_scales)
        ])
        
        # Factorized attention: time-wise and feature-wise
        self.time_attns = nn.ModuleList([
            nn.MultiheadAttention(self.d_model, self.n_heads, batch_first=True, dropout=self.dropout_p)
            for _ in range(self.n_scales)
        ])
        
        self.scale_combiner = nn.Sequential(
            nn.Linear(self.d_model * self.n_scales, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_model, self.n_features)
        )
        
        # Direct mapping
        self.direct = nn.Linear(self.input_len, self.output_len)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Multi-scale decomposition
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            # Adjust scale if too large
            actual_scale = min(scale, self.input_len)
            if actual_scale % 2 == 0:
                actual_scale = actual_scale - 1  # ensure odd for symmetric padding
            actual_scale = max(actual_scale, 1)
            
            padding = actual_scale // 2
            
            # Pad symmetrically to maintain length
            x_padded = F.pad(
                x.permute(0, 2, 1),  # (batch, n_features, input_len)
                (padding, padding),
                mode='replicate'
            )  # (batch, n_features, input_len + 2*padding)
            
            # Moving average for trend
            trend = F.avg_pool1d(
                x_padded,
                kernel_size=actual_scale,
                stride=1,
                padding=0,
                count_include_pad=False
            )  # (batch, n_features, input_len)
            trend = trend.permute(0, 2, 1)  # (batch, input_len, n_features)
            
            seasonal = x - trend  # (batch, input_len, n_features)
            
            # Use seasonal component for this scale
            comp = seasonal  # (batch, input_len, n_features)
            
            # Project
            comp = self.input_projs[i](comp)  # (batch, input_len, d_model)
            
            # Factorized attention (time-wise only for efficiency)
            attn_out, _ = self.time_attns[i](comp, comp, comp)  # (batch, input_len, d_model)
            
            # Pool to fixed size then expand
            pooled = attn_out.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
            scale_outputs.append(pooled)  # (batch, 1, d_model)
        
        # Combine scales
        combined = torch.cat(scale_outputs, dim=-1)  # (batch, 1, d_model * n_scales)
        combined = combined.expand(-1, self.output_len, -1)  # (batch, output_len, d_model * n_scales)
        
        out = self.scale_combiner(combined)  # (batch, output_len, n_features)
        
        # Add direct skip
        direct = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        direct = self.direct(direct)  # (batch, n_features, output_len)
        direct = direct.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = out + direct  # (batch, output_len, n_features)
        
        x = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)

