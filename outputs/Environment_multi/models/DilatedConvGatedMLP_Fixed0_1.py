import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DilatedConvGatedMLP_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.3],
            'kernel_size': [3, 5, 7],
            'expansion': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.kernel_size = self.config.get('kernel_size', 5)
        self.expansion = self.config.get('expansion', 4)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.d_model)
        
        # Dilated convolution blocks
        self.conv_blocks = nn.ModuleList()
        self.gate_blocks = nn.ModuleList()
        self.proj_blocks = nn.ModuleList()
        self.norm_blocks = nn.ModuleList()
        
        for i in range(self.n_layers):
            dilation = 2 ** i
            # Causal padding: only pad on the left
            padding = (self.kernel_size - 1) * dilation
            
            self.conv_blocks.append(
                nn.Conv1d(
                    self.d_model, 
                    self.d_model * self.expansion,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    padding=padding,
                    padding_mode='zeros'
                )
            )
            
            # Projection back to d_model for gating
            self.gate_blocks.append(
                nn.Sequential(
                    nn.Conv1d(self.d_model * self.expansion, self.d_model, 1),
                    nn.SiLU()
                )
            )
            
            # Projection for residual if dimensions match (they do here since gate outputs d_model)
            self.proj_blocks.append(
                nn.Conv1d(self.d_model * self.expansion, self.d_model, 1)
                if self.expansion != 1 else nn.Identity()
            )
            
            self.norm_blocks.append(nn.LayerNorm(self.d_model))
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.n_features)
        
        # Direct prediction from encoded sequence
        self.predictor = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # RevIN normalize
        x = self.revin(x, mode="norm")  # (batch, input_len, n_features)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, input_len, d_model)
        
        # Transpose for conv: (batch, d_model, input_len)
        x = x.permute(0, 2, 1)  # (batch, d_model, input_len)
        
        # Dilated convolutions with gating
        for conv, gate, proj, norm in zip(self.conv_blocks, self.gate_blocks, self.proj_blocks, self.norm_blocks):
            residual = x  # (batch, d_model, input_len)
            
            x = conv(x)  # (batch, d_model * expansion, input_len + padding)
            
            # Trim causal padding to maintain sequence length
            # Causal conv pads only on left, so trim from right
            trim = x.size(2) - residual.size(2)
            if trim > 0:
                x = x[:, :, :-trim]  # (batch, d_model * expansion, input_len)
            
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Gating
            x = gate(x)  # (batch, d_model, input_len)
            
            # Residual connection
            x = x + residual  # (batch, d_model, input_len)
            
            # Layer norm on features
            x_t = x.permute(0, 2, 1)  # (batch, input_len, d_model)
            x_t = norm(x_t)  # (batch, input_len, d_model)
            x = x_t.permute(0, 2, 1)  # (batch, d_model, input_len)
        
        # Predict: (batch, d_model, input_len) -> (batch, d_model, output_len)
        x = self.predictor(x)  # (batch, d_model, output_len)
        
        # Project back to features
        x = x.permute(0, 2, 1)  # (batch, output_len, d_model)
        x = self.output_proj(x)  # (batch, output_len, n_features)
        
        # RevIN denormalize
        x = self.revin(x, mode="denorm")  # (batch, output_len, n_features)
        
        return x

