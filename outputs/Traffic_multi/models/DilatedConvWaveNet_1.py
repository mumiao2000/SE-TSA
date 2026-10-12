import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DilatedConvWaveNet(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_channels': [64, 256, 512],
            'n_layers': [4, 6, 8],
            'dropout': [0.0, 0.3],
            'kernel_size': [2, 3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_channels = self.config.get('hidden_channels', 256)
        self.n_layers = self.config.get('n_layers', 6)
        self.dropout = self.config.get('dropout', 0.1)
        self.kernel_size = self.config.get('kernel_size', 2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Conv1d(self.n_features, self.hidden_channels, 1)
        
        # Dilated causal convolution blocks
        self.conv_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for i in range(self.n_layers):
            dilation = 2 ** i
            self.conv_blocks.append(nn.ModuleDict({
                'conv_f': nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 
                                   padding=(self.kernel_size - 1) * dilation, dilation=dilation),
                'conv_g': nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size,
                                   padding=(self.kernel_size - 1) * dilation, dilation=dilation),
                'bn': nn.BatchNorm1d(self.hidden_channels),
            }))
            self.skip_convs.append(nn.Conv1d(self.hidden_channels, self.hidden_channels, 1))
        
        # Output layers
        self.output_conv1 = nn.Conv1d(self.hidden_channels, self.hidden_channels, 1)
        self.output_conv2 = nn.Conv1d(self.hidden_channels, self.n_features, 1)
        
        # Projection from input_len to output_len
        self.length_proj = nn.Linear(input_len, output_len)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for conv1d: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Input projection
        h = self.input_proj(x)  # (batch, hidden_channels, input_len)
        
        # Causal padding and dilated convolutions
        skip_sum = 0
        for i, block in enumerate(self.conv_blocks):
            residual = h
            
            # Gated activation: tanh(f) * sigmoid(g)
            f = block['conv_f'](h)
            g = block['conv_g'](h)
            # Remove extra padding from causal convolution (keep left side)
            if f.size(2) > self.input_len:
                f = f[:, :, :self.input_len]
                g = g[:, :, :self.input_len]
            
            h = torch.tanh(f) * torch.sigmoid(g)  # (batch, hidden_channels, input_len)
            h = block['bn'](h)
            h = self.dropout_layer(h)
            
            # Skip connection
            skip = self.skip_convs[i](h)
            if skip.size(2) > self.input_len:
                skip = skip[:, :, :self.input_len]
            skip_sum = skip_sum + skip
            
            # Residual
            h = h + residual[:, :, :h.size(2)]
        
        # Combine skip connections
        out = F.relu(skip_sum)  # (batch, hidden_channels, input_len)
        out = self.output_conv1(out)  # (batch, hidden_channels, input_len)
        out = F.relu(out)
        out = self.dropout_layer(out)
        out = self.output_conv2(out)  # (batch, n_features, input_len)
        
        # Project to output length
        out = self.length_proj(out)  # (batch, n_features, output_len)
        
        # Transpose back
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

