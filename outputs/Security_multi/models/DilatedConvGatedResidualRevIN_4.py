import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DilatedConvGatedResidualRevIN(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [2, 4, 8],
            'kernel_size': [2, 3, 5],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.kernel_size = self.config.get('kernel_size', 3)
        self.dropout_p = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Conv1d(n_features, self.hidden_dim, 1)
        
        # Dilated convolution blocks with gating
        self.conv_blocks = nn.ModuleList()
        for i in range(self.n_layers):
            dilation = 2 ** i
            self.conv_blocks.append(nn.ModuleDict({
                'conv': nn.Conv1d(
                    self.hidden_dim, 
                    self.hidden_dim * 2,  # 2x for gating
                    kernel_size=self.kernel_size,
                    padding=(self.kernel_size - 1) * dilation,
                    dilation=dilation
                ),
                'norm': nn.LayerNorm(self.hidden_dim),
                'dropout': nn.Dropout(self.dropout_p)
            }))
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, n_features)
        
        # Direct skip connection from input to output
        self.skip_proj = nn.Linear(input_len, output_len)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save for skip
        x_input = x  # (batch, input_len, n_features)
        
        # Input projection
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.input_proj(x)  # (batch, hidden_dim, input_len)
        
        # Dilated conv blocks with gating
        for block in self.conv_blocks:
            residual = x  # (batch, hidden_dim, input_len)
            x = block['conv'](x)  # (batch, hidden_dim*2, input_len + padding)
            x = x[:, :, :residual.shape[2]]  # (batch, hidden_dim*2, input_len)
            
            # Gating: split into gate and value
            gate, value = torch.chunk(x, 2, dim=1)  # each (batch, hidden_dim, input_len)
            x = torch.sigmoid(gate) * torch.tanh(value)  # (batch, hidden_dim, input_len)
            x = block['dropout'](x)  # (batch, hidden_dim, input_len)
            
            # Residual connection
            x = x + residual  # (batch, hidden_dim, input_len)
        
        # Prepare for output
        x = x.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
        x = self.output_proj(x)  # (batch, input_len, n_features)
        
        # Add direct skip
        skip = x_input.permute(0, 2, 1)  # (batch, n_features, input_len)
        skip = self.skip_proj(skip)  # (batch, n_features, output_len)
        skip = skip.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Temporal interpolation for main path
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = F.interpolate(x, size=self.output_len, mode='linear', align_corners=False)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = x + skip  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)

