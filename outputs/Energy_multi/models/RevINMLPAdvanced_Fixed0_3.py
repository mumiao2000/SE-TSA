import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class RevINMLPAdvanced_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.3],
            'dilation_base': [2, 4, 8],
            'bottleneck_ratio': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        c = self.config or {}
        self.hidden_dim = c.get('hidden_dim', 512)
        self.n_layers = c.get('n_layers', 4)
        self.dropout_rate = c.get('dropout', 0.3)
        self.dilation_base = c.get('dilation_base', 4)
        self.bottleneck_ratio = c.get('bottleneck_ratio', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale dilated temporal convolutions
        self.dilated_convs = nn.ModuleList()
        for i in range(3):  # 3 scales
            dilation = self.dilation_base ** i
            # Ensure kernel size with dilation fits in input_len
            kernel_size = min(3, (input_len - 1) // max(dilation, 1) + 1)
            if kernel_size < 2:
                kernel_size = 2
            padding = (kernel_size - 1) * dilation // 2
            self.dilated_convs.append(
                nn.Conv1d(
                    self.n_features, 
                    self.n_features, 
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    groups=self.n_features  # channel-independent
                )
            )
        
        # Adaptive gating for multi-scale fusion
        self.scale_gate = nn.Linear(self.n_features * 3, self.n_features)
        
        # Bottleneck dimension
        self.bottleneck_dim = max(self.hidden_dim // self.bottleneck_ratio, 16)
        
        # MLP layers with bottleneck architecture
        self.mlp_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_gates = nn.ModuleList()
        
        in_dim = input_len
        for i in range(self.n_layers):
            if i == 0:
                layer_in = in_dim
                layer_out = self.hidden_dim
            elif i == self.n_layers - 1:
                layer_in = self.hidden_dim
                layer_out = output_len
            else:
                # Bottleneck: wide -> narrow -> wide
                if i % 2 == 1:
                    layer_in = self.hidden_dim
                    layer_out = self.bottleneck_dim
                else:
                    layer_in = self.bottleneck_dim
                    layer_out = self.hidden_dim
            
            self.mlp_layers.append(nn.Linear(layer_in, layer_out))
            
            # Layer norm on feature dimension (n_features is the feature dim after transpose)
            if i < self.n_layers - 1:
                self.norms.append(nn.LayerNorm(self.n_features))
            
            # Learned residual gate (only when dimensions match)
            if i > 0 and i < self.n_layers - 1 and layer_in == layer_out:
                self.residual_gates.append(nn.Parameter(torch.zeros(1)))
            else:
                self.residual_gates.append(None)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = nn.GELU()
        
        # Direct skip from input to output for stability
        self.skip_proj = nn.Linear(input_len, output_len) if input_len != output_len else nn.Identity()
        self.skip_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store for skip connection
        x_input = x  # (batch, input_len, n_features)
        
        # Multi-scale dilated convolutions
        x_conv = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        conv_outputs = []
        for conv in self.dilated_convs:
            out = conv(x_conv)  # (batch, n_features, input_len)
            # Trim or pad to match input_len
            if out.shape[-1] != x_conv.shape[-1]:
                if out.shape[-1] > x_conv.shape[-1]:
                    out = out[..., :x_conv.shape[-1]]
                else:
                    pad_len = x_conv.shape[-1] - out.shape[-1]
                    out = F.pad(out, (0, pad_len))
            conv_outputs.append(out.permute(0, 2, 1))  # (batch, input_len, n_features)
        
        # Concatenate scales and apply adaptive gating
        multi_scale = torch.cat(conv_outputs, dim=-1)  # (batch, input_len, n_features * 3)
        gate_weights = torch.sigmoid(self.scale_gate(multi_scale))  # (batch, input_len, n_features)
        
        # Weighted combination of scales
        x_scaled = conv_outputs[0] * gate_weights + conv_outputs[1] * (1 - gate_weights) * 0.5 + conv_outputs[2] * (1 - gate_weights) * 0.5
        # Refine: use the gated output to modulate original input
        x = x + x_scaled  # (batch, input_len, n_features)
        
        # Channel-independent MLP: process each feature separately
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        prev_x = None
        for i, layer in enumerate(self.mlp_layers):
            out = layer(x)  # (batch, n_features, layer_out)
            
            if i < self.n_layers - 1:
                out = self.activation(out)
                out = self.dropout(out)
                
                # Apply layer norm on the n_features dimension
                # Transpose to (batch, layer_out, n_features), normalize, transpose back
                out = self.norms[i](out.transpose(-1, -2)).transpose(-1, -2)  # (batch, n_features, layer_out)
                
                # Learned residual connection
                res_gate = self.residual_gates[i]
                if res_gate is not None and prev_x is not None and prev_x.shape == out.shape:
                    gate = torch.sigmoid(res_gate)
                    out = gate * out + (1 - gate) * prev_x  # (batch, n_features, layer_out)
                
                prev_x = out
            
            x = out  # (batch, n_features, layer_out)
        
        # x now: (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Add learned skip connection from input
        skip = self.skip_proj(x_input.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, output_len, n_features)
        skip_gate = torch.sigmoid(self.skip_gate)
        x = skip_gate * x + (1 - skip_gate) * skip  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)

