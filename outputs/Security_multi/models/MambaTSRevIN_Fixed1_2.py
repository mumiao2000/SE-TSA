import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class MambaTSRevIN_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'd_state': [16, 64, 256],
            'expand': [1, 2, 4],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.d_state = self.config.get('d_state', 64)
        self.expand = self.config.get('expand', 2)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.d_inner = self.d_model * self.expand
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input embedding
        self.in_proj = nn.Linear(self.n_features, self.d_model)
        
        # Mamba-inspired blocks
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer = nn.ModuleDict({
                'x_proj': nn.Linear(self.d_model, self.d_inner + self.d_state * 2),  # delta: d_inner, B: d_state, C: d_state
                'dt_proj': nn.Linear(self.d_inner, self.d_inner),
                'in_proj_inner': nn.Linear(self.d_model, self.d_inner),  # project to d_inner for conv
                'conv1d': nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3, padding=2, groups=self.d_inner, dilation=1),
                'act': nn.SiLU(),
                'out_proj': nn.Linear(self.d_inner, self.d_model),
                'norm': nn.LayerNorm(self.d_model),
                'drop': nn.Dropout(self.dropout)
            })
            self.layers.append(layer)
            # Register A_log and D as parameters on the module, not inside ModuleDict
            self.register_parameter(f'A_log_{len(self.layers)-1}', nn.Parameter(torch.log(torch.arange(1, self.d_inner + 1)).repeat(self.d_state, 1).T))
            self.register_parameter(f'D_{len(self.layers)-1}', nn.Parameter(torch.ones(self.d_inner)))
        
        # Output head
        self.out_norm = nn.LayerNorm(self.d_model)
        self.forecast = nn.Linear(self.d_model, self.output_len * self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Embed
        x = self.in_proj(x)  # (batch, input_len, d_model)
        
        for i, layer in enumerate(self.layers):
            residual = x  # (batch, input_len, d_model)
            x = layer['norm'](x)  # (batch, input_len, d_model)
            
            # Get parameters for this layer
            A_log = getattr(self, f'A_log_{i}')  # (d_inner, d_state)
            D = getattr(self, f'D_{i}')  # (d_inner,)
            
            # Selective parameters: x_proj outputs [d_inner + d_state * 2]
            x_proj = layer['x_proj'](x)  # (batch, input_len, d_inner + d_state * 2)
            delta, B, C = torch.split(x_proj, [self.d_inner, self.d_state, self.d_state], dim=-1)
            # delta: (batch, input_len, d_inner), B: (batch, input_len, d_state), C: (batch, input_len, d_state)
            
            delta = F.softplus(layer['dt_proj'](delta))  # (batch, input_len, d_inner)
            
            # Discretize A
            A = -torch.exp(A_log)  # (d_inner, d_state)
            
            # Project to inner dimension for convolution
            x_inner = layer['in_proj_inner'](x)  # (batch, input_len, d_inner)
            
            # Conv1d on inner dimension
            x_conv = x_inner.transpose(1, 2)  # (batch, d_inner, input_len)
            x_conv = layer['conv1d'](x_conv)  # (batch, d_inner, input_len + 2 due to padding=2)
            x_conv = x_conv[:, :, :self.input_len]  # trim to original length  # (batch, d_inner, input_len)
            x_conv = x_conv.transpose(1, 2)  # (batch, input_len, d_inner)
            x_conv = layer['act'](x_conv)  # (batch, input_len, d_inner)
            
            # SSM approximation: simple state transition with cumulative sum
            # Discretized state space model
            deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (batch, input_len, d_inner, d_state)
            deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * x_conv.unsqueeze(-1)  # (batch, input_len, d_inner, d_state)
            
            # Causal scan with associative scan (cumsum approximation)
            h = torch.zeros(x.shape[0], self.d_inner, self.d_state, device=x.device)  # (batch, d_inner, d_state)
            outputs = []
            for t in range(self.input_len):
                h = deltaA[:, t] * h + deltaB_u[:, t]  # (batch, d_inner, d_state)
                out_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (batch, d_inner)
                outputs.append(out_t)
            
            x_ssm = torch.stack(outputs, dim=1)  # (batch, input_len, d_inner)
            x_ssm = x_ssm * D.unsqueeze(0).unsqueeze(0)  # (batch, input_len, d_inner)
            x_ssm = x_ssm + x_conv  # skip connection  # (batch, input_len, d_inner)
            
            # Output projection
            x = layer['out_proj'](x_ssm)  # (batch, input_len, d_model)
            x = layer['drop'](x)  # (batch, input_len, d_model)
            x = x + residual  # (batch, input_len, d_model)
        
        # Forecast from last hidden state
        x = self.out_norm(x[:, -1, :])  # (batch, d_model)
        out = self.forecast(x)  # (batch, output_len * n_features)
        out = out.view(-1, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
