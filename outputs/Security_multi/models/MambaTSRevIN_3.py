import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class MambaTSRevIN(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_state': [16, 64, 256],
            'd_conv': [2, 4, 8],
            'expand_factor': [2, 4],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_state = self.config.get('d_state', 64)
        self.d_conv = self.config.get('d_conv', 4)
        self.expand_factor = self.config.get('expand_factor', 2)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.d_inner = self.n_features * self.expand_factor
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input expansion
        self.in_proj = nn.Linear(self.n_features, self.d_inner * 2)
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(self.d_inner, self.d_state, self.d_conv, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.n_features)
        
        # Temporal decoder
        self.temporal = nn.Linear(input_len, output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Expand
        x = self.in_proj(x)  # (batch, input_len, d_inner * 2)
        x, res = x.chunk(2, dim=-1)  # each (batch, input_len, d_inner)
        
        # Apply layers
        for layer in self.layers:
            x = layer(x)  # (batch, input_len, d_inner)
        
        # Gating
        x = x * F.silu(res)  # (batch, input_len, d_inner)
        
        # Project to features
        x = self.out_proj(x)  # (batch, input_len, n_features)
        
        # Temporal decode
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.temporal(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class MambaBlock(nn.Module):
    def __init__(self, d_inner, d_state, d_conv, dropout):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        
        # Conv1d for local context
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, 
            kernel_size=d_conv, 
            padding=d_conv - 1,
            groups=d_inner
        )
        
        # SSM parameters (selective)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1)  # for B, C, delta
        self.dt_proj = nn.Linear(1, d_inner)
        
        # State space parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float().repeat(d_inner, 1)))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        self.norm = nn.LayerNorm(d_inner)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, d_inner)
        batch, seq_len, _ = x.shape
        
        residual = x
        x = self.norm(x)
        
        # Conv1d
        x_conv = x.permute(0, 2, 1)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # (batch, d_inner, seq_len)
        x_conv = x_conv.permute(0, 2, 1)  # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)
        
        # Selective SSM
        x_ssm = self.ssm(x_conv)  # (batch, seq_len, d_inner)
        
        x = self.dropout(x_ssm) + residual
        return x
    
    def ssm(self, x):
        # Simplified selective scan
        batch, seq_len, d_inner = x.shape
        
        # Get selective parameters
        x_proj = self.x_proj(x)  # (batch, seq_len, d_state*2 + 1)
        delta, B, C = x_proj.split([1, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))  # (batch, seq_len, d_inner)
        
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Discretize and scan (simplified sequential version)
        state = torch.zeros(batch, d_inner, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            dt_t = delta[:, t, :]  # (batch, d_inner)
            u_t = x[:, t, :]  # (batch, d_inner)
            B_t = B[:, t, :]  # (batch, d_state)
            C_t = C[:, t, :]  # (batch, d_state)
            
            # Discretize A
            dA = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))  # (batch, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (batch, d_inner, d_state)
            
            # Update state
            state = dA * state + dB * u_t.unsqueeze(-1)
            
            # Output
            y = (state * C_t.unsqueeze(1)).sum(dim=-1) + self.D.unsqueeze(0) * u_t
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
