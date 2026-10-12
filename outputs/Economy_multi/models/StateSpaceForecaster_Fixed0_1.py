import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 2048],
            'n_layers': [1, 4, 8],
            'state_dim': [16, 64, 256],
            'dropout': [0.0, 0.3],
            'expand_factor': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 512)
        self.n_layers = cfg.get('n_layers', 4)
        self.state_dim = cfg.get('state_dim', 64)
        self.dropout = cfg.get('dropout', 0.1)
        self.expand_factor = cfg.get('expand_factor', 4)
        
        self.d_inner = self.d_model * self.expand_factor
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.d_model)
        
        # SSM layers
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(SSMLayer(self.d_model, self.d_inner, self.state_dim, self.dropout))
        
        # Output projection
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Linear(self.d_model, self.n_features)
        
        # Direct skip from input to output for residual learning
        self.skip_proj = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch = x.shape[0]  # (batch, input_len, n_features)
        
        # Store input for skip connection
        x_input = x  # (batch, input_len, n_features)
        
        # RevIN normalize
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project input
        x = self.input_proj(x)  # (batch, input_len, d_model)
        
        # Apply SSM layers with residual connections
        for layer in self.layers:
            x = x + layer(x)  # (batch, input_len, d_model)
        
        # Final normalization
        x = self.output_norm(x)  # (batch, input_len, d_model)
        
        # Project to output features
        x = self.output_proj(x)  # (batch, input_len, n_features)
        
        # Add skip connection: project input_len to output_len
        x_skip = x_input.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_skip = self.skip_proj(x_skip)  # (batch, n_features, output_len)
        x_skip = x_skip.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Use last values with learned correction
        x_last = x[:, -1:, :]  # (batch, 1, n_features)
        x_pred = x_last.repeat(1, self.output_len, 1)  # (batch, output_len, n_features)
        
        # Add the model's prediction (residual)
        x_pred = x_pred + x_skip  # (batch, output_len, n_features)
        
        # Denormalize back
        x_pred = self.revin(x_pred, mode='denorm')  # (batch, output_len, n_features)
        
        return x_pred


class SSMLayer(nn.Module):
    def __init__(self, d_model, d_inner, state_dim, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.state_dim = state_dim
        
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)  # x and gate
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)
        self.A_log = nn.Parameter(torch.log(torch.ones(d_inner, state_dim)))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.x_proj = nn.Linear(d_inner, state_dim * 2)  # B and C
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape  # (batch, seq_len, d_model)
        
        x = self.norm(x)  # (batch, seq_len, d_model)
        
        # Project to inner dim with gating
        x_and_gate = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_inner, gate = x_and_gate.chunk(2, dim=-1)  # each (batch, seq_len, d_inner)
        gate = F.silu(gate)  # (batch, seq_len, d_inner)
        
        # Causal convolution
        x_conv = x_inner.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv(x_conv)  # (batch, d_inner, seq_len)
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)  # (batch, seq_len, d_inner)
        
        # Selective parameters
        A = -torch.exp(self.A_log)  # (d_inner, state_dim)
        
        # Discretization
        dt = F.softplus(self.dt_proj(x_conv))  # (batch, seq_len, d_inner)
        
        # Get B, C from input
        BC = self.x_proj(x_conv)  # (batch, seq_len, state_dim * 2)
        B, C = BC.chunk(2, dim=-1)  # each (batch, seq_len, state_dim)
        
        # Simplified selective scan
        dtA = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (batch, seq_len, d_inner, state_dim)
        
        # State evolution with selective input
        h = torch.zeros(batch, self.d_inner, self.state_dim, device=x.device)  # (batch, d_inner, state_dim)
        ys = []
        
        for t in range(seq_len):
            # Discretized state update
            dA = torch.exp(dtA[:, t])  # (batch, d_inner, state_dim)
            dB = dt[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1)  # (batch, d_inner, state_dim)
            
            h = dA * h + dB * x_conv[:, t].unsqueeze(-1)  # (batch, d_inner, state_dim)
            y = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)  # (batch, d_inner)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # (batch, seq_len, d_inner)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv  # (batch, seq_len, d_inner)
        
        # Gating
        y = y * gate  # (batch, seq_len, d_inner)
        
        # Output projection
        y = self.out_proj(y)  # (batch, seq_len, d_model)
        y = self.dropout(y)  # (batch, seq_len, d_model)
        
        return y
