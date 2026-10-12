import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceForecaster_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [32, 128, 512],
            'n_layers': [1, 3, 6],
            'dt_rank': [4, 16]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.state_dim = self.config.get('state_dim', 128)
        self.n_layers = self.config.get('n_layers', 3)
        self.dt_rank = self.config.get('dt_rank', 16)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.state_dim)
        
        # SSM layers - each layer is a proper nn.Module
        self.ssm_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer = SSMLayer(self.state_dim, self.n_features)
            self.ssm_layers.append(layer)
        
        # Output head: state -> output_len steps
        self.output_head = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim * 2),
            nn.GELU(),
            nn.Linear(self.state_dim * 2, self.output_len * self.n_features)
        )
        
        self.revin = RevIN(self.n_features, affine=True)

    def _ssm_step(self, x_t, h, A, B, dt):
        # x_t: (batch, state_dim), h: (batch, state_dim), B: (batch, state_dim), dt: (batch, state_dim)
        # Simple Euler discretization
        dh = dt * (h @ A.T + B * x_t)  # (batch, state_dim)
        h_new = h + dh  # (batch, state_dim)
        return h_new

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project input
        u = self.input_proj(x)  # (batch, input_len, state_dim)
        
        # Process through SSM layers
        h = torch.zeros(x.size(0), self.state_dim, device=x.device)  # (batch, state_dim)
        
        for layer in self.ssm_layers:
            h_layer = h.clone()  # (batch, state_dim)
            outputs = []
            for t in range(self.input_len):
                x_t = u[:, t, :]  # (batch, state_dim)
                # Input-dependent B and dt (use raw input x, not projected u)
                B = layer.B_proj(x[:, t, :])  # (batch, state_dim)
                dt = F.softplus(layer.dt_proj(x[:, t, :]))  # (batch, state_dim)
                
                # Discrete step
                h_layer = self._ssm_step(x_t, h_layer, layer.A, B, dt)  # (batch, state_dim)
                h_layer = layer.norm(h_layer)  # (batch, state_dim)
                
                # Gating
                gate = torch.sigmoid(layer.gate(h_layer))  # (batch, state_dim)
                h_layer = h_layer * gate  # (batch, state_dim)
                
                outputs.append(h_layer)
            
            # Residual update
            h = h + outputs[-1]  # (batch, state_dim)
        
        # Final state to output
        out = self.output_head(h)  # (batch, output_len * n_features)
        out = out.reshape(x.size(0), self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class SSMLayer(nn.Module):
    def __init__(self, state_dim, n_features):
        super().__init__()
        self.state_dim = state_dim
        self.n_features = n_features
        
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B_proj = nn.Linear(n_features, state_dim)
        self.C_proj = nn.Linear(state_dim, n_features)
        self.dt_proj = nn.Linear(n_features, state_dim)
        self.norm = nn.LayerNorm(state_dim)
        self.gate = nn.Linear(state_dim, state_dim)
        
        # Initialize A as diagonal dominant (stable)
        with torch.no_grad():
            self.A.data = -torch.abs(self.A.data) - 0.5
            self.A.data.fill_diagonal_(-1.0)

