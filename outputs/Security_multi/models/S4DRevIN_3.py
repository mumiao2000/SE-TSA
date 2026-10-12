import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class S4DRevIN(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_state': [16, 64, 256],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'expansion_factor': [2, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_state = self.config.get('d_state', 64)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.expansion_factor = self.config.get('expansion_factor', 2)
        
        self.d_model = self.n_features * self.expansion_factor
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.d_model)
        
        # S4D layers
        self.s4d_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.s4d_layers.append(S4DLayer(self.d_model, self.d_state))
            self.norms.append(nn.LayerNorm(self.d_model))
            self.dropouts.append(nn.Dropout(self.dropout))
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.n_features)
        
        # Temporal decoder: project from input_len to output_len
        self.temporal_decoder = nn.Linear(input_len, output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, input_len, d_model)
        
        # Apply S4D layers
        for s4d, norm, drop in zip(self.s4d_layers, self.norms, self.dropouts):
            residual = x  # (batch, input_len, d_model)
            x = s4d(x)  # (batch, input_len, d_model)
            x = drop(x)  # (batch, input_len, d_model)
            x = norm(x + residual)  # (batch, input_len, d_model)
        
        # Project back to n_features
        x = self.output_proj(x)  # (batch, input_len, n_features)
        
        # Temporal decoding: (batch, n_features, input_len) -> (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.temporal_decoder(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class S4DLayer(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # S4D parameters: diagonal state matrix A, input B, output C
        # Initialize A with HiPPO-like decay (negative real parts)
        log_A_real = torch.log(0.5 * torch.ones(d_state))
        A_imag = torch.pi * torch.arange(d_state)
        
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)
        
        # B and C are complex for full expressiveness, but we use real for simplicity
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        
        # D skip connection
        self.D = nn.Parameter(torch.ones(d_model) * 0.01)
        
        # Discretization step
        self.log_dt = nn.Parameter(torch.log(torch.ones(d_model) * 0.01))
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Discretize: continuous to discrete
        dt = torch.exp(self.log_dt)  # (d_model,)
        A_real = -torch.exp(self.log_A_real)  # (d_state,)
        A = A_real + 1j * self.A_imag  # (d_state,)
        
        # Discrete state matrix: A_discrete = exp(dt * A)
        A_discrete = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))  # (d_model, d_state)
        
        # Discrete input matrix: B_discrete = dt * B (simplified Euler)
        B_discrete = dt.unsqueeze(-1) * self.B  # (d_model, d_state)
        
        # Recurrent scan (parallel scan for efficiency)
        # For simplicity, use sequential scan here
        state = torch.zeros(batch_size, d_model, self.d_state, device=x.device, dtype=torch.complex64)
        
        # Convert B_discrete to complex
        B_discrete_c = B_discrete.to(torch.complex64)
        C_c = self.C.to(torch.complex64)
        
        outputs = []
        for t in range(seq_len):
            u = x[:, t, :]  # (batch, d_model)
            
            # State update: state = A_discrete * state + B_discrete * u
            state = A_discrete.unsqueeze(0) * state + B_discrete_c.unsqueeze(0) * u.unsqueeze(-1).to(torch.complex64)
            
            # Output: y = Re(C * state) + D * u
            y = torch.real((C_c.unsqueeze(0) * state).sum(dim=-1)) + self.D.unsqueeze(0) * u
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return output
