import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class S4ShiftRevIN_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'shift_factor': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.shift_factor = self.config.get('shift_factor', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Time shift: create multiple temporal views
        self.max_shift = max(1, self.input_len // self.shift_factor)
        self.n_shifts = min(self.shift_factor, 4)  # cap at 4 shifts
        
        # S4-inspired layers: diagonal state with skip connections
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer = S4Layer(self.n_features, self.state_dim, self.dropout)
            self.layers.append(layer)
        
        # Output: predict from final state
        self.out_proj = nn.Sequential(
            nn.Linear(self.state_dim * self.n_shifts, 512),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.output_len * self.n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Collect shifted versions
        shift_states = []
        for s in range(self.n_shifts):
            shift = s * self.max_shift // max(1, self.n_shifts - 1) if self.n_shifts > 1 else 0
            if shift > 0:
                x_shift = F.pad(x[:, shift:, :], (0, 0, shift, 0), value=0)  # (batch, input_len, n_features)
            else:
                x_shift = x  # (batch, input_len, n_features)
            
            # Initialize state
            h = torch.zeros(batch_size, self.state_dim, device=x.device)  # (batch, state_dim)
            
            # Recurrent scan
            for t in range(self.input_len):
                xt = x_shift[:, t, :]  # (batch, n_features)
                xt_prev = None
                for layer in self.layers:
                    h, xt_prev = layer(h, xt, xt_prev)  # (batch, state_dim), (batch, state_dim)
            
            shift_states.append(h)  # (batch, state_dim)
        
        # Combine all shift states
        combined = torch.cat(shift_states, dim=-1)  # (batch, state_dim * n_shifts)
        out = self.out_proj(combined)  # (batch, output_len * n_features)
        out = out.view(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class S4Layer(nn.Module):
    def __init__(self, n_features, state_dim, dropout):
        super().__init__()
        self.n_features = n_features
        self.state_dim = state_dim
        self.A_real = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.A_imag = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.B = nn.Linear(n_features, state_dim)
        self.C = nn.Linear(state_dim, state_dim)
        self.D = nn.Parameter(torch.randn(state_dim))
        self.norm = nn.LayerNorm(state_dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, h, xt_orig, xt_prev=None):
        # h: (batch, state_dim), xt_orig: (batch, n_features), xt_prev: (batch, state_dim) or None
        B_u = self.B(xt_orig)  # (batch, state_dim)
        # A = -exp(A_real) + i*A_imag for stability
        A_diag = -torch.exp(self.A_real) + 1j * self.A_imag  # (state_dim,)
        # Simple Euler: h = h + dt * (A*h + B*u), with dt=1/input_len
        dt = 1.0 / 1000.0  # Use a small fixed dt for stability
        h = h + dt * (h * A_diag.real + B_u)  # (batch, state_dim)
        h = self.norm(h.real)  # (batch, state_dim)
        h = self.drop(h)  # (batch, state_dim)
        # Skip connection
        h = h + self.D * B_u.real  # (batch, state_dim)
        # C transforms state to state_dim for next layer's feedback
        xt_out = self.C(h).real  # (batch, state_dim)
        return h, xt_out
