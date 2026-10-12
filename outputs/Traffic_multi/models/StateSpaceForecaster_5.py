import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'd_model': [64, 256, 1024],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.2, 0.4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.d_model = self.config.get('d_model', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.d_model)  # (batch, seq_len, d_model)
        
        # State-space layers
        self.ss_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.ss_layers.append(nn.ModuleDict({
                'ssm': DiagonalSSM(self.d_model, self.state_dim),
                'gate': nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * 2),
                    nn.GLU(dim=-1)
                ),
                'norm': nn.LayerNorm(self.d_model),
                'ff': nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * 2),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.d_model * 2, self.d_model),
                    nn.Dropout(self.dropout_rate)
                ),
                'norm2': nn.LayerNorm(self.d_model)
            }))
        
        # Output: use last hidden state to predict future
        self.output_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model * 2, self.output_len * self.n_features)
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to d_model
        h = self.input_proj(x)  # (batch, input_len, d_model)
        h = self.dropout(h)  # (batch, input_len, d_model)
        
        # Apply state-space layers
        for layer in self.ss_layers:
            # SSM forward
            h_ssm = layer['ssm'](h)  # (batch, input_len, d_model)
            
            # Gating
            h_gate = layer['gate'](h_ssm)  # (batch, input_len, d_model)
            h = layer['norm'](h + h_gate)  # (batch, input_len, d_model)
            
            # FF
            h_ff = layer['ff'](h)  # (batch, input_len, d_model)
            h = layer['norm2'](h + h_ff)  # (batch, input_len, d_model)
        
        # Use last time step for prediction
        h_last = h[:, -1, :]  # (batch, d_model)
        
        # Generate output
        out = self.output_net(h_last)  # (batch, output_len * n_features)
        out = out.view(-1, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class DiagonalSSM(nn.Module):
    def __init__(self, d_model, state_dim):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        # Diagonal transition matrix parameters (log for stability)
        self.log_A_real = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.A_imag = nn.Parameter(torch.randn(state_dim) * 0.01)
        
        # Input-to-state matrix
        self.B = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
        
        # State-to-output matrix
        self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
        
        # Skip connection
        self.D = nn.Parameter(torch.randn(d_model) * 0.01)
        
        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape
        
        # Project input
        u = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Compute A = exp(log_A_real + i * A_imag)
        A_real = torch.exp(self.log_A_real)  # (state_dim,)
        A = torch.complex(A_real, self.A_imag)  # (state_dim,)
        
        # Discretization: simple Euler with step size 1
        # A_discrete = exp(A), B_discrete = B
        # For diagonal: just element-wise
        
        # Parallel scan / convolution approach for efficiency
        # For simplicity, use sequential scan (can be optimized)
        
        # Materialize A powers for convolution kernel
        # K_t = C^T * A^t * B for t = 0, ..., seq_len-1
        
        # Compute the kernel in frequency domain for efficiency
        # Or use simple sequential computation
        
        # Simplified: use cumsum-based approach with decay
        # h_t = A * h_{t-1} + B * u_t
        # y_t = C * h_t + D * u_t
        
        # For diagonal A, this is element-wise
        # We'll use a simple RNN-like scan
        
        h = torch.zeros(batch, self.state_dim, device=x.device, dtype=torch.cfloat)
        outputs = []
        
        for t in range(seq_len):
            # h = A * h + B^T * u_t
            # B^T * u_t: (state_dim, d_model) @ (d_model,) -> (state_dim,)
            Bu = torch.matmul(u[:, t, :], self.B)  # (batch, state_dim)
            Bu = torch.complex(Bu, torch.zeros_like(Bu))
            
            h = A.unsqueeze(0) * h + Bu  # (batch, state_dim)
            
            # y = C^T * h + D * u
            # C^T * h: (d_model, state_dim) @ (state_dim,) -> (d_model,)
            y = torch.matmul(h.real, self.C.T)  # (batch, d_model)
            y = y + self.D.unsqueeze(0) * u[:, t, :]  # (batch, d_model)
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return y
