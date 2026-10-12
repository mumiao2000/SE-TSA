import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [64, 256, 1024],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3],
            'expansion_factor': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 256)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        self.expansion_factor = self.config.get('expansion_factor', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.state_dim)
        
        # Deep SSM layers
        self.ssm_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.ssm_layers.append(
                SSMBlock(
                    state_dim=self.state_dim,
                    expansion_factor=self.expansion_factor,
                    dropout=self.dropout
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(self.state_dim, self.n_features)
        
        # Temporal decoder: from input_len to output_len
        self.temporal_decoder = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to state dimension
        x = self.input_proj(x)  # (batch, input_len, state_dim)
        
        # Apply SSM layers
        for layer in self.ssm_layers:
            x = layer(x)  # (batch, input_len, state_dim)
        
        # Output projection per time step
        x = self.output_proj(x)  # (batch, input_len, n_features)
        
        # Temporal decoder: predict future steps
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.temporal_decoder(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class SSMBlock(nn.Module):
    def __init__(self, state_dim, expansion_factor, dropout):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = state_dim * expansion_factor
        
        # Layer norms
        self.norm1 = nn.LayerNorm(state_dim)
        self.norm2 = nn.LayerNorm(state_dim)
        
        # SSM parameters: diagonal state matrix A, input matrix B, output matrix C
        # Initialize A with exponential decay (real part negative)
        self.log_A_real = nn.Parameter(torch.randn(state_dim) * 0.01 - 0.5)
        self.A_imag = nn.Parameter(torch.randn(state_dim) * 0.01)
        
        # Input-dependent B and C (liquid SSM / S5 style)
        self.B_proj = nn.Linear(state_dim, state_dim)
        self.C_proj = nn.Linear(state_dim, state_dim)
        
        # D skip connection
        self.D = nn.Parameter(torch.randn(state_dim) * 0.01)
        
        # MLP for mixing
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, state_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, state_dim)
        batch_size, seq_len, _ = x.shape
        
        # First sublayer: SSM
        residual = x
        x = self.norm1(x)  # (batch, seq_len, state_dim)
        
        # Compute input-dependent B and C
        B = self.B_proj(x)  # (batch, seq_len, state_dim)
        C = self.C_proj(x)  # (batch, seq_len, state_dim)
        
        # Discretize A
        dt = 0.01  # step size
        A_real = -torch.exp(self.log_A_real)  # negative for stability
        A_discrete = torch.exp((A_real + 1j * self.A_imag) * dt)  # (state_dim,)
        
        # Simplified SSM computation: use real part only for stability
        # Compute the state evolution using parallel scan approximation
        # For simplicity, use cumulative sum approximation
        
        # Discretize B
        B_discrete = B * dt  # (batch, seq_len, state_dim)
        
        # State evolution via cumsum (simplified linear attention / cumsum SSM)
        # h_t = A * h_{t-1} + B * x_t
        # For diagonal A, this is element-wise
        
        # Use cumulative product for A and cumulative sum for input
        # Simplified: use exponential moving average style
        
        A_real_discrete = A_real * dt  # (state_dim,)
        A_factor = torch.exp(A_real_discrete)  # (state_dim,)
        
        # Manual scan
        h = torch.zeros(batch_size, self.state_dim, device=x.device)  # (batch, state_dim)
        outputs = []
        
        for t in range(seq_len):
            h = A_factor * h + B_discrete[:, t, :]  # (batch, state_dim)
            y_t = C[:, t, :] * h  # (batch, state_dim)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, state_dim)
        
        # Skip connection with D
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x  # (batch, seq_len, state_dim)
        
        x = y + residual  # (batch, seq_len, state_dim)
        
        # Second sublayer: MLP
        residual = x
        x = self.norm2(x)  # (batch, seq_len, state_dim)
        x = self.mlp(x)  # (batch, seq_len, state_dim)
        x = x + residual  # (batch, seq_len, state_dim)
        
        return x
