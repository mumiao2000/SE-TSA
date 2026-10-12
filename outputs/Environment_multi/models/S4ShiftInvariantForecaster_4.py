import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class S4ShiftInvariantForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [64, 256, 1024],
            'n_layers': [2, 4, 8],
            'd_model': [64, 256, 1024],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.d_model = self.config.get('d_model', 256)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.d_model)
        
        # S4 layers with multi-scale gating
        self.s4_layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(self.n_layers):
            self.s4_layers.append(
                S4Layer(self.d_model, self.state_dim, dropout=self.dropout)
            )
            # Multi-scale gating: different dilations for different layers
            dilation = 2 ** (i % 3)  # 1, 2, 4, 1, 2, 4, ...
            self.gates.append(
                MultiScaleGate(self.d_model, dilation)
            )
            self.norms.append(nn.LayerNorm(self.d_model))
        
        # Output: predict future from hidden states
        self.output_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.output_len * self.n_features)
        )
        
        # Direct skip from input
        self.skip_proj = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save for skip connection
        x_skip = x  # (batch, input_len, n_features)
        
        # Project to d_model
        h = self.input_proj(x)  # (batch, input_len, d_model)
        
        # Apply S4 layers with multi-scale gating
        for i in range(self.n_layers):
            # S4 transformation
            h_new = self.s4_layers[i](h)  # (batch, input_len, d_model)
            # Multi-scale gating
            h_new = self.gates[i](h_new)  # (batch, input_len, d_model)
            # Residual + norm
            h = self.norms[i](h + h_new)  # (batch, input_len, d_model)
        
        # Pool across time with attention
        attn = torch.softmax(h.mean(dim=-1, keepdim=True), dim=1)  # (batch, input_len, 1)
        h_pooled = (h * attn).sum(dim=1)  # (batch, d_model)
        
        # Generate output
        out = self.output_net(h_pooled)  # (batch, output_len * n_features)
        out = out.reshape(out.shape[0], self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Add skip connection from input
        x_skip_perm = x_skip.permute(0, 2, 1)  # (batch, n_features, input_len)
        skip_out = self.skip_proj(x_skip_perm)  # (batch, n_features, output_len)
        skip_out = skip_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = out + skip_out * 0.1  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)


class S4Layer(nn.Module):
    def __init__(self, d_model, state_dim, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.dropout = dropout
        
        # Simplified diagonal S4: state matrix A is diagonal
        # Initialize with exponential decay
        self.log_A_real = nn.Parameter(torch.randn(state_dim) * 0.01 - 0.5)  # negative values for stability
        self.A_imag = nn.Parameter(torch.randn(state_dim) * 0.1)
        
        # Input-to-state
        self.B = nn.Linear(d_model, state_dim, bias=False)
        
        # State-to-output
        self.C = nn.Linear(state_dim, d_model, bias=False)
        
        # D skip connection
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Discretize: simple Euler with fixed step
        # A = exp(log_A_real + i * A_imag)
        A = torch.exp(self.log_A_real) * torch.exp(1j * self.A_imag)  # (state_dim,) complex
        
        # Compute state sequence using parallel scan (simplified)
        # For efficiency, use convolutional view for diagonal case
        Bx = self.B(x)  # (batch, seq_len, state_dim)
        
        # Manual RNN-like scan (can be optimized)
        states = []
        state = torch.zeros(batch_size, self.state_dim, dtype=torch.complex64, device=x.device)
        
        # Make A complex for computation
        A_complex = A.to(torch.complex64)  # (state_dim,)
        
        for t in range(seq_len):
            # state = A * state + Bx[t]
            state = A_complex * state + Bx[:, t, :].to(torch.complex64)  # (batch, state_dim)
            states.append(state)
        
        states = torch.stack(states, dim=1)  # (batch, seq_len, state_dim), complex
        
        # Project to output
        out = self.C(states.real)  # (batch, seq_len, d_model)
        
        # Add skip connection
        out = out + x * self.D.unsqueeze(0).unsqueeze(0)  # (batch, seq_len, d_model)
        
        out = F.dropout(out, p=self.dropout, training=self.training)  # (batch, seq_len, d_model)
        
        return out  # (batch, seq_len, d_model)


class MultiScaleGate(nn.Module):
    def __init__(self, d_model, dilation):
        super().__init__()
        self.dilation = dilation
        
        # Gating mechanism
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1)
        )
        
        # Dilated convolution for multi-scale
        if dilation > 1:
            self.dilated_conv = nn.Conv1d(
                d_model, d_model, 
                kernel_size=3, 
                padding=dilation, 
                dilation=dilation
            )
        else:
            self.dilated_conv = None
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        
        # Apply gating
        gate = self.gate_proj(x)  # (batch, seq_len, d_model)
        
        # Apply dilated convolution if applicable
        if self.dilated_conv is not None:
            x_perm = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
            x_conv = self.dilated_conv(x_perm)  # (batch, d_model, seq_len)
            x_conv = x_conv.permute(0, 2, 1)  # (batch, seq_len, d_model)
            # Truncate to match length
            x_conv = x_conv[:, :x.shape[1], :]  # (batch, seq_len, d_model)
            x = x + x_conv  # (batch, seq_len, d_model)
        
        return x * gate  # (batch, seq_len, d_model)
