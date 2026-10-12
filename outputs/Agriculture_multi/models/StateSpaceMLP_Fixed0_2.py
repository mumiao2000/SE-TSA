import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceMLP_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [32, 128, 512],
            'n_layers': [2, 4, 8],
            'emission_hidden': [64, 256, 1024],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 128)
        self.n_layers = self.config.get('n_layers', 4)
        self.emission_hidden = self.config.get('emission_hidden', 256)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input encoder: project features to state space
        self.input_encoder = nn.Sequential(
            nn.Linear(self.n_features, self.state_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.state_dim * 2, self.state_dim)
        )  # (batch, input_len, state_dim)
        
        # Structured state space layers
        self.ss_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.ss_layers.append(
                StructuredSSM(self.state_dim, self.dropout)
            )
        
        # Deep emission MLP
        self.emission = nn.ModuleList()
        emission_dims = [self.state_dim, self.emission_hidden, self.emission_hidden // 2, self.n_features]
        for i in range(len(emission_dims) - 1):
            self.emission.append(nn.Linear(emission_dims[i], emission_dims[i+1]))
            if i < len(emission_dims) - 2:
                self.emission.append(nn.GELU())
                self.emission.append(nn.Dropout(self.dropout))
        
        # Direct skip from input to output for autoregressive bias
        self.skip_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x_norm = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Encode to state space
        h = self.input_encoder(x_norm)  # (batch, input_len, state_dim)
        
        # Apply structured state space layers
        for layer in self.ss_layers:
            h = layer(h)  # (batch, input_len, state_dim)
        
        # Take final state
        h_final = h[:, -1, :]  # (batch, state_dim)
        
        # Generate future predictions autoregressively in latent space
        # Use state transition to propagate
        predictions = []
        h_t = h_final  # (batch, state_dim)
        for t in range(self.output_len):
            # Emit observation
            out_t = h_t  # (batch, state_dim)
            for layer in self.emission:
                out_t = layer(out_t)  # (batch, n_features)
            predictions.append(out_t)
            
            # Transition state (use last layer's dynamics)
            if self.n_layers > 0:
                h_t = self.ss_layers[-1].transition(h_t.unsqueeze(1)).squeeze(1)  # (batch, state_dim)
        
        out = torch.stack(predictions, dim=1)  # (batch, output_len, n_features)
        
        # Skip connection: repeat last input with slight trend
        last_val = x_norm[:, -1:, :].expand(-1, self.output_len, -1)  # (batch, output_len, n_features)
        out = out + self.skip_weight * last_val  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class StructuredSSM(nn.Module):
    def __init__(self, state_dim, dropout):
        super().__init__()
        self.state_dim = state_dim
        
        # Diagonal state transition (real poles)
        self.log_lambda_real = nn.Parameter(torch.randn(state_dim // 2) * -0.5)
        
        # Complex conjugate pairs
        self.n_complex = state_dim // 4
        self.log_lambda_complex_real = nn.Parameter(torch.randn(self.n_complex) * -0.5)
        self.lambda_complex_imag = nn.Parameter(torch.randn(self.n_complex) * 0.5)
        
        # Input-dependent state update
        self.B = nn.Linear(state_dim, state_dim)
        self.C = nn.Linear(state_dim, state_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)
        
    def transition(self, x):
        # x: (batch, seq_len, state_dim) or (batch, state_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        
        batch_size, seq_len, _ = x.shape
        
        # Build diagonal transition matrix
        lambda_real = torch.exp(self.log_lambda_real)  # (state_dim//2,)
        
        # Complex: |lambda| * exp(i*theta), but we parameterize differently
        # Use 2x2 blocks for complex conjugate pairs
        complex_real = torch.exp(self.log_lambda_complex_real) * torch.cos(self.lambda_complex_imag)
        complex_imag = torch.exp(self.log_lambda_complex_real) * torch.sin(self.lambda_complex_imag)
        
        # Apply transition (simplified: just scale for now, parallel scan would be better)
        # For efficiency, use cumulative product for diagonal
        h = x
        
        # Real part
        h_real = h[..., :len(lambda_real)] * lambda_real.view(1, 1, -1)
        
        # Complex part (interleaved)
        h_complex = h[..., len(lambda_real):len(lambda_real)+2*self.n_complex]
        h_complex = h_complex.reshape(batch_size, seq_len, self.n_complex, 2)
        
        # Rotation-scaling
        rot = torch.stack([
            torch.stack([complex_real, -complex_imag], dim=-1),
            torch.stack([complex_imag, complex_real], dim=-1)
        ], dim=-2)  # (n_complex, 2, 2)
        
        # Fix: expand dimensions for batch matrix multiplication
        rot = rot.unsqueeze(0).unsqueeze(0)  # (1, 1, n_complex, 2, 2)
        h_complex = h_complex.unsqueeze(-2)  # (batch, seq_len, n_complex, 1, 2)
        
        h_complex = torch.matmul(h_complex, rot.transpose(-2, -1))  # (batch, seq_len, n_complex, 1, 2)
        h_complex = h_complex.squeeze(-2)  # (batch, seq_len, n_complex, 2)
        h_complex = h_complex.reshape(batch_size, seq_len, -1)
        
        # Remaining (if odd)
        if h.shape[-1] > len(lambda_real) + 2*self.n_complex:
            h_rem = h[..., len(lambda_real)+2*self.n_complex:]
        else:
            h_rem = torch.empty(batch_size, seq_len, 0, device=h.device)
        
        h_out = torch.cat([h_real, h_complex, h_rem], dim=-1)
        
        if squeeze:
            h_out = h_out.squeeze(1)
        
        return h_out
        
    def forward(self, x):
        # x: (batch, seq_len, state_dim)
        # State transition
        h = self.transition(x)  # (batch, seq_len, state_dim)
        
        # Input modulation
        Bx = self.B(x)  # (batch, seq_len, state_dim)
        h = h + Bx  # (batch, seq_len, state_dim)
        
        # Output projection
        h = self.C(h)  # (batch, seq_len, state_dim)
        
        h = self.dropout(h)
        h = self.norm(h + x)  # residual  # (batch, seq_len, state_dim)
        
        return h
