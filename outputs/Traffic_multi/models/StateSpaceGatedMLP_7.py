import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceGatedMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_state': [16, 64, 256],
            'd_inner': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'dt_rank': [4, 16, 64]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_state = cfg.get('d_state', 64)
        self.d_inner = cfg.get('d_inner', 256)
        self.n_layers = cfg.get('n_layers', 4)
        self.dropout = cfg.get('dropout', 0.1)
        self.dt_rank = cfg.get('dt_rank', 16)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection per channel
        self.input_proj = nn.Linear(input_len, self.d_inner)
        
        # SSM layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(SSMBlock(self.d_inner, self.d_state, self.dt_rank, self.dropout))
            self.norms.append(nn.LayerNorm(self.d_inner))
            
        # Output projection
        self.output_proj = nn.Linear(self.d_inner, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-independent processing
        batch_size = x.shape[0]
        
        # Reshape: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Process each feature independently but in parallel
        # Apply input projection
        x = self.input_proj(x)  # (batch, n_features, d_inner)
        
        # Apply SSM layers with residual connections
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            residual = x
            x = layer(x)  # (batch, n_features, d_inner)
            x = norm(x)
            x = x + residual
            
        # Output projection
        x = self.output_proj(x)  # (batch, n_features, output_len)
        
        # Reshape back
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class SSMBlock(nn.Module):
    def __init__(self, d_inner, d_state, dt_rank, dropout):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        
        # Input-dependent step size
        self.dt_proj = nn.Linear(d_inner, dt_rank)
        self.dt_proj2 = nn.Linear(dt_rank, d_inner)
        
        # State space parameters (diagonal)
        # A: negative real for stability
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_inner + 1)).unsqueeze(-1).repeat(1, d_state))
        
        # B, C, D parameters
        self.B = nn.Linear(d_inner, d_state, bias=False)
        self.C = nn.Linear(d_inner, d_state, bias=False)
        self.D = nn.Parameter(torch.randn(d_inner))
        
        # Gating
        self.gate = nn.Linear(d_inner, d_inner)
        self.out_proj = nn.Linear(d_inner, d_inner)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, n_features, d_inner)
        batch_size, n_features, d_inner = x.shape
        
        # Flatten batch and features for sequential processing
        x_flat = x.reshape(batch_size * n_features, d_inner)  # (batch*n_features, d_inner)
        
        # Compute input-dependent step size
        delta = F.softplus(self.dt_proj2(F.gelu(self.dt_proj(x_flat))))  # (batch*n_features, d_inner)
        
        # Discretize: simple Euler
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Compute B and C for each input
        B = self.B(x_flat)  # (batch*n_features, d_state)
        C = self.C(x_flat)  # (batch*n_features, d_state)
        
        # Simplified SSM scan (parallel associative scan would be better but complex)
        # We'll use a simple recurrent approach for clarity
        # State: (batch*n_features, d_state)
        h = torch.zeros(batch_size * n_features, self.d_state, device=x.device, dtype=x.dtype)
        
        # For efficiency, we process all at once with matrix operations
        # This is a simplified linear attention-like formulation
        # y = C @ (I - exp(delta A))^{-1} @ B @ x + D @ x
        
        # Approximate with: y ≈ sum over states
        # Using the parallel scan formulation
        A_discrete = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))  # (batch*n_features, d_inner, d_state)
        
        # Simplified: use cumulative product approximation
        # For true S4, we'd need careful discretization; here we use a gated approximation
        
        # Gated MLP-style computation with state-space flavor
        gate = torch.sigmoid(self.gate(x_flat))  # (batch*n_features, d_inner)
        
        # State contribution
        state_contrib = torch.bmm(B.unsqueeze(1), A_discrete.transpose(1, 2)).squeeze(1)  # (batch*n_features, d_inner)
        state_contrib = state_contrib * C.sum(dim=-1, keepdim=True)  # simplified
        
        # Combine
        y = gate * (x_flat + state_contrib * 0.1)  # gated residual with small state contribution
        y = y * self.D  # skip connection
        
        y = self.out_proj(y)  # (batch*n_features, d_inner)
        y = self.dropout(y)
        
        # Reshape back
        y = y.reshape(batch_size, n_features, d_inner)  # (batch, n_features, d_inner)
        return y

