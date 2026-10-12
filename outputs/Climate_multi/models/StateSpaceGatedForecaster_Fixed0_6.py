import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceGatedForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'n_resolutions': [2, 4, 8],
            'gate_dim': [16, 32, 64],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.n_resolutions = self.config.get('n_resolutions', 4)
        self.gate_dim = self.config.get('gate_dim', 32)
        self.dropout = self.config.get('dropout', 0.3)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to state space
        self.input_proj = nn.Linear(self.n_features, self.state_dim)
        
        # Multi-resolution state space layers
        self.resolutions = nn.ModuleList()
        for r in range(self.n_resolutions):
            scale = 2 ** r
            self.resolutions.append(StateSpaceResolution(
                state_dim=self.state_dim,
                scale=scale,
                input_len=self.input_len,
                output_len=self.output_len
            ))
        
        # Channel-adaptive gating
        self.gate_proj = nn.Sequential(
            nn.Linear(self.state_dim * self.n_resolutions, self.gate_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.gate_dim, self.n_features),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.state_dim * self.n_resolutions, 512),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, self.n_features)
        )
        
        # Direct residual from input
        self.residual_fc = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save for residual
        x_orig = x  # (batch, input_len, n_features)
        
        # Project to state space
        h = self.input_proj(x)  # (batch, input_len, state_dim)
        
        # Multi-resolution state space processing
        resolution_outputs = []
        for res in self.resolutions:
            h_res = res(h)  # (batch, output_len, state_dim)
            resolution_outputs.append(h_res)  # (batch, output_len, state_dim)
        
        # Concatenate all resolutions
        h_multi = torch.cat(resolution_outputs, dim=-1)  # (batch, output_len, state_dim * n_resolutions)
        
        # Channel-adaptive gating
        gate = self.gate_proj(h_multi)  # (batch, output_len, n_features)
        
        # Base prediction
        out = self.output_proj(h_multi)  # (batch, output_len, n_features)
        
        # Apply channel gating
        out = out * gate  # (batch, output_len, n_features)
        
        # Add direct residual
        x_res = x_orig.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_res = self.residual_fc(x_res)  # (batch, n_features, output_len)
        x_res = x_res.permute(0, 2, 1)  # (batch, output_len, n_features)
        out = out + x_res * 0.1  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)


class StateSpaceResolution(nn.Module):
    def __init__(self, state_dim, scale, input_len, output_len):
        super().__init__()
        self.state_dim = state_dim
        self.scale = scale
        self.input_len = input_len
        self.output_len = output_len
        
        # State transition with learnable A matrix (diagonal for stability)
        self.A_log = nn.Parameter(torch.randn(state_dim) * 0.01)  # log of diagonal A
        self.B = nn.Linear(state_dim, state_dim)
        self.C = nn.Linear(state_dim, state_dim)
        self.D = nn.Parameter(torch.randn(state_dim) * 0.01)  # skip connection
        self.dt = nn.Parameter(torch.ones(state_dim) * 0.1)  # time step
        
        self.pool = nn.AvgPool1d(kernel_size=min(scale, input_len), stride=min(scale, input_len)) if scale > 1 else nn.Identity()
        self.upsample = nn.Upsample(size=output_len, mode='linear', align_corners=False) if scale > 1 else nn.Identity()
    
    def forward(self, h):
        # h: (batch, input_len, state_dim)
        # Pool if needed
        h_scaled = h  # (batch, input_len, state_dim)
        if not isinstance(self.pool, nn.Identity):
            h_perm = h_scaled.permute(0, 2, 1)  # (batch, state_dim, input_len)
            h_pooled = self.pool(h_perm)  # (batch, state_dim, pooled_len)
            h_scaled = h_pooled.permute(0, 2, 1)  # (batch, pooled_len, state_dim)
        
        # Discretized state space step (S4-like)
        A = -torch.exp(self.A_log)  # (state_dim,)
        dt = F.softplus(self.dt)  # (state_dim,)
        
        # Simple Euler discretization for state evolution
        h_out = []
        h_state = torch.zeros(h.size(0), self.state_dim, device=h.device)  # (batch, state_dim)
        for t in range(h_scaled.size(1)):
            h_t = h_scaled[:, t, :]  # (batch, state_dim)
            # dh/dt = Ah + Bx
            dh = A.unsqueeze(0) * h_state + self.B(h_t)  # (batch, state_dim)
            h_state = h_state + dt.unsqueeze(0) * dh  # (batch, state_dim)
            # Output: y = Ch + Dx
            y_t = self.C(h_state) + self.D.unsqueeze(0) * h_t  # (batch, state_dim)
            h_out.append(y_t)
        
        h_res = torch.stack(h_out, dim=1)  # (batch, seq_len, state_dim)
        
        # Upsample to output length if needed
        if not isinstance(self.upsample, nn.Identity):
            h_perm = h_res.permute(0, 2, 1)  # (batch, state_dim, seq_len)
            h_up = self.upsample(h_perm)  # (batch, state_dim, output_len)
            h_res = h_up.permute(0, 2, 1)  # (batch, output_len, state_dim)
        else:
            # Pad or trim to output_len
            if h_res.size(1) < self.output_len:
                h_res = F.interpolate(h_res.permute(0, 2, 1), size=self.output_len, mode='linear', align_corners=False).permute(0, 2, 1)  # (batch, output_len, state_dim)
            elif h_res.size(1) > self.output_len:
                h_res = h_res[:, :self.output_len, :]  # (batch, output_len, state_dim)
        
        return h_res  # (batch, output_len, state_dim)
