import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class S5MambaForecast_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'gate_dim': [16, 64, 256]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.0)
        self.gate_dim = self.config.get('gate_dim', 64)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.state_dim)
        
        # S5 layers with Mamba-style gating
        self.s5_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gate_projs = nn.ModuleList()
        
        for _ in range(self.n_layers):
            # S5 state space parameters: diagonal Lambda + low-rank correction
            layer = S5Layer(self.state_dim)
            self.s5_layers.append(layer)
            self.norms.append(nn.LayerNorm(self.state_dim))
            # Mamba-style input-dependent gate
            self.gate_projs.append(nn.Sequential(
                nn.Linear(self.state_dim, self.gate_dim),
                nn.SiLU(),
                nn.Linear(self.gate_dim, self.state_dim),
                nn.Sigmoid()
            ))
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Multi-scale aggregation: different temporal resolutions
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.state_dim // 2, self.output_len * self.n_features)
        )
    
    def s5_forward(self, x, layer):
        # x: (batch, seq_len, state_dim)
        batch, seq_len, _ = x.shape
        
        # Discretize: simple Euler with learned step
        dt = F.softplus(layer.dt)  # (state_dim,)
        lambda_z = -F.softplus(layer.lambda_real) + 1j * layer.lambda_imag
        
        # Simplified S5 recurrence (diagonal case for stability)
        h = torch.zeros(batch, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            u = x[:, t, :]  # (batch, state_dim)
            # State update: h = lambda * h + B * u
            h = h * lambda_z.real + torch.matmul(u, layer.B.T)  # (batch, state_dim)
            # Output: y = C * h + D * u
            y = torch.matmul(h, layer.C.T) + layer.D * u  # (batch, state_dim)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)  # (batch, seq_len, state_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection
        x = self.input_proj(x)  # (batch, input_len, state_dim)
        
        # Multi-scale processing
        scale_outputs = []
        
        # Original scale
        x_orig = x.clone()
        for i in range(self.n_layers):
            residual = x_orig
            x_orig = self.s5_forward(x_orig, self.s5_layers[i])  # (batch, input_len, state_dim)
            gate = self.gate_projs[i](x_orig)  # (batch, input_len, state_dim)
            x_orig = x_orig * gate  # selective gating
            x_orig = self.dropout(x_orig)
            x_orig = self.norms[i](x_orig + residual)  # (batch, input_len, state_dim)
        scale_outputs.append(x_orig[:, -1, :])  # (batch, state_dim)
        
        # Downsampled scale (factor 2)
        if self.input_len >= 4:
            x_down = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=2, stride=2).permute(0, 2, 1)  # (batch, input_len//2, state_dim)
            for i in range(min(2, self.n_layers)):
                residual = x_down
                x_down = self.s5_forward(x_down, self.s5_layers[i])
                gate = self.gate_projs[i](x_down)
                x_down = x_down * gate
                x_down = self.dropout(x_down)
                x_down = self.norms[i](x_down + residual)
            scale_outputs.append(F.interpolate(x_down[:, -1:, :].permute(0, 2, 1), size=1, mode='linear').squeeze(-1))  # (batch, state_dim)
        else:
            scale_outputs.append(scale_outputs[0])
        
        # Very coarse scale (factor 4)
        if self.input_len >= 8:
            x_coarse = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=4, stride=4).permute(0, 2, 1)  # (batch, input_len//4, state_dim)
            for i in range(min(1, self.n_layers)):
                residual = x_coarse
                x_coarse = self.s5_forward(x_coarse, self.s5_layers[i])
                gate = self.gate_projs[i](x_coarse)
                x_coarse = x_coarse * gate
                x_coarse = self.dropout(x_coarse)
                x_coarse = self.norms[i](x_coarse + residual)
            scale_outputs.append(F.interpolate(x_coarse[:, -1:, :].permute(0, 2, 1), size=1, mode='linear').squeeze(-1))  # (batch, state_dim)
        else:
            scale_outputs.append(scale_outputs[0])
        
        # Weighted aggregation of scales
        weights = F.softmax(self.scale_weights, dim=0)  # (3,)
        x_agg = sum(w * s for w, s in zip(weights, scale_outputs))  # (batch, state_dim)
        
        # Output projection
        out = self.output_proj(x_agg)  # (batch, output_len * n_features)
        out = out.view(-1, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class S5Layer(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.lambda_real = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.lambda_imag = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.D = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.dt = nn.Parameter(torch.ones(state_dim) * 0.01)

