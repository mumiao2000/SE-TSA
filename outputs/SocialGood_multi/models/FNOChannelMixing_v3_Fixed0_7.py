import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNOChannelMixing_v3_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [256, 512, 1024],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.2, 0.4],
            'n_modes_coarse': [8, 32, 128],
            'n_modes_fine': [32, 128, 512],
            'cde_dim': [32, 64, 128],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.2)
        self.n_modes_coarse = self.config.get('n_modes_coarse', 32)
        self.n_modes_fine = self.config.get('n_modes_fine', 128)
        self.cde_dim = self.config.get('cde_dim', 64)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Multi-scale FNO blocks
        self.fno_blocks = nn.ModuleList()
        for i in range(self.n_layers):
            self.fno_blocks.append(
                MultiScaleFNOBlock(
                    self.hidden_dim, 
                    self.n_modes_coarse, 
                    self.n_modes_fine,
                    self.dropout
                )
            )
        
        # Neural CDE-inspired temporal evolution
        self.cde_encoder = CDETemporalEncoder(
            self.hidden_dim, 
            self.cde_dim,
            self.input_len
        )
        
        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim + self.cde_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features * self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection
        h = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Multi-scale FNO processing
        for block in self.fno_blocks:
            h = block(h) + h  # (batch, input_len, hidden_dim)
        
        # CDE temporal evolution
        cde_out = self.cde_encoder(h)  # (batch, cde_dim)
        
        # Take last timestep for prediction
        h_last = h[:, -1, :]  # (batch, hidden_dim)
        
        # Fusion
        fused = torch.cat([h_last, cde_out], dim=-1)  # (batch, hidden_dim + cde_dim)
        fused = self.fusion(fused)  # (batch, hidden_dim)
        
        # Output projection
        out = self.output_proj(fused)  # (batch, n_features * output_len)
        out = out.view(-1, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class MultiScaleFNOBlock(nn.Module):
    def __init__(self, hidden_dim, n_modes_coarse, n_modes_fine, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modes_coarse = n_modes_coarse
        self.n_modes_fine = n_modes_fine
        
        # Coarse scale spectral convolution
        self.coarse_filter = nn.Parameter(
            torch.randn(hidden_dim, n_modes_coarse, 2) * 0.02
        )
        
        # Fine scale spectral convolution  
        self.fine_filter = nn.Parameter(
            torch.randn(hidden_dim, n_modes_fine, 2) * 0.02
        )
        
        # Learnable gating for scale fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        batch, seq_len, hidden = x.shape  # (batch, seq_len, hidden_dim)
        
        # Coarse spectral path
        x_ft_coarse = torch.fft.rfft(x, dim=1, norm='ortho')  # (batch, seq_len//2+1, hidden_dim)
        x_ft_coarse = x_ft_coarse.permute(0, 2, 1)  # (batch, hidden_dim, seq_len//2+1)
        
        out_coarse = torch.zeros_like(x_ft_coarse)  # (batch, hidden_dim, seq_len//2+1)
        modes_c = min(self.n_modes_coarse, x_ft_coarse.size(-1))
        # Fix: use element-wise multiplication with broadcasted complex filter
        coarse_complex = torch.view_as_complex(self.coarse_filter[:, :modes_c, :])  # (hidden_dim, modes_c)
        out_coarse[:, :, :modes_c] = x_ft_coarse[:, :, :modes_c] * coarse_complex.unsqueeze(0)  # (batch, hidden_dim, modes_c)
        
        out_coarse = out_coarse.permute(0, 2, 1)  # (batch, seq_len//2+1, hidden_dim)
        x_coarse = torch.fft.irfft(out_coarse, n=seq_len, dim=1, norm='ortho')  # (batch, seq_len, hidden_dim)
        
        # Fine spectral path
        x_ft_fine = torch.fft.rfft(x, dim=1, norm='ortho')  # (batch, seq_len//2+1, hidden_dim)
        x_ft_fine = x_ft_fine.permute(0, 2, 1)  # (batch, hidden_dim, seq_len//2+1)
        
        out_fine = torch.zeros_like(x_ft_fine)  # (batch, hidden_dim, seq_len//2+1)
        modes_f = min(self.n_modes_fine, x_ft_fine.size(-1))
        # Fix: use element-wise multiplication with broadcasted complex filter
        fine_complex = torch.view_as_complex(self.fine_filter[:, :modes_f, :])  # (hidden_dim, modes_f)
        out_fine[:, :, :modes_f] = x_ft_fine[:, :, :modes_f] * fine_complex.unsqueeze(0)  # (batch, hidden_dim, modes_f)
        
        out_fine = out_fine.permute(0, 2, 1)  # (batch, seq_len//2+1, hidden_dim)
        x_fine = torch.fft.irfft(out_fine, n=seq_len, dim=1, norm='ortho')  # (batch, seq_len, hidden_dim)
        
        # Adaptive gating based on per-timestep features
        gate_weights = self.gate(x)  # (batch, seq_len, 2)
        
        # Fuse with gating
        x_fused = gate_weights[:, :, 0:1] * x_coarse + gate_weights[:, :, 1:2] * x_fine  # (batch, seq_len, hidden_dim)
        
        # FFN with residual
        x_out = self.norm(x + self.dropout(x_fused))  # (batch, seq_len, hidden_dim)
        x_out = x_out + self.ffn(x_out)  # (batch, seq_len, hidden_dim)
        
        return x_out


class CDETemporalEncoder(nn.Module):
    def __init__(self, hidden_dim, cde_dim, input_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cde_dim = cde_dim
        
        # CDE dynamics network: maps current state to derivative
        self.dynamics = nn.Sequential(
            nn.Linear(cde_dim + 1, cde_dim * 2),  # +1 for time
            nn.Tanh(),
            nn.Linear(cde_dim * 2, cde_dim),
        )
        
        # Initial state from input sequence
        self.init_proj = nn.Linear(hidden_dim, cde_dim)
        
        # Fixed time points for integration
        self.register_buffer('time_points', torch.linspace(0, 1, input_len))
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        batch, seq_len, hidden = x.shape  # (batch, seq_len, hidden_dim)
        
        # Initial state from mean pooling
        x_mean = x.mean(dim=1)  # (batch, hidden_dim)
        z0 = self.init_proj(x_mean)  # (batch, cde_dim)
        
        # Simple Euler integration of CDE dynamics
        z = z0  # (batch, cde_dim)
        dt = 1.0 / seq_len
        
        for t in range(seq_len):
            t_val = torch.tensor(t * dt, device=x.device).expand(batch, 1)  # (batch, 1)
            z_input = torch.cat([z, t_val], dim=-1)  # (batch, cde_dim + 1)
            dz = self.dynamics(z_input)  # (batch, cde_dim)
            z = z + dt * dz  # (batch, cde_dim)
        
        return z  # (batch, cde_dim)
