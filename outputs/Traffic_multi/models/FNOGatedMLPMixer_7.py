import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNOGatedMLPMixer(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'n_modes': [8, 32, 128],
            'dropout': [0.0, 0.3],
            'expansion_factor': [2, 4, 8],
            'mixing_ratio': [0.25, 0.5, 0.75]  # FNO vs MLP ratio
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 256)
        self.n_layers = cfg.get('n_layers', 4)
        self.n_modes = cfg.get('n_modes', 32)
        self.dropout = cfg.get('dropout', 0.1)
        self.expansion_factor = cfg.get('expansion_factor', 4)
        self.mixing_ratio = cfg.get('mixing_ratio', 0.5)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(n_features, self.d_model)
        
        # FNO-MLP blocks
        n_fno_layers = max(1, int(self.n_layers * self.mixing_ratio))
        n_mlp_layers = max(1, self.n_layers - n_fno_layers)
        
        self.fno_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(n_fno_layers):
            self.fno_layers.append(FNOBlock(self.d_model, self.n_modes, self.dropout))
            self.norms.append(nn.LayerNorm(self.d_model))
            
        for _ in range(n_mlp_layers):
            self.mlp_layers.append(GatedMLPBlock(self.d_model, self.expansion_factor, self.dropout))
            self.norms.append(nn.LayerNorm(self.d_model))
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, n_features)
        
        # Temporal interpolation for variable length
        self.temporal_upsample = nn.Linear(input_len, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, input_len, d_model)
        x = x.permute(0, 2, 1)  # (batch, d_model, input_len)
        
        # Apply FNO layers
        for i, fno in enumerate(self.fno_layers):
            residual = x
            x = fno(x)  # (batch, d_model, input_len)
            x = x.permute(0, 2, 1)  # (batch, input_len, d_model)
            x = self.norms[i](x)
            x = x.permute(0, 2, 1)  # (batch, d_model, input_len)
            x = x + residual
            
        # Apply MLP layers
        for j, mlp in enumerate(self.mlp_layers):
            idx = len(self.fno_layers) + j
            residual = x
            x = x.permute(0, 2, 1)  # (batch, input_len, d_model)
            x = mlp(x)  # (batch, input_len, d_model)
            x = self.norms[idx](x)
            x = x.permute(0, 2, 1)  # (batch, d_model, input_len)
            x = x + residual
            
        # Temporal upsampling
        x = self.temporal_upsample(x)  # (batch, d_model, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, d_model)
        
        # Project back to features
        x = self.output_proj(x)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class FNOBlock(nn.Module):
    def __init__(self, d_model, n_modes, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_modes = n_modes
        
        # Complex weights for frequency domain
        self.scale = 1.0 / (d_model * d_model)
        self.weights = nn.Parameter(self.scale * torch.randn(d_model, d_model, n_modes, 2))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, d_model, seq_len)
        batch_size, d_model, seq_len = x.shape
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, d_model, seq_len//2+1)
        
        # Filter modes
        out_ft = torch.zeros_like(x_ft)
        n_modes = min(self.n_modes, x_ft.shape[-1])
        
        # Apply learned weights to low frequencies
        weights_complex = torch.view_as_complex(self.weights[:, :, :n_modes, :].contiguous())
        # weights: (d_model, d_model, n_modes)
        
        # Manual complex multiplication
        for i in range(n_modes):
            out_ft[:, :, i] = torch.einsum('bd,dm->bm', x_ft[:, :, i], weights_complex[:, :, i])
            
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=seq_len, dim=-1)  # (batch, d_model, seq_len)
        x = self.dropout(x)
        return x


class GatedMLPBlock(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)
        self.fc1 = nn.Linear(d_model, hidden_dim * 2)  # For gating
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.fc1(x)  # (batch, seq_len, hidden_dim*2)
        x, gate = x.chunk(2, dim=-1)  # each (batch, seq_len, hidden_dim)
        x = x * F.gelu(gate)  # Gated activation
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, seq_len, d_model)
        return x

