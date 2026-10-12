import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierNeuralOperatorForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'fourier_modes': [8, 32, 128],
            'hidden_dim': [64, 256, 1024],
            'n_fourier_layers': [1, 3, 6],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.fourier_modes = self.config.get('fourier_modes', 32)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_fourier_layers = self.config.get('n_fourier_layers', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Project to hidden dim
        self.lift = nn.Linear(self.n_features, self.hidden_dim)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList()
        for _ in range(self.n_fourier_layers):
            self.fourier_layers.append(FourierLayer(self.hidden_dim, self.fourier_modes, self.dropout_rate))
        
        # Project back
        self.project = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.n_features)
        )
        
        # Temporal interpolation for output length
        self.temporal_upsample = nn.Sequential(
            nn.Linear(self.input_len, self.input_len * 2),
            nn.GELU(),
            nn.Linear(self.input_len * 2, self.output_len)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Lift to hidden dimension
        h = self.lift(x)  # (batch, input_len, hidden_dim)
        
        # Apply Fourier layers
        for layer in self.fourier_layers:
            h = h + layer(h)  # (batch, input_len, hidden_dim)
        
        # Temporal upsampling to output length
        h = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        h = self.temporal_upsample(h)  # (batch, hidden_dim, output_len)
        h = h.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Project to features
        out = self.project(h)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class FourierLayer(nn.Module):
    def __init__(self, hidden_dim, modes, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modes = modes
        
        # Complex weights for frequency filtering
        self.scale = 1.0 / (hidden_dim * hidden_dim)
        self.weights = nn.Parameter(self.scale * torch.randn(modes, hidden_dim, hidden_dim, 2))
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        batch, seq_len, _ = x.shape
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=1)  # (batch, seq_len//2+1, hidden_dim), complex
        
        # Truncate to modes and apply learned filter
        out_ft = torch.zeros_like(x_ft)
        modes = min(self.modes, x_ft.size(1))
        
        # Apply complex multiplication: (batch, modes, hidden_dim) @ (modes, hidden_dim, hidden_dim)
        weights_complex = torch.view_as_complex(self.weights[:modes])  # (modes, hidden_dim, hidden_dim)
        
        x_ft_modes = x_ft[:, :modes, :]  # (batch, modes, hidden_dim)
        
        # Batch matrix multiply
        out_ft[:, :modes, :] = torch.einsum('bmi,mio->bmo', x_ft_modes, weights_complex)  # (batch, modes, hidden_dim)
        
        # IFFT
        out = torch.fft.irfft(out_ft, n=seq_len, dim=1)  # (batch, seq_len, hidden_dim)
        
        # MLP on channel dimension
        out = self.norm(out)  # (batch, seq_len, hidden_dim)
        out = out + self.mlp(out)  # (batch, seq_len, hidden_dim)
        return out
