import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierNeuralOperatorTS_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_modes': [8, 32, 128],
            'hidden_dim': [64, 256, 1024],
            'n_fno_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_modes = self.config.get('n_modes', 32)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_fno_layers = self.config.get('n_fno_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Lifting: project features to hidden dimension
        self.lifting = nn.Linear(self.n_features, self.hidden_dim)
        
        # FNO layers in frequency domain
        self.fno_layers = nn.ModuleList()
        for _ in range(self.n_fno_layers):
            self.fno_layers.append(
                SpectralConv1d(self.hidden_dim, self.hidden_dim, self.n_modes, self.dropout)
            )
        
        # Time extension: input_len -> output_len in frequency domain or via learned interpolation
        # We'll use a learned temporal MLP
        self.time_extend = nn.Sequential(
            nn.Linear(self.input_len, self.output_len),
            nn.GELU()
        )
        
        # Projection back to features
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.n_features)
        )
        
        # Direct skip
        self.skip_weight = nn.Parameter(torch.tensor(0.05))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Lifting
        h = self.lifting(x)  # (batch, input_len, hidden_dim)
        
        # FNO layers with residual connections
        for layer in self.fno_layers:
            h = h + layer(h)  # (batch, input_len, hidden_dim)
        
        # Extend time dimension
        h = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        h = self.time_extend(h)  # (batch, hidden_dim, output_len)
        h = h.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Project to features
        out = self.projection(h)  # (batch, output_len, n_features)
        
        # Skip: repeat last with linear trend estimate
        last = x[:, -1:, :]  # (batch, 1, n_features)
        # Simple trend: difference between last two
        if self.input_len > 1:
            trend = x[:, -1:, :] - x[:, -2:-1, :]  # (batch, 1, n_features)
        else:
            trend = 0
        
        time_idx = torch.arange(1, self.output_len + 1, device=x.device).float().view(1, -1, 1)  # (1, output_len, 1)
        skip = last + self.skip_weight * trend * time_idx  # (batch, output_len, n_features)
        
        out = out + skip  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes, dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes  # Number of Fourier modes to keep
        
        # Complex weights for Fourier modes - initialize with n_modes size
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(torch.randn(in_channels, out_channels, n_modes) * scale)
        self.weights_imag = nn.Parameter(torch.randn(in_channels, out_channels, n_modes) * scale)
        
        # Gating in frequency domain - will adapt to actual n_freq at runtime
        self.freq_gate = nn.Sequential(
            nn.Linear(n_modes * 2, n_modes),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, in_channels)
        batch_size, seq_len, _ = x.shape
        
        # FFT along time dimension
        x_fft = torch.fft.rfft(x, dim=1)  # (batch, seq_len//2+1, in_channels)
        
        # Get relevant modes - actual number of frequencies we'll use
        n_freq = min(self.n_modes, x_fft.size(1))
        
        # Truncate to n_freq (do NOT pad)
        x_fft_trunc = x_fft[:, :n_freq, :]  # (batch, n_freq, in_channels)
        
        # Transpose for matrix multiply: (batch, in_channels, n_freq)
        x_fft_trunc = x_fft_trunc.permute(0, 2, 1)  # (batch, in_channels, n_freq)
        
        # Apply complex weights - slice weights to n_freq to match
        weights = torch.complex(self.weights_real[:, :, :n_freq], self.weights_imag[:, :, :n_freq])
        # (in_channels, out_channels, n_freq)
        
        # Batch matrix multiply
        # x_fft_trunc: (batch, in_channels, n_freq)
        # weights: (in_channels, out_channels, n_freq)
        out_fft = torch.einsum('bif,iof->bof', x_fft_trunc, weights)  # (batch, out_channels, n_freq)
        
        # Gating based on frequency magnitudes
        gate_input = torch.cat([
            x_fft_trunc.real.mean(dim=1),  # (batch, n_freq)
            x_fft_trunc.imag.mean(dim=1)   # (batch, n_freq)
        ], dim=-1)  # (batch, 2*n_freq)
        
        # Pad gate_input to match expected input size of freq_gate if needed
        if n_freq < self.n_modes:
            gate_input = F.pad(gate_input, (0, (self.n_modes - n_freq) * 2))
        
        gate = self.freq_gate(gate_input)  # (batch, n_modes)
        gate = gate[:, :n_freq]  # (batch, n_freq)
        
        out_fft = out_fft * gate.unsqueeze(1)  # (batch, out_channels, n_freq)
        
        # Pad back to original frequency size for irfft
        if seq_len // 2 + 1 > n_freq:
            out_fft_full = F.pad(out_fft, (0, seq_len // 2 + 1 - n_freq))  # (batch, out_channels, seq_len//2+1)
        else:
            out_fft_full = out_fft
        
        # Inverse FFT
        out_fft_full = out_fft_full.permute(0, 2, 1)  # (batch, seq_len//2+1, out_channels)
        out = torch.fft.irfft(out_fft_full, n=seq_len, dim=1)  # (batch, seq_len, out_channels)
        
        out = self.dropout(out)
        
        return out
