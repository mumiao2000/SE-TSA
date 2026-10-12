import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierGatedMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_harmonics': [4, 16, 64],
            'dropout': [0.0, 0.3],
            'n_layers': [1, 3, 6]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_harmonics = self.config.get('n_harmonics', 16)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_layers = self.config.get('n_layers', 3)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # FFT produces complex output; we work with magnitude and phase
        self.fft_len = input_len // 2 + 1
        # Learnable harmonic selector based on dominant period ~370
        self.harmonic_gates = nn.Parameter(torch.randn(self.n_harmonics, self.fft_len) * 0.02)
        
        # Project frequency features to hidden
        self.freq_proj = nn.Linear(self.fft_len * 2, self.hidden_dim)  # *2 for real+imag
        
        # MLP backbone
        layers = []
        for i in range(self.n_layers):
            layers.append(nn.Linear(self.hidden_dim if i > 0 else self.hidden_dim, self.hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.LayerNorm(self.hidden_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, output_len)
        
        # Channel-wise adapter
        self.channel_mix = nn.Conv1d(n_features, n_features, 1)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # FFT per channel
        x_fft = torch.fft.rfft(x, dim=1)  # (batch, fft_len, n_features)
        x_real = x_fft.real  # (batch, fft_len, n_features)
        x_imag = x_fft.imag  # (batch, fft_len, n_features)
        x_freq = torch.cat([x_real, x_imag], dim=-1)  # (batch, fft_len, n_features*2) - wait, need to handle properly
        
        # Correct: process each channel's FFT separately
        # x_freq shape should be (batch, n_features, fft_len*2)
        x_real_t = x_real.permute(0, 2, 1)  # (batch, n_features, fft_len)
        x_imag_t = x_imag.permute(0, 2, 1)  # (batch, n_features, fft_len)
        x_freq = torch.cat([x_real_t, x_imag_t], dim=-1)  # (batch, n_features, fft_len*2)
        
        # Apply harmonic gating: select important frequencies
        gate = torch.softmax(self.harmonic_gates[:min(self.n_harmonics, self.fft_len)], dim=0)  # (n_harmonics, fft_len)
        # Expand and apply
        x_freq_proj = self.freq_proj(x_freq)  # (batch, n_features, hidden_dim)
        
        # MLP processing
        h = self.mlp(x_freq_proj)  # (batch, n_features, hidden_dim)
        
        # Project to output length
        out = self.output_proj(h)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Channel mixing
        out = out.permute(0, 2, 1)  # (batch, n_features, output_len)
        out = self.channel_mix(out)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

