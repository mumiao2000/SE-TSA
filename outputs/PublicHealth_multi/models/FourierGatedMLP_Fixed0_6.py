import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierGatedMLP_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'freq_dim': [16, 64, 256]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.freq_dim = self.config.get('freq_dim', 64)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # FFT-based frequency encoder
        self.freq_encoder = nn.Sequential(
            nn.Linear(input_len // 2 + 1, self.freq_dim),  # (batch, n_features, freq_dim)
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Project frequency features to gating signal (match hidden_dim * 4)
        self.freq_to_gate = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim * 4),
            nn.Sigmoid()
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_len, self.hidden_dim)
        
        # Gated MLP layers
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(nn.ModuleDict({
                'ln': nn.LayerNorm(self.hidden_dim),
                'up': nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                'gate': nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                'down': nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                'dropout': nn.Dropout(self.dropout_rate)
            }))
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Compute FFT-based gating signal
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_fft = torch.fft.rfft(x_perm, dim=-1)  # (batch, n_features, input_len // 2 + 1)
        x_fft_mag = torch.abs(x_fft)  # (batch, n_features, input_len // 2 + 1)
        
        # Encode frequency features
        freq_feat = self.freq_encoder(x_fft_mag)  # (batch, n_features, freq_dim)
        
        # Global average pooling over features for gating
        freq_global = freq_feat.mean(dim=1)  # (batch, freq_dim)
        gate_signal = self.freq_to_gate(freq_global)  # (batch, hidden_dim * 4)
        
        # Main MLP pathway
        h = self.input_proj(x_perm)  # (batch, n_features, hidden_dim)
        
        # Apply gated MLP layers with frequency modulation
        for layer in self.layers:
            h = layer['ln'](h)  # (batch, n_features, hidden_dim)
            h_up = layer['up'](h)  # (batch, n_features, hidden_dim * 4)
            h_gate = layer['gate'](h)  # (batch, n_features, hidden_dim * 4)
            # Apply frequency gating: broadcast gate_signal across features
            h_gated = h_up * h_gate * gate_signal.unsqueeze(1)  # (batch, n_features, hidden_dim * 4)
            h_gated = F.gelu(h_gated)
            h = h + layer['dropout'](layer['down'](h_gated))  # (batch, n_features, hidden_dim)
        
        # Output projection
        out = self.output_proj(h)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

