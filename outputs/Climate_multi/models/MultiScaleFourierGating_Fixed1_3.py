import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class MultiScaleFourierGating_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_frequencies': [8, 32, 128],
            'n_scales': [2, 4, 8],
            'hidden_dim': [64, 256, 512],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.n_frequencies = self.config.get('n_frequencies', 32)
        self.n_scales = self.config.get('n_scales', 4)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multiple scales: different window sizes for FFT
        self.scales = [max(2 ** i, 4) for i in range(self.n_scales)]
        
        # Per-scale, per-channel frequency gating networks
        # Each scale produces n_frequencies*2 features (real/imag)
        self.freq_gates = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.n_frequencies * 2, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.n_frequencies * 2),
                    nn.Sigmoid()
                ) for _ in range(self.n_scales)
            ]) for _ in range(self.n_features)
        ])
        
        # Frequency to time projection per channel
        total_freq_dim = self.n_frequencies * 2 * self.n_scales
        self.freq_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(total_freq_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_len)
            ) for _ in range(self.n_features)
        ])
        
        # Time-domain MLP for residual
        self.time_mlp = nn.Sequential(
            nn.Linear(self.input_len * self.n_features, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_len * self.n_features)
        )
        
        # Learnable mixing weights per channel
        self.mix_weights = nn.Parameter(torch.ones(self.n_features, 2) / 2)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Process each channel separately
        freq_outputs = []
        for c in range(self.n_features):
            xc = x[:, :, c]  # (batch, input_len)
            
            scale_features = []
            for i, scale in enumerate(self.scales):
                # Pad or truncate to scale length
                if self.input_len >= scale:
                    xc_pad = xc[:, -scale:]  # (batch, scale)
                else:
                    xc_pad = F.pad(xc, (0, scale - self.input_len))  # (batch, scale)
                
                # FFT
                xc_fft = torch.fft.rfft(xc_pad, n=scale, dim=-1)  # (batch, scale//2+1)
                
                # Pad or truncate to exactly n_frequencies components
                fft_len = xc_fft.shape[-1]
                if fft_len >= self.n_frequencies:
                    xc_fft = xc_fft[:, :self.n_frequencies]  # (batch, n_frequencies)
                else:
                    pad_size = self.n_frequencies - fft_len
                    xc_fft = F.pad(xc_fft, (0, pad_size))  # (batch, n_frequencies)
                
                # Real and imaginary parts
                xc_fft_feat = torch.cat([xc_fft.real, xc_fft.imag], dim=-1)  # (batch, n_frequencies*2)
                
                # Apply per-scale gating
                gate = self.freq_gates[c][i](xc_fft_feat)  # (batch, n_frequencies*2)
                xc_fft_gated = xc_fft_feat * gate  # (batch, n_frequencies*2)
                scale_features.append(xc_fft_gated)
            
            # Concatenate all scales
            freq_feat = torch.cat(scale_features, dim=-1)  # (batch, n_frequencies*2*n_scales)
            
            # Project to output
            out_c = self.freq_projectors[c](freq_feat)  # (batch, output_len)
            freq_outputs.append(out_c.unsqueeze(-1))  # (batch, output_len, 1)
        
        freq_out = torch.cat(freq_outputs, dim=-1)  # (batch, output_len, n_features)
        
        # Time-domain residual
        time_in = x.reshape(batch_size, -1)  # (batch, input_len*n_features)
        time_out = self.time_mlp(time_in)  # (batch, output_len*n_features)
        time_out = time_out.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Mix frequency and time domain predictions
        mix = F.softmax(self.mix_weights, dim=-1)  # (n_features, 2)
        out = freq_out * mix[None, None, :, 0] + time_out * mix[None, None, :, 1]  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

