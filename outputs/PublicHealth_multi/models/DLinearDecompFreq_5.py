import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DLinearDecompFreq(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'kernel_size': [3, 13, 25],
            'hidden_ratio': [1, 2, 4],
            'use_seasonal': [True, False],
            'individual': [True, False],
            'freq_dim': [16, 64, 256],
            'use_freq_gate': [True, False],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.kernel_size = self.config.get('kernel_size', 13)
        self.hidden_ratio = self.config.get('hidden_ratio', 2)
        self.use_seasonal = self.config.get('use_seasonal', True)
        self.individual = self.config.get('individual', True)
        self.freq_dim = self.config.get('freq_dim', 64)
        self.use_freq_gate = self.config.get('use_freq_gate', True)
        
        # Ensure odd kernel size for moving average
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
        self.padding = self.kernel_size // 2
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Decomposition: learnable moving average for trend extraction
        self.avg_pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=self.padding)
        
        # Frequency branch: FFT-based feature extraction
        # Use real FFT to get magnitude and phase information
        self.fft_proj = nn.Linear(input_len, self.freq_dim)  # Project FFT features
        self.freq_encoder = nn.Sequential(
            nn.Linear(self.freq_dim, self.freq_dim // 2),
            nn.ReLU(),
            nn.Linear(self.freq_dim // 2, self.freq_dim),
        )
        self.freq_decoder = nn.Linear(self.freq_dim, output_len)
        
        # Hidden dimension for MLPs
        self.hidden_dim = int(self.n_features * self.hidden_ratio)
        
        # Trend projection
        if self.individual:
            self.trend_linears = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_len, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, output_len)
                ) for _ in range(self.n_features)
            ])
        else:
            self.trend_linear = nn.Sequential(
                nn.Linear(input_len, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, output_len)
            )
        
        # Seasonal projection
        if self.use_seasonal:
            if self.individual:
                self.seasonal_linears = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_len, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, output_len)
                    ) for _ in range(self.n_features)
                ])
            else:
                self.seasonal_linear = nn.Sequential(
                    nn.Linear(input_len, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, output_len)
                )
        
        # Adaptive gating: learn to blend trend, seasonal, and frequency
        if self.use_freq_gate:
            # Per-channel gate based on input statistics
            self.gate_encoder = nn.Sequential(
                nn.Linear(n_features * 2, self.hidden_dim),  # mean and std of input
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 3 if self.use_seasonal else 2),  # weights for trend, seasonal, freq
            )
        
        # Frequency fusion: combine frequency features with time-domain features
        self.freq_fusion = nn.Linear(output_len, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Permute for 1D operations: (batch, n_features, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Trend extraction via moving average
        trend = self.avg_pool(x_perm)  # (batch, n_features, input_len)
        seasonal = x_perm - trend  # (batch, n_features, input_len)
        
        # Frequency branch: FFT-based features
        # Apply FFT per channel
        fft_vals = torch.fft.rfft(x_perm, dim=-1)  # (batch, n_features, input_len//2+1)
        fft_magnitude = torch.abs(fft_vals)  # (batch, n_features, input_len//2+1)
        fft_phase = torch.angle(fft_vals)  # (batch, n_features, input_len//2+1)
        
        # Concatenate magnitude and phase, then project
        fft_features = torch.cat([fft_magnitude, fft_phase], dim=-1)  # (batch, n_features, input_len+2)
        # Truncate or pad to match input_len for projection
        fft_features = fft_features[:, :, :self.input_len]  # (batch, n_features, input_len)
        
        # Project and encode frequency features
        fft_proj = self.fft_proj(fft_features)  # (batch, n_features, freq_dim)
        freq_encoded = self.freq_encoder(fft_proj)  # (batch, n_features, freq_dim)
        freq_out = self.freq_decoder(freq_encoded)  # (batch, n_features, output_len)
        
        # Trend projection
        if self.individual:
            trend_out = torch.stack([
                self.trend_linears[i](trend[:, i, :])  # (batch, output_len)
                for i in range(self.n_features)
            ], dim=1)  # (batch, n_features, output_len)
        else:
            trend_out = self.trend_linear(trend)  # (batch, n_features, output_len)
        
        # Seasonal projection
        if self.use_seasonal:
            if self.individual:
                seasonal_out = torch.stack([
                    self.seasonal_linears[i](seasonal[:, i, :])  # (batch, output_len)
                    for i in range(self.n_features)
                ], dim=1)  # (batch, n_features, output_len)
            else:
                seasonal_out = self.seasonal_linear(seasonal)  # (batch, n_features, output_len)
            
            # Base output: trend + seasonal
            base_out = trend_out + seasonal_out  # (batch, n_features, output_len)
        else:
            base_out = trend_out  # (batch, n_features, output_len)
        
        # Frequency fusion
        freq_fused = self.freq_fusion(freq_out)  # (batch, n_features, output_len)
        
        # Adaptive gating
        if self.use_freq_gate:
            # Compute input statistics for gating
            x_mean = x.mean(dim=1)  # (batch, n_features)
            x_std = x.std(dim=1)  # (batch, n_features)
            gate_input = torch.cat([x_mean, x_std], dim=-1)  # (batch, n_features * 2)
            
            # Compute gate weights
            gate_logits = self.gate_encoder(gate_input)  # (batch, 3) or (batch, 2)
            gate_weights = F.softmax(gate_logits, dim=-1)  # (batch, num_components)
            
            # Apply gates
            if self.use_seasonal:
                # gate_weights: [w_trend, w_seasonal, w_freq]
                w_trend = gate_weights[:, 0:1].unsqueeze(-1)  # (batch, 1, 1)
                w_seasonal = gate_weights[:, 1:2].unsqueeze(-1)  # (batch, 1, 1)
                w_freq = gate_weights[:, 2:3].unsqueeze(-1)  # (batch, 1, 1)
                
                out = w_trend * trend_out + w_seasonal * seasonal_out + w_freq * freq_fused  # (batch, n_features, output_len)
            else:
                # gate_weights: [w_trend, w_freq]
                w_trend = gate_weights[:, 0:1].unsqueeze(-1)  # (batch, 1, 1)
                w_freq = gate_weights[:, 1:2].unsqueeze(-1)  # (batch, 1, 1)
                
                out = w_trend * trend_out + w_freq * freq_fused  # (batch, n_features, output_len)
        else:
            # Simple addition without gating
            out = base_out + freq_fused  # (batch, n_features, output_len)
        
        # Permute back: (batch, output_len, n_features)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
