import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SpectralTemporalMixer(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_freq_bands': [8, 32, 128],
            'mixing_layers': [2, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_freq_bands = self.config.get('n_freq_bands', 32)
        self.mixing_layers = self.config.get('mixing_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.d_model)
        
        # FFT frequency filters (learnable complex weights for key frequencies)
        self.max_freq = self.input_len // 2 + 1
        self.freq_filter_real = nn.Parameter(torch.randn(min(self.n_freq_bands, self.max_freq), self.d_model) * 0.02)
        self.freq_filter_imag = nn.Parameter(torch.randn(min(self.n_freq_bands, self.max_freq), self.d_model) * 0.02)
        
        # Temporal mixing MLPs
        self.temporal_mlps = nn.ModuleList()
        for _ in range(self.mixing_layers):
            self.temporal_mlps.append(nn.Sequential(
                nn.Linear(self.input_len, self.d_model * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * 2, self.input_len)
            ))
        
        # Cross-layer mixing
        self.cross_mix = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Output projection with direct skip
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, self.n_features)
        )
        
        # Direct linear forecast for stability
        self.direct_forecast = nn.Linear(self.input_len, self.output_len)
        
        # Gating for spectral vs temporal
        self.gate = nn.Linear(self.d_model * 2, self.d_model)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to d_model
        x_proj = self.input_proj(x)  # (batch, input_len, d_model)
        
        # === Spectral Branch ===
        # Apply FFT per feature dimension
        x_fft = torch.fft.rfft(x_proj, dim=1, norm='ortho')  # (batch, max_freq, d_model)
        
        # Apply learnable frequency filters (only to first n_freq_bands)
        n_apply = min(self.n_freq_bands, x_fft.shape[1])
        freq_weights = torch.view_as_complex(torch.stack([
            self.freq_filter_real[:n_apply],
            self.freq_filter_imag[:n_apply]
        ], dim=-1))  # (n_apply, d_model)
        
        x_fft[:, :n_apply] = x_fft[:, :n_apply] * freq_weights.unsqueeze(0)  # (batch, n_apply, d_model)
        
        # Inverse FFT
        x_spectral = torch.fft.irfft(x_fft, n=self.input_len, dim=1, norm='ortho')  # (batch, input_len, d_model)
        
        # === Temporal Branch ===
        x_temp = x_proj.permute(0, 2, 1)  # (batch, d_model, input_len)
        for mlp in self.temporal_mlps:
            x_temp = x_temp + mlp(x_temp)  # (batch, d_model, input_len), residual connection
        x_temp = x_temp.permute(0, 2, 1)  # (batch, input_len, d_model)
        
        # === Fusion ===
        # Global pooling for context
        spectral_context = x_spectral.mean(dim=1)  # (batch, d_model)
        temporal_context = x_temp.mean(dim=1)  # (batch, d_model)
        
        # Gating mechanism
        gate_input = torch.cat([spectral_context, temporal_context], dim=-1)  # (batch, d_model * 2)
        gate = torch.sigmoid(self.gate(gate_input))  # (batch, d_model)
        
        # Apply gate to time-step features
        x_fused = gate.unsqueeze(1) * x_spectral + (1 - gate.unsqueeze(1)) * x_temp  # (batch, input_len, d_model)
        
        # Cross-mix
        x_mixed = self.cross_mix(torch.cat([x_fused, x_proj], dim=-1))  # (batch, input_len, d_model)
        
        # Output: predict each future time step
        # Use last hidden state and project
        last_hidden = x_mixed[:, -1, :]  # (batch, d_model)
        
        # Generate output sequence
        outputs = []
        current = last_hidden
        for _ in range(self.output_len):
            out_step = self.output_proj(current)  # (batch, n_features)
            outputs.append(out_step)
            # Update current (simple recurrence)
            current = current + torch.tanh(self.cross_mix[0](torch.cat([current, last_hidden], dim=-1)))
        
        out = torch.stack(outputs, dim=1)  # (batch, output_len, n_features)
        
        # Direct residual connection
        direct = self.direct_forecast(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, output_len, n_features)
        out = out + direct * 0.1  # soft residual
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

