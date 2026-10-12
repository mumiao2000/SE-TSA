import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveFourierGatedForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_modes': [16, 32, 64],
            'hidden_channels': [64, 256, 512],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_modes = self.config.get('n_modes', 32)
        self.hidden_channels = self.config.get('hidden_channels', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.3)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input embedding
        self.input_embed = nn.Linear(self.n_features, self.hidden_channels)
        
        # Fourier layers with adaptive gating
        self.fourier_layers = nn.ModuleList()
        self.temporal_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(self.n_layers):
            # Custom module to hold parameter and gate together
            fourier_module = nn.Module()
            fourier_module.freq_weights = nn.Parameter(torch.ones(self.n_modes))  # adaptive frequency importance
            fourier_module.freq_gate = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels // 4),
                nn.GELU(),
                nn.Linear(self.hidden_channels // 4, self.n_modes),
                nn.Sigmoid()
            )  # channel-specific frequency gating
            fourier_module.R = nn.Linear(self.hidden_channels, self.hidden_channels)
            self.fourier_layers.append(fourier_module)
            
            # Parallel temporal convolution for local patterns
            self.temporal_convs.append(nn.Sequential(
                nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1, groups=self.hidden_channels // 4),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=1),
            ))
            
            self.norms.append(nn.LayerNorm(self.hidden_channels))
        
        # Channel-adaptive output
        self.channel_gate = nn.Sequential(
            nn.Linear(self.hidden_channels, 64),
            nn.GELU(),
            nn.Linear(64, self.n_features),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_channels, 256),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.n_features)
        )
        
        # Direct mapping
        self.direct_map = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Direct residual
        x_direct = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_direct = self.direct_map(x_direct)  # (batch, n_features, output_len)
        x_direct = x_direct.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Embed
        h = self.input_embed(x)  # (batch, input_len, hidden_channels)
        
        # Process through Fourier + temporal layers
        for i, (fourier, tconv, norm) in enumerate(zip(self.fourier_layers, self.temporal_convs, self.norms)):
            h_in = h  # (batch, input_len, hidden_channels)
            
            # Fourier branch
            h_fft = torch.fft.rfft(h, dim=1)  # (batch, fft_len, hidden_channels)
            
            # Apply adaptive frequency gating
            freq_gate = fourier.freq_gate(h.mean(dim=1))  # (batch, n_modes)
            
            # Weight and gate frequencies
            n_active = min(self.n_modes, h_fft.size(1))
            weights = fourier.freq_weights[:n_active] * freq_gate[:, :n_active]  # (batch, n_active)
            
            h_fft_gated = h_fft.clone()
            h_fft_gated[:, :n_active, :] = h_fft[:, :n_active, :] * weights.unsqueeze(-1)  # (batch, fft_len, hidden_channels)
            
            h_fourier = torch.fft.irfft(h_fft_gated, n=h.size(1), dim=1)  # (batch, input_len, hidden_channels)
            h_fourier = fourier.R(h_fourier)  # (batch, input_len, hidden_channels)
            
            # Temporal convolution branch
            h_conv = h.permute(0, 2, 1)  # (batch, hidden_channels, input_len)
            h_conv = tconv(h_conv)  # (batch, hidden_channels, input_len)
            h_conv = h_conv.permute(0, 2, 1)  # (batch, input_len, hidden_channels)
            
            # Combine with residual
            h = norm(h_fourier + h_conv + h_in)  # (batch, input_len, hidden_channels)
        
        # Project to output length
        h = h.permute(0, 2, 1)  # (batch, hidden_channels, input_len)
        h = F.interpolate(h, size=self.output_len, mode='linear', align_corners=False)  # (batch, hidden_channels, output_len)
        h = h.permute(0, 2, 1)  # (batch, output_len, hidden_channels)
        
        # Channel-adaptive output
        gate = self.channel_gate(h)  # (batch, output_len, n_features)
        out = self.output_proj(h)  # (batch, output_len, n_features)
        out = out * gate  # (batch, output_len, n_features)
        
        # Add residual
        out = out + x_direct * 0.1  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)
