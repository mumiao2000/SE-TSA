import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SpectralHybridLiteV2_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_modes': [4, 8, 16],
            'bottleneck_ratio': [2, 4, 8],
            'dropout': [0.0, 0.3],
            'mlp_depth': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_modes = self.config.get('n_modes', 8)
        self.bottleneck_ratio = self.config.get('bottleneck_ratio', 4)
        self.dropout_rate = self.config.get('dropout', 0.3)
        self.mlp_depth = self.config.get('mlp_depth', 4)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: (batch, input_len, n_features) -> (batch, input_len, hidden_dim)
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Per-channel spectral weights: each channel has independent low-frequency modes
        # Shape: (n_modes, n_features) for real/imag parts - will be sliced to (n_active, n_features)
        self.spectral_real = nn.Parameter(torch.randn(self.n_modes, self.n_features) * 0.02)
        self.spectral_imag = nn.Parameter(torch.randn(self.n_modes, self.n_features) * 0.02)
        
        # Spectral gating bias
        self.spectral_bias = nn.Parameter(torch.zeros(self.n_features))
        
        # Bottleneck dimension for channel mixing
        self.bottleneck_dim = max(self.hidden_dim // self.bottleneck_ratio, 16)
        
        # Post-spectral MLP: deep bottlenecked network for channel mixing
        mlp_layers = []
        mlp_layers.append(nn.Linear(self.hidden_dim, self.bottleneck_dim))
        mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Dropout(self.dropout_rate))
        
        for _ in range(self.mlp_depth - 2):
            mlp_layers.append(nn.Linear(self.bottleneck_dim, self.bottleneck_dim))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(self.dropout_rate))
        
        mlp_layers.append(nn.Linear(self.bottleneck_dim, self.hidden_dim))
        mlp_layers.append(nn.Dropout(self.dropout_rate))
        
        self.post_mlp = nn.Sequential(*mlp_layers)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        
        # Output projection: hidden_dim -> n_features
        # We need to map from input_len to output_len first, then project features
        self.temporal_fc = nn.Linear(self.input_len, self.output_len)
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store residual for skip connection
        x_input = x  # (batch, input_len, n_features)
        
        # Input projection
        h = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # === Spectral processing: per-channel FFT with truncated modes ===
        # Process each feature channel independently in frequency domain
        
        # Per-channel FFT on original features
        x_fft = torch.fft.rfft(x, dim=1)  # (batch, n_fft, n_features)
        
        # Truncate to n_modes
        n_fft = x_fft.shape[1]
        n_active = min(self.n_modes, n_fft)
        
        x_fft_trunc = x_fft[:, :n_active, :]  # (batch, n_active, n_features)
        
        # Apply learned spectral weights (complex multiplication)
        # Build complex weights of shape (n_active, n_features)
        spectral_weight = torch.view_as_complex(
            torch.stack([
                self.spectral_real[:n_active, :],  # (n_active, n_features)
                self.spectral_imag[:n_active, :]   # (n_active, n_features)
            ], dim=-1)
        )  # (n_active, n_features)
        
        # Add dimensions for broadcasting: (1, n_active, n_features)
        spectral_weight = spectral_weight.unsqueeze(0)  # (1, n_active, n_features)
        
        # Apply spectral gating: multiply in frequency domain
        # x_fft_trunc: (batch, n_active, n_features)
        # spectral_weight: (1, n_active, n_features) -> broadcasts to (batch, n_active, n_features)
        x_fft_gated = x_fft_trunc * spectral_weight  # (batch, n_active, n_features)
        
        # Pad back to full FFT size
        x_fft_out = torch.zeros_like(x_fft)  # (batch, n_fft, n_features)
        x_fft_out[:, :n_active, :] = x_fft_gated  # (batch, n_fft, n_features)
        
        # Inverse FFT back to time domain
        x_spectral = torch.fft.irfft(x_fft_out, n=self.input_len, dim=1)  # (batch, input_len, n_features)
        
        # Add spectral bias and residual
        x_spectral = x_spectral + self.spectral_bias.view(1, 1, self.n_features)  # (batch, input_len, n_features)
        x_spectral = x_spectral + x_input  # (batch, input_len, n_features) - residual
        
        # Re-project to hidden_dim for MLP processing
        h_spectral = self.input_proj(x_spectral)  # (batch, input_len, hidden_dim)
        
        # First norm
        h_spectral = self.norm1(h_spectral)  # (batch, input_len, hidden_dim)
        
        # === Bottlenecked MLP for channel mixing ===
        # Process each timestep independently with shared MLP
        h_mlp = self.post_mlp(h_spectral)  # (batch, input_len, hidden_dim)
        
        # Residual connection
        h_out = h_spectral + h_mlp  # (batch, input_len, hidden_dim)
        
        # Second norm
        h_out = self.norm2(h_out)  # (batch, input_len, hidden_dim)
        
        # === Temporal mapping: input_len -> output_len ===
        # Permute for linear layer: (batch, hidden_dim, input_len)
        h_perm = h_out.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        
        # Temporal FC
        h_temporal = self.temporal_fc(h_perm)  # (batch, hidden_dim, output_len)
        
        # Permute back
        h_temporal = h_temporal.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Output projection to n_features
        out = self.output_proj(h_temporal)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

