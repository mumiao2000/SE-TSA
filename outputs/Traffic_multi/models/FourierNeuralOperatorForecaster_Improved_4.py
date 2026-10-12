import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierNeuralOperatorForecaster_Improved(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_layers': [2, 4, 8],
            'n_modes': [16, 32, 64],
            'dropout': [0.0, 0.1, 0.3],
            'expansion_factor': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.n_modes = self.config.get('n_modes', 32)
        self.dropout = self.config.get('dropout', 0.1)
        self.expansion_factor = self.config.get('expansion_factor', 4)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: (batch, input_len, n_features) -> (batch, input_len, hidden_dim)
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Fourier layers with channel mixing and gated MLPs
        self.fourier_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()
        
        for i in range(self.n_layers):
            # Hierarchical mode decay: deeper layers use fewer modes
            layer_modes = max(self.n_modes // (2 ** (i // 2)), 4)
            
            # Spectral convolution with channel mixing
            self.fourier_layers.append(
                SpectralConv1dChannelMixing(
                    self.hidden_dim, 
                    layer_modes,
                    self.n_features
                )
            )
            
            # Gated MLP (GLU-style): expansion then gate
            mlp_hidden = self.hidden_dim * self.expansion_factor
            self.mlp_layers.append(
                GatedMLP(self.hidden_dim, mlp_hidden, self.dropout)
            )
            
            # Layer norms
            self.norm1_layers.append(nn.LayerNorm(self.hidden_dim))
            self.norm2_layers.append(nn.LayerNorm(self.hidden_dim))
        
        # Output projection with skip from input encoding
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # *2 for skip connection
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
        # Temporal interpolation for variable length forecasting
        # Use learned positional encoding for output positions
        self.output_positions = nn.Parameter(torch.randn(self.output_len, self.hidden_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store input for skip connection
        x_input = x  # (batch, input_len, n_features)
        
        # Input projection
        h = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Fourier layers with residual connections
        for i in range(self.n_layers):
            # Fourier block
            h_norm = self.norm1_layers[i](h)  # (batch, input_len, hidden_dim)
            h_fourier = self.fourier_layers[i](h_norm)  # (batch, input_len, hidden_dim)
            h = h + h_fourier  # (batch, input_len, hidden_dim)
            
            # Gated MLP block
            h_norm = self.norm2_layers[i](h)  # (batch, input_len, hidden_dim)
            h_mlp = self.mlp_layers[i](h_norm)  # (batch, input_len, hidden_dim)
            h = h + h_mlp  # (batch, input_len, hidden_dim)
        
        # Encode full sequence for global context
        h_global = h.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        h_global = h_global.expand(-1, self.output_len, -1)  # (batch, output_len, hidden_dim)
        
        # Add learned positional encoding for output
        h_pos = h_global + self.output_positions.unsqueeze(0)  # (batch, output_len, hidden_dim)
        
        # Interpolate from input sequence to output length using attention-like pooling
        # Compute attention weights between output positions and input positions
        attn_scores = torch.matmul(h_pos, h.transpose(1, 2))  # (batch, output_len, input_len)
        attn_weights = F.softmax(attn_scores / (self.hidden_dim ** 0.5), dim=-1)  # (batch, output_len, input_len)
        h_attended = torch.matmul(attn_weights, h)  # (batch, output_len, hidden_dim)
        
        # Concatenate with global context for skip
        h_combined = torch.cat([h_attended, h_global], dim=-1)  # (batch, output_len, hidden_dim * 2)
        
        # Output projection
        out = self.output_proj(h_combined)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out


class SpectralConv1dChannelMixing(nn.Module):
    """Spectral convolution with channel mixing across features."""
    
    def __init__(self, hidden_dim, n_modes, n_features):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modes = n_modes
        self.n_features = n_features
        
        # Complex weights for frequency domain: (n_modes, hidden_dim, hidden_dim)
        # Using real and imag separately for stability
        self.weights_real = nn.Parameter(
            torch.randn(n_modes, hidden_dim, hidden_dim) * (1.0 / hidden_dim)
        )
        self.weights_imag = nn.Parameter(
            torch.randn(n_modes, hidden_dim, hidden_dim) * (1.0 / hidden_dim)
        )
        
        # Channel mixing for cross-feature interaction
        self.feature_mix = nn.Linear(n_features, n_features)
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.shape
        
        # FFT to frequency domain
        x_ft = torch.fft.rfft(x, dim=1)  # (batch, seq_len//2+1, hidden_dim)
        
        # Get relevant modes
        out_ft = torch.zeros_like(x_ft)
        n_active = min(self.n_modes, x_ft.shape[1])
        
        # Apply learned spectral weights with channel mixing
        # For each frequency mode, apply linear transformation
        for i in range(n_active):
            # Get complex weights
            weight = torch.complex(self.weights_real[i], self.weights_imag[i])  # (hidden_dim, hidden_dim)
            # Apply to this frequency: (batch, hidden_dim) @ (hidden_dim, hidden_dim)
            out_ft[:, i, :] = torch.matmul(x_ft[:, i, :].real, weight.real) - \
                              torch.matmul(x_ft[:, i, :].imag, weight.imag)
            # Imaginary part
            out_ft[:, i, :] = out_ft[:, i, :] + 1j * (
                torch.matmul(x_ft[:, i, :].real, weight.imag) + 
                torch.matmul(x_ft[:, i, :].imag, weight.real)
            )
        
        # Inverse FFT back to time domain
        x = torch.fft.irfft(out_ft, n=seq_len, dim=1)  # (batch, seq_len, hidden_dim)
        
        return x


class GatedMLP(nn.Module):
    """Gated MLP with GLU-style gating for expressive nonlinear transformations."""
    
    def __init__(self, hidden_dim, mlp_hidden, dropout):
        super().__init__()
        # GLU: split into gate and value, then multiply
        self.fc_in = nn.Linear(hidden_dim, mlp_hidden * 2)
        self.fc_out = nn.Linear(mlp_hidden, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        x_proj = self.fc_in(x)  # (batch, seq_len, mlp_hidden * 2)
        
        # Split into gate and value
        gate, value = x_proj.chunk(2, dim=-1)  # each (batch, seq_len, mlp_hidden)
        
        # GLU: gate with sigmoid, multiply with value
        x_gated = F.sigmoid(gate) * value  # (batch, seq_len, mlp_hidden)
        x_gated = self.dropout(x_gated)
        
        # Project back
        out = self.fc_out(x_gated)  # (batch, seq_len, hidden_dim)
        
        return out
