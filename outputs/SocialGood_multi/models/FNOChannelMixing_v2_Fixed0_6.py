import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNOChannelMixing_v2_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.3],
            'coarse_modes': [8, 32, 128],
            'fine_modes': [32, 128, 512],
            'bottleneck_ratio': [2, 4, 8],
            'gate_hidden_dim': [32, 128, 512],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.coarse_modes = self.config.get('coarse_modes', 32)
        self.fine_modes = self.config.get('fine_modes', 128)
        self.bottleneck_ratio = self.config.get('bottleneck_ratio', 4)
        self.gate_hidden_dim = self.config.get('gate_hidden_dim', 128)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # FNO layers with multi-scale spectral processing
        self.fno_layers = nn.ModuleList()
        self.coarse_spectral_convs = nn.ModuleList()
        self.fine_spectral_convs = nn.ModuleList()
        self.channel_gates = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(self.n_layers):
            # Progressive mode reduction with depth
            layer_coarse = max(4, self.coarse_modes // (2 ** (i // 2)))
            layer_fine = max(8, self.fine_modes // (2 ** (i // 2)))
            
            # Spectral convolution weights for coarse (low-freq) path
            self.coarse_spectral_convs.append(
                SpectralConvWeights(self.hidden_dim, layer_coarse)
            )
            # Spectral convolution weights for fine (high-freq) path  
            self.fine_spectral_convs.append(
                SpectralConvWeights(self.hidden_dim, layer_fine)
            )
            
            # Channel-adaptive gating network
            # Computes per-channel weights for blending coarse/fine paths
            gate_input_dim = self.hidden_dim
            self.channel_gates.append(nn.Sequential(
                nn.Linear(gate_input_dim, self.gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.gate_hidden_dim, self.hidden_dim * 2),  # 2 for coarse/fine
            ))
            
            # Layer normalization
            self.norms.append(nn.LayerNorm(self.hidden_dim))
            self.dropouts.append(nn.Dropout(self.dropout_rate))
        
        # Bottleneck compression for output
        bottleneck_dim = max(self.hidden_dim // self.bottleneck_ratio, self.n_features)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )
        
        # Output projection: map from temporal + feature space to prediction
        # We use a learned temporal integration followed by feature projection
        self.temporal_pool = nn.Linear(self.input_len, self.output_len)
        self.output_proj = nn.Linear(bottleneck_dim, self.n_features)
        
    def _spectral_conv(self, x, spectral_conv_module, n_modes):
        """Apply spectral convolution with given weights and mode truncation."""
        # x: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape
        weights = spectral_conv_module.weights  # (hidden_dim, max_modes, 2)
        
        # FFT along time dimension
        x_ft = torch.fft.rfft(x, dim=1)  # (batch, seq_len//2+1, hidden_dim)
        
        # Truncate to specified modes
        out_ft = torch.zeros_like(x_ft)
        modes_to_keep = min(n_modes, x_ft.shape[1])
        
        # Apply learned complex weights
        # weights shape: (hidden_dim, n_modes, 2) for real/imag
        for h in range(hidden_dim):
            if modes_to_keep <= weights.shape[1]:
                w = weights[h, :modes_to_keep, :]  # (modes_to_keep, 2)
            else:
                w = F.pad(weights[h], (0, 0, 0, modes_to_keep - weights.shape[1]))
            
            real_part = x_ft[:, :modes_to_keep, h].real * w[:, 0] - x_ft[:, :modes_to_keep, h].imag * w[:, 1]
            imag_part = x_ft[:, :modes_to_keep, h].real * w[:, 1] + x_ft[:, :modes_to_keep, h].imag * w[:, 0]
            out_ft[:, :modes_to_keep, h] = torch.complex(real_part, imag_part)
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=seq_len, dim=1)  # (batch, seq_len, hidden_dim)
        return x
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # FNO layers with multi-scale spectral processing
        for i in range(self.n_layers):
            residual = x
            
            # Compute channel-adaptive gates from global statistics
            # Use mean pooling over time for stability
            global_feat = x.mean(dim=1)  # (batch, hidden_dim)
            gate_logits = self.channel_gates[i](global_feat)  # (batch, hidden_dim * 2)
            gate_logits = gate_logits.view(batch_size, 2, self.hidden_dim)  # (batch, 2, hidden_dim)
            gates = F.softmax(gate_logits, dim=1)  # (batch, 2, hidden_dim)
            
            # Coarse path: low-frequency modes capture trends
            x_coarse = self._spectral_conv(x, self.coarse_spectral_convs[i], 
                                           self.coarse_modes // (2 ** (i // 2)))
            # (batch, input_len, hidden_dim)
            
            # Fine path: high-frequency modes capture residuals/details
            x_fine = self._spectral_conv(x, self.fine_spectral_convs[i],
                                         self.fine_modes // (2 ** (i // 2)))
            # (batch, input_len, hidden_dim)
            
            # Apply channel-adaptive gating
            # gates[:, 0:1, :] for coarse, gates[:, 1:2, :] for fine
            x = gates[:, 0:1, :] * x_coarse + gates[:, 1:2, :] * x_fine
            # (batch, input_len, hidden_dim)
            
            # Residual connection and normalization
            x = self.norms[i](x + residual)  # (batch, input_len, hidden_dim)
            x = self.dropouts[i](x)  # (batch, input_len, hidden_dim)
        
        # Bottleneck compression
        x = self.bottleneck(x)  # (batch, input_len, bottleneck_dim)
        
        # Temporal projection: map input_len to output_len
        x = x.permute(0, 2, 1)  # (batch, bottleneck_dim, input_len)
        x = self.temporal_pool(x)  # (batch, bottleneck_dim, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, bottleneck_dim)
        
        # Feature projection
        x = self.output_proj(x)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x


class SpectralConvWeights(nn.Module):
    """Wrapper module to hold spectral convolution weights as a proper nn.Module."""
    def __init__(self, hidden_dim, n_modes):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(hidden_dim, n_modes, 2) * 0.02)
    
    def forward(self, x):
        # This module is not used directly in forward; weights are accessed directly
        return x
