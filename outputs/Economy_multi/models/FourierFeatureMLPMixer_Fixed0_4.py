import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierFeatureMLPMixer_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_blocks': [1, 4, 8],
            'ff_dim': [128, 512, 2048],
            'dropout': [0.0, 0.3],
            'n_frequencies': [8, 32, 64],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_blocks = self.config.get('n_blocks', 4)
        self.ff_dim = self.config.get('ff_dim', 512)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.n_frequencies = self.config.get('n_frequencies', 32)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Learnable Fourier frequencies
        self.freqs = nn.Parameter(torch.randn(self.n_frequencies) * 0.1)
        self.freq_phases = nn.Parameter(torch.randn(self.n_frequencies) * 0.1)
        
        # Fourier feature dimension
        fourier_dim = self.n_frequencies * 2  # sin and cos
        
        # Input embedding: project (n_features + fourier_dim) to hidden_dim per time step
        # We concatenate raw features with Fourier features along feature dimension
        self.input_proj = nn.Linear(self.n_features + fourier_dim, self.hidden_dim)
        
        # MLP-Mixer blocks
        self.mixer_blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.mixer_blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(self.hidden_dim),
                'temporal_mlp': nn.Sequential(
                    nn.Linear(self.input_len, self.ff_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.ff_dim, self.input_len),
                ),
                'norm2': nn.LayerNorm(self.hidden_dim),
                'channel_mlp': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.ff_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.ff_dim, self.hidden_dim),
                ),
            }))
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Output head: project from hidden_dim to n_features, then from input_len to output_len
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        self.temporal_proj = nn.Linear(self.input_len, self.output_len)
        
    def compute_fourier_features(self, x):
        # x: (batch, input_len, n_features)
        batch_size, seq_len, n_feat = x.shape
        
        # Time positions
        t = torch.arange(seq_len, device=x.device).float()  # (seq_len,)
        
        # Compute Fourier features: (seq_len, n_frequencies)
        freqs = self.freqs.unsqueeze(0)  # (1, n_frequencies)
        phases = self.freq_phases.unsqueeze(0)  # (1, n_frequencies)
        
        angles = t.unsqueeze(1) * freqs + phases  # (seq_len, n_frequencies)
        
        sin_feat = torch.sin(angles)  # (seq_len, n_frequencies)
        cos_feat = torch.cos(angles)  # (seq_len, n_frequencies)
        fourier = torch.cat([sin_feat, cos_feat], dim=-1)  # (seq_len, n_frequencies*2)
        
        # Expand to batch and features: broadcast across batch and n_features
        fourier = fourier.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, fourier_dim)
        fourier = fourier.expand(batch_size, -1, n_feat, -1)  # (batch, seq_len, n_features, fourier_dim)
        
        # Average across feature dimension to get per-timestep Fourier features
        fourier = fourier.mean(dim=2)  # (batch, seq_len, fourier_dim)
        
        return fourier  # (batch, seq_len, fourier_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Compute Fourier features
        fourier = self.compute_fourier_features(x)  # (batch, input_len, fourier_dim)
        
        # Concatenate raw and Fourier features along feature dimension
        x = torch.cat([x, fourier], dim=-1)  # (batch, input_len, n_features + fourier_dim)
        
        # Project to hidden_dim per time step
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Apply MLP-Mixer blocks
        for block in self.mixer_blocks:
            # Temporal mixing: mix across time dimension (dim 1)
            residual = x  # (batch, input_len, hidden_dim)
            x = block['norm1'](x)  # (batch, input_len, hidden_dim)
            x = x.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
            x = block['temporal_mlp'](x)  # (batch, hidden_dim, input_len)
            x = x.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
            x = residual + x  # (batch, input_len, hidden_dim)
            
            # Channel mixing: mix across hidden_dim (dim 2)
            residual = x  # (batch, input_len, hidden_dim)
            x = block['norm2'](x)  # (batch, input_len, hidden_dim)
            x = block['channel_mlp'](x)  # (batch, input_len, hidden_dim)
            x = residual + x  # (batch, input_len, hidden_dim)
        
        # Output: project from hidden_dim to n_features
        x = self.output_norm(x)  # (batch, input_len, hidden_dim)
        x = self.output_proj(x)  # (batch, input_len, n_features)
        
        # Project from input_len to output_len (temporal dimension)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.temporal_proj(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)

