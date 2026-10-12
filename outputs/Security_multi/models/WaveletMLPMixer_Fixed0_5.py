import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class WaveletMLPMixer_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_levels': [1, 2, 3],
            'hidden_dim': [64, 256, 512],
            'expansion': [2, 4, 8],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_levels = self.config.get('n_levels', 2)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.expansion = self.config.get('expansion', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Calculate wavelet coefficients sizes
        self.level_sizes = []
        size = self.input_len
        for _ in range(self.n_levels):
            size = size // 2
            self.level_sizes.append(size)
        
        self.total_coeffs = sum(self.level_sizes) + size  # detail coeffs + final approximation
        
        # Embed wavelet coefficients
        self.coeff_embed = nn.Linear(self.n_features, self.hidden_dim)
        
        # MLP-Mixer blocks
        self.mixer_blocks = nn.ModuleList()
        n_blocks = 2
        for _ in range(n_blocks):
            # Token mixing: across time/frequency positions (total_coeffs dimension)
            token_mix = nn.Sequential(
                nn.LayerNorm(self.total_coeffs),
                nn.Linear(self.total_coeffs, self.total_coeffs * self.expansion),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.total_coeffs * self.expansion, self.total_coeffs),
                nn.Dropout(self.dropout)
            )
            
            # Channel mixing: across hidden dimensions
            channel_mix = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim * self.expansion),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * self.expansion, self.hidden_dim),
                nn.Dropout(self.dropout)
            )
            
            self.mixer_blocks.append(nn.ModuleDict({
                'token_mix': token_mix,
                'channel_mix': channel_mix
            }))
        
        # Output projection
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
        # Temporal projection: from wavelet domain to output time domain
        self.temporal_proj = nn.Linear(self.total_coeffs, self.output_len)

    def haar_wavelet(self, x):
        """Simple Haar wavelet decomposition. x: (batch, n_features, seq_len)"""
        coeffs = []
        approx = x
        
        for level in range(self.n_levels):
            # Ensure even length
            if approx.shape[-1] % 2 == 1:
                approx = F.pad(approx, (0, 1), mode='replicate')
            
            # Split into even and odd
            even = approx[:, :, 0::2]  # (batch, n_features, seq_len//2)
            odd = approx[:, :, 1::2]   # (batch, n_features, seq_len//2)
            
            # Approximation (low pass) and detail (high pass)
            new_approx = (even + odd) / 2  # (batch, n_features, seq_len//2)
            detail = (even - odd) / 2      # (batch, n_features, seq_len//2)
            
            coeffs.append(detail)
            approx = new_approx
        
        coeffs.append(approx)  # Final approximation
        return coeffs

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Wavelet decomposition
        coeffs = self.haar_wavelet(x)  # List of (batch, n_features, level_size)
        
        # Concatenate all coefficients
        all_coeffs = torch.cat(coeffs, dim=-1)  # (batch, n_features, total_coeffs)
        
        # Embed: (batch, total_coeffs, hidden_dim)
        all_coeffs = all_coeffs.permute(0, 2, 1)  # (batch, total_coeffs, n_features)
        embedded = self.coeff_embed(all_coeffs)  # (batch, total_coeffs, hidden_dim)
        
        # MLP-Mixer blocks
        for block in self.mixer_blocks:
            # Token mixing: operates across total_coeffs (coefficient positions)
            # Need shape (batch, hidden_dim, total_coeffs) so total_coeffs is last for Linear
            h = embedded.permute(0, 2, 1)  # (batch, hidden_dim, total_coeffs)
            h = block['token_mix'](h)  # (batch, hidden_dim, total_coeffs)
            h = h.permute(0, 2, 1)  # (batch, total_coeffs, hidden_dim)
            embedded = embedded + h  # (batch, total_coeffs, hidden_dim)
            
            # Channel mixing: operates across hidden_dim
            h = block['channel_mix'](embedded)  # (batch, total_coeffs, hidden_dim)
            embedded = embedded + h  # (batch, total_coeffs, hidden_dim)
        
        # Project to output: (batch, total_coeffs, hidden_dim) -> (batch, output_len, n_features)
        embedded = embedded.permute(0, 2, 1)  # (batch, hidden_dim, total_coeffs)
        out = self.temporal_proj(embedded)  # (batch, hidden_dim, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        out = self.head(out)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

