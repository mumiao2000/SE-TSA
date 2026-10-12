import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SpectralMLPMixer_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'patch_size': [2, 8, 24],
            'hidden_dim': [64, 256, 512],
            'expansion_factor': [2, 4, 8],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.patch_size = self.config.get('patch_size', 8)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.expansion_factor = self.config.get('expansion_factor', 4)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Ensure patch size divides input length
        self.n_patches = self.input_len // self.patch_size
        self.actual_input_len = self.n_patches * self.patch_size
        
        # Patch embedding: linear projection of each patch
        self.patch_embed = nn.Linear(self.patch_size, self.hidden_dim)
        
        # Spectral MLP mixer layers
        self.mixer_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.mixer_layers.append(nn.ModuleDict({
                # Token mixing in spectral domain
                'token_norm': nn.LayerNorm(self.hidden_dim),
                'token_mlp': nn.Sequential(
                    nn.Linear(self.n_patches, self.n_patches * self.expansion_factor),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.n_patches * self.expansion_factor, self.n_patches),
                    nn.Dropout(self.dropout)
                ),
                # Channel mixing
                'channel_norm': nn.LayerNorm(self.hidden_dim),
                'channel_mlp': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * self.expansion_factor),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim * self.expansion_factor, self.hidden_dim),
                    nn.Dropout(self.dropout)
                )
            }))
        
        # Output head: from (n_features, hidden_dim) to (n_features, output_len)
        self.head = nn.Linear(self.hidden_dim, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Truncate to multiple of patch_size
        x = x[:, :self.actual_input_len, :]  # (batch, actual_input_len, n_features)
        
        # Patch: (batch, n_patches, patch_size, n_features)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.n_patches, self.patch_size, self.n_features)
        x = x.permute(0, 3, 1, 2)  # (batch, n_features, n_patches, patch_size)
        
        # Embed each patch: treat n_features as separate channels
        x = x.reshape(batch_size * self.n_features, self.n_patches, self.patch_size)
        # (batch*n_features, n_patches, patch_size)
        x = self.patch_embed(x)  # (batch*n_features, n_patches, hidden_dim)
        
        # Mixer layers
        for layer in self.mixer_layers:
            # Token mixing with spectral preprocessing
            residual = x  # (batch*n_features, n_patches, hidden_dim)
            x = layer['token_norm'](x)  # (batch*n_features, n_patches, hidden_dim)
            
            # Apply FFT-based token mixing: transpose to (batch*n_features, hidden_dim, n_patches)
            x_t = x.permute(0, 2, 1)  # (batch*n_features, hidden_dim, n_patches)
            
            # Spectral gating: compute magnitude spectrum and use as gating signal
            x_fft = torch.fft.rfft(x_t, dim=-1, norm='ortho')  # (batch*n_features, hidden_dim, n_patches//2+1)
            x_fft_mag = torch.abs(x_fft)  # magnitude for real MLP  # (batch*n_features, hidden_dim, n_patches//2+1)
            
            # Pad or truncate to match n_patches for gating
            fft_len = x_fft_mag.shape[-1]
            if fft_len < self.n_patches:
                x_fft_mag = F.pad(x_fft_mag, (0, self.n_patches - fft_len))  # (batch*n_features, hidden_dim, n_patches)
            else:
                x_fft_mag = x_fft_mag[..., :self.n_patches]  # (batch*n_features, hidden_dim, n_patches)
            
            # MLP on tokens
            x_mlp = layer['token_mlp'](x_t)  # (batch*n_features, hidden_dim, n_patches)
            # Apply spectral gating
            gate = torch.sigmoid(x_fft_mag)  # (batch*n_features, hidden_dim, n_patches)
            x_mlp = x_mlp * gate  # (batch*n_features, hidden_dim, n_patches)
            
            x = x_mlp.permute(0, 2, 1)  # (batch*n_features, n_patches, hidden_dim)
            x = x + residual  # (batch*n_features, n_patches, hidden_dim)
            
            # Channel mixing
            residual = x  # (batch*n_features, n_patches, hidden_dim)
            x = layer['channel_norm'](x)  # (batch*n_features, n_patches, hidden_dim)
            x = layer['channel_mlp'](x)  # (batch*n_features, n_patches, hidden_dim)
            x = x + residual  # (batch*n_features, n_patches, hidden_dim)
        
        # Output: aggregate patches and project to output length
        x = x.mean(dim=1)  # (batch*n_features, hidden_dim)
        x = self.head(x)  # (batch*n_features, output_len)
        
        # Reshape back
        x = x.reshape(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x

