import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierFilterMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [32, 128, 512],
            'n_bands': [2, 3, 5],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 128)
        self.n_bands = self.config.get('n_bands', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Frequency band boundaries (as fraction of Nyquist)
        # For input_len=8, we have 4 meaningful frequencies (0 to 3)
        # Band 0: DC component (trend)
        # Band 1: low frequencies (slow cycles)
        # Band 2+: higher frequencies
        
        # Learnable band masks - one per feature
        self.band_masks = nn.Parameter(torch.ones(n_features, self.n_bands))
        
        # MLP for each band
        self.band_mlps = nn.ModuleList()
        for b in range(self.n_bands):
            self.band_mlps.append(nn.Sequential(
                nn.Linear(input_len, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, output_len)
            ))
        
        # Feature-wise mixing after band recombination
        self.mixer = nn.Sequential(
            nn.Linear(output_len * self.n_bands, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, output_len)
        )
        
    def decompose_bands(self, x):
        # x: (batch, n_features, input_len)
        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)  # (batch, n_features, input_len//2 + 1)
        
        # For input_len=8, rfft gives 5 coefficients (0, 1, 2, 3, 4)
        n_freq = x_fft.shape[-1]  # input_len//2 + 1
        
        # Create band masks in frequency domain
        bands = []
        freqs_per_band = max(1, n_freq // self.n_bands)
        
        for b in range(self.n_bands):
            start = b * freqs_per_band
            end = min((b + 1) * freqs_per_band, n_freq)
            if b == self.n_bands - 1:
                end = n_freq  # last band gets remainder
            
            # Create mask
            mask = torch.zeros(n_freq, device=x.device)
            mask[start:end] = 1.0
            
            # Apply mask and inverse FFT
            x_band_fft = x_fft * mask.view(1, 1, -1)  # (batch, n_features, n_freq)
            x_band = torch.fft.irfft(x_band_fft, n=x.shape[-1], dim=-1)  # (batch, n_features, input_len)
            bands.append(x_band)
        
        return bands  # list of (batch, n_features, input_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose
        x_t = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Decompose into frequency bands
        bands = self.decompose_bands(x_t)  # list of (batch, n_features, input_len)
        
        # Process each band
        band_outputs = []
        for b, (band, mlp) in enumerate(zip(bands, self.band_mlps)):
            # Apply learnable feature-specific mask
            mask = F.softmax(self.band_masks[:, b], dim=0)  # (n_features,)
            masked_band = band * mask.view(1, -1, 1)  # (batch, n_features, input_len)
            out = mlp(masked_band)  # (batch, n_features, output_len)
            band_outputs.append(out)
        
        # Concatenate band outputs
        combined = torch.cat(band_outputs, dim=-1)  # (batch, n_features, output_len * n_bands)
        
        # Final mixing
        out = self.mixer(combined)  # (batch, n_features, output_len)
        
        # Transpose back
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
