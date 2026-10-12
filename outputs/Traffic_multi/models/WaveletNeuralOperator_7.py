import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class WaveletNeuralOperator(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_scales': [2, 4, 6],
            'wavelet_hidden': [32, 128, 512],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 256)
        self.n_scales = cfg.get('n_scales', 4)
        self.wavelet_hidden = cfg.get('wavelet_hidden', 128)
        self.n_layers = cfg.get('n_layers', 3)
        self.dropout = cfg.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Ensure input_len is power of 2 for wavelet
        self.padded_len = 2 ** ((input_len - 1).bit_length())
        self.pad_front = (self.padded_len - input_len) // 2
        self.pad_back = self.padded_len - input_len - self.pad_front
        
        # Learnable wavelet filters (simple Haar-like but learnable)
        self.low_pass = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.high_pass = nn.Parameter(torch.tensor([0.5, -0.5]))
        
        # Input projection
        self.input_proj = nn.Linear(n_features, self.d_model)
        
        # Multi-scale processors
        self.scale_procs = nn.ModuleList()
        for scale in range(self.n_scales):
            scale_layers = []
            for _ in range(self.n_layers):
                scale_layers.append(nn.Sequential(
                    nn.Linear(self.d_model, self.wavelet_hidden),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.wavelet_hidden, self.d_model)
                ))
            self.scale_procs.append(nn.ModuleList(scale_layers))
            
        self.scale_norms = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_layers)])
            for _ in range(self.n_scales)
        ])
        
        # Output projection with temporal upsampling
        self.temporal_proj = nn.Linear(self.padded_len, output_len)
        self.output_proj = nn.Linear(self.d_model, n_features)
        
    def wavelet_decompose(self, x, n_levels):
        # x: (batch, d_model, seq_len)
        coeffs = []
        current = x
        for level in range(n_levels):
            seq_len = current.shape[-1]
            if seq_len < 2:
                break
                
            # Pad for convolution
            current_padded = F.pad(current, (1, 0), mode='circular')  # (batch, d_model, seq_len+1)
            
            # Low-pass and high-pass filtering
            low = F.conv1d(current_padded, self.low_pass.view(1, 1, 2).expand(self.d_model, -1, -1), 
                          groups=self.d_model, stride=2)  # (batch, d_model, seq_len//2)
            high = F.conv1d(current_padded, self.high_pass.view(1, 1, 2).expand(self.d_model, -1, -1),
                           groups=self.d_model, stride=2)  # (batch, d_model, seq_len//2)
            
            coeffs.append(high)
            current = low
            
        coeffs.append(current)  # final approximation
        return coeffs
        
    def wavelet_reconstruct(self, coeffs):
        # Reconstruct from wavelet coefficients
        current = coeffs[-1]  # approximation
        
        for i in range(len(coeffs) - 2, -1, -1):
            detail = coeffs[i]
            
            # Upsample
            batch, d_model, seq_len = current.shape
            current_up = current.repeat_interleave(2, dim=-1)  # (batch, d_model, seq_len*2)
            detail_up = detail.repeat_interleave(2, dim=-1)  # (batch, d_model, seq_len*2)
            
            # Apply synthesis filters (transposed of analysis)
            # Simplified: use learned combination
            current = current_up + detail_up  # simplified reconstruction
            
        return current
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Pad to power of 2
        x = F.pad(x, (0, 0, self.pad_front, self.pad_back))  # (batch, padded_len, n_features)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, padded_len, d_model)
        x = x.permute(0, 2, 1)  # (batch, d_model, padded_len)
        
        # Wavelet decomposition
        coeffs = self.wavelet_decompose(x, self.n_scales)  # list of (batch, d_model, seq_len//2^level)
        
        # Process each scale
        processed_coeffs = []
        for scale_idx, coeff in enumerate(coeffs):
            # Transpose for MLP
            c = coeff.permute(0, 2, 1)  # (batch, seq_len, d_model)
            
            # Apply scale-specific layers
            if scale_idx < len(self.scale_procs):
                for layer_idx, (layer, norm) in enumerate(zip(self.scale_procs[scale_idx], self.scale_norms[scale_idx])):
                    residual = c
                    c = layer(c)  # (batch, seq_len, d_model)
                    c = norm(c)
                    c = c + residual
                    
            processed_coeffs.append(c.permute(0, 2, 1))  # (batch, d_model, seq_len)
            
        # Reconstruct
        x = self.wavelet_reconstruct(processed_coeffs)  # (batch, d_model, padded_len)
        
        # Temporal projection
        x = self.temporal_proj(x)  # (batch, d_model, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, d_model)
        
        # Output projection
        x = self.output_proj(x)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x

