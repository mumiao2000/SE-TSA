import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AttentionHybridFNO_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_modes': [8, 32, 128],
            'n_fno_layers': [1, 2, 4],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_modes = self.config.get('n_modes', 32)
        self.n_fno_layers = self.config.get('n_fno_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Ensure d_model divisible by n_heads
        self.d_model = (self.d_model // self.n_heads) * self.n_heads
        if self.d_model == 0:
            self.d_model = self.n_heads
        
        # Hybrid normalization: RevIN + feature-wise scale
        self.revin = RevIN(self.n_features, affine=True)
        self.feature_scale = nn.Parameter(torch.ones(self.n_features))
        self.feature_shift = nn.Parameter(torch.zeros(self.n_features))
        
        # Input projection to d_model per feature (treat features as tokens)
        self.input_proj = nn.Linear(self.input_len, self.d_model)
        
        # Feature-wise self-attention (attention across features, not time)
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.d_model)
        
        # Custom module to wrap spectral weights so they work in ModuleDict
        class SpectralWeight(nn.Module):
            def __init__(self, d_model, n_modes):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(d_model, d_model, n_modes, 2) * 0.02
                )
            
            def forward(self):
                return self.weight
        
        # FNO layers for temporal modeling
        self.fno_temporal = nn.ModuleList()
        for _ in range(self.n_fno_layers):
            self.fno_temporal.append(
                nn.ModuleDict({
                    'spectral': SpectralWeight(self.d_model, self.n_modes),
                    'norm': nn.LayerNorm(self.d_model),
                    'dropout': nn.Dropout(self.dropout)
                })
            )
        
        # Cross-temporal attention for output generation
        self.output_queries = nn.Parameter(torch.randn(self.output_len, self.d_model))
        self.cross_temporal_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Output projection to all features
        self.output_proj = nn.Linear(self.d_model, self.n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # Hybrid normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        x = x * self.feature_scale.unsqueeze(0).unsqueeze(0) + self.feature_shift.unsqueeze(0).unsqueeze(0)  # (batch, input_len, n_features)
        
        # Transpose to treat features as sequence: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project each feature's time series to d_model
        x = self.input_proj(x)  # (batch, n_features, d_model)
        
        # Feature-wise self-attention: features attend to each other
        attn_out, _ = self.feature_attn(x, x, x)  # (batch, n_features, d_model)
        x = self.attn_norm(x + attn_out)  # (batch, n_features, d_model)
        
        # Apply FNO along n_features dimension (spatial frequency across features)
        for layer in self.fno_temporal:
            x_fft = torch.fft.rfft(x, dim=1)  # (batch, n_features//2+1, d_model)
            n_fft = min(self.n_modes, x_fft.size(1))
            
            weight = layer['spectral'].weight[:, :, :n_fft, :]  # (d_model, d_model, n_fft, 2)
            
            x_fft_real = x_fft[:, :n_fft, :].real  # (batch, n_fft, d_model)
            x_fft_imag = x_fft[:, :n_fft, :].imag  # (batch, n_fft, d_model)
            
            out_real = torch.einsum('bti,iof->bto', x_fft_real, weight[..., 0]) - \
                       torch.einsum('bti,iof->bto', x_fft_imag, weight[..., 1])  # (batch, n_fft, d_model)
            out_imag = torch.einsum('bti,iof->bto', x_fft_real, weight[..., 1]) + \
                       torch.einsum('bti,iof->bto', x_fft_imag, weight[..., 0])  # (batch, n_fft, d_model)
            
            out_fft = torch.complex(out_real, out_imag)  # (batch, n_fft, d_model)
            
            if n_fft < x_fft.size(1):
                padding = torch.zeros(x_fft.size(0), x_fft.size(1) - n_fft, self.d_model,
                                     dtype=out_fft.dtype, device=out_fft.device)  # (batch, pad, d_model)
                out_fft = torch.cat([out_fft, padding], dim=1)  # (batch, n_features//2+1, d_model)
            
            x_spec = torch.fft.irfft(out_fft, n=x.size(1), dim=1)  # (batch, n_features, d_model)
            x = x + layer['dropout'](layer['norm'](x_spec))  # (batch, n_features, d_model)
        
        # Cross-temporal attention: use learned queries for output timesteps
        queries = self.output_queries.unsqueeze(0).expand(x.size(0), -1, -1)  # (batch, output_len, d_model)
        
        # Cross-attention from queries to feature representation
        cross_out, _ = self.cross_temporal_attn(queries, x, x)  # (batch, output_len, d_model)
        
        # Project to all features
        out = self.output_proj(cross_out)  # (batch, output_len, n_features)
        
        # Apply feature-specific modulation
        out = out * self.feature_scale.unsqueeze(0).unsqueeze(0)  # (batch, output_len, n_features)
        
        # Denormalize with RevIN
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)

