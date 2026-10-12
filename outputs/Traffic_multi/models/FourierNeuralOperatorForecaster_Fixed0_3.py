import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierNeuralOperatorForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_fno_layers': [1, 4, 8],
            'n_modes': [8, 32, 128],
            'dropout': [0.0, 0.3],
            'ff_mult': [2, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_fno_layers = self.config.get('n_fno_layers', 4)
        self.n_modes = self.config.get('n_modes', 32)
        self.dropout = self.config.get('dropout', 0.1)
        self.ff_mult = self.config.get('ff_mult', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # FNO blocks
        self.fno_blocks = nn.ParameterList()  # Fixed: use ParameterList for Parameters
        self.fno_norms = nn.ModuleList()
        self.fno_ffns = nn.ModuleList()
        self.fno_ffn_norms = nn.ModuleList()
        
        for _ in range(self.n_fno_layers):
            # Spectral convolution weights (complex)
            self.fno_blocks.append(
                nn.Parameter(torch.randn(self.n_modes, self.hidden_dim, self.hidden_dim, 2) * 0.02)
            )
            self.fno_norms.append(nn.LayerNorm(self.hidden_dim))
            self.fno_ffns.append(nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * self.ff_mult),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * self.ff_mult, self.hidden_dim),
                nn.Dropout(self.dropout)
            ))
            self.fno_ffn_norms.append(nn.LayerNorm(self.hidden_dim))
        
        # Output projection: from input_len to output_len
        # We use a learned temporal upsampling
        self.temporal_upsample = nn.Sequential(
            nn.Linear(self.input_len, self.input_len + self.output_len),
            nn.GELU(),
            nn.Linear(self.input_len + self.output_len, self.output_len)
        )
        
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden dim
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # FNO layers
        for i in range(self.n_fno_layers):
            residual = x  # (batch, input_len, hidden_dim)
            
            # FFT
            x_ft = torch.fft.rfft(x, dim=1)  # (batch, input_len//2+1, hidden_dim), complex
            
            # Get the weight and apply as complex multiplication
            weight = torch.view_as_complex(self.fno_blocks[i])  # (n_modes, hidden_dim, hidden_dim)
            
            # Truncate or pad to match n_modes
            out_ft = torch.zeros_like(x_ft)
            modes = min(self.n_modes, x_ft.shape[1])
            
            # Apply learned filter to low-frequency modes
            # x_ft[:, :modes, :]: (batch, modes, hidden_dim)
            # weight[:modes, :, :]: (modes, hidden_dim, hidden_dim)
            out_ft[:, :modes, :] = torch.einsum('bmi,mio->bmo', x_ft[:, :modes, :], weight[:modes, :, :])
            
            # Inverse FFT
            x = torch.fft.irfft(out_ft, n=self.input_len, dim=1)  # (batch, input_len, hidden_dim)
            
            # Add residual and normalize
            x = self.fno_norms[i](x + residual)  # (batch, input_len, hidden_dim)
            
            # FFN
            ff_residual = x  # (batch, input_len, hidden_dim)
            x = self.fno_ffns[i](x)  # (batch, input_len, hidden_dim)
            x = self.fno_ffn_norms[i](x + ff_residual)  # (batch, input_len, hidden_dim)
        
        # Temporal upsampling: project from input_len to output_len
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        x = self.temporal_upsample(x)  # (batch, hidden_dim, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Project to output features
        out = self.output_proj(x)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

