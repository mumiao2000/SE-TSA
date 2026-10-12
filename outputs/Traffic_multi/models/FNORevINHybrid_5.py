import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNORevINHybrid(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'modes': [16, 32, 64],
            'width': [64, 256, 1024],
            'n_layers': [4, 8, 12],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.modes = self.config.get('modes', 32)
        self.width = self.config.get('width', 256)
        self.n_layers = self.config.get('n_layers', 8)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Lift input to channel dimension
        self.lift = nn.Linear(self.n_features, self.width)  # (batch, seq_len, width)
        
        # Spectral convolution layers
        self.fno_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.fno_layers.append(SpectralConv1d(self.width, self.width, self.modes))
        
        # Pointwise (skip) connections
        self.w_layers = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1) for _ in range(self.n_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.width) for _ in range(self.n_layers)
        ])
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Project to output: need to map from input_len to output_len
        # Use learned interpolation or linear projection
        self.project = nn.Sequential(
            nn.Linear(self.width, self.width * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.width * 2, self.width)
        )
        
        # Final output projection
        self.output_proj = nn.Linear(self.width, self.n_features)
        
        # Learned temporal upsampling/downsampling
        self.temporal_proj = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Lift: (batch, input_len, width)
        x = self.lift(x)  # (batch, input_len, width)
        
        # Apply FNO layers
        for i, (fno, w, norm) in enumerate(zip(self.fno_layers, self.w_layers, self.norms)):
            # FNO branch: need (batch, width, seq_len) for spectral conv
            x_fno = x.permute(0, 2, 1)  # (batch, width, input_len)
            x_fno = fno(x_fno)  # (batch, width, input_len)
            x_fno = x_fno.permute(0, 2, 1)  # (batch, input_len, width)
            
            # Skip connection
            x_skip = x.permute(0, 2, 1)  # (batch, width, input_len)
            x_skip = w(x_skip)  # (batch, width, input_len)
            x_skip = x_skip.permute(0, 2, 1)  # (batch, input_len, width)
            
            # Residual and norm
            x = norm(x_fno + x_skip)  # (batch, input_len, width)
            x = F.gelu(x)  # (batch, input_len, width)
            x = self.dropout(x)  # (batch, input_len, width)
        
        # Project features
        x = self.project(x)  # (batch, input_len, width)
        
        # Temporal projection: (batch, input_len, width) -> (batch, output_len, width)
        x = x.permute(0, 2, 1)  # (batch, width, input_len)
        x = self.temporal_proj(x)  # (batch, width, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, width)
        
        # Final projection to features
        x = self.output_proj(x)  # (batch, output_len, n_features)
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, 2)
        )  # Complex weights: (in, out, modes, 2) for real/imag
    
    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        batchsize = x.shape[0]
        seq_len = x.shape[-1]
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, in_channels, seq_len//2+1)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-1), 
                             dtype=torch.cfloat, device=x.device)
        # (batch, out_channels, seq_len//2+1)
        
        # Only keep first 'modes' frequencies
        modes = min(self.modes, x_ft.size(-1))
        
        # Complex multiplication
        for i in range(modes):
            # weights: (in_channels, out_channels, 2)
            w_real = self.weights[:, :, i, 0]  # (in_channels, out_channels)
            w_imag = self.weights[:, :, i, 1]  # (in_channels, out_channels)
            
            # x_ft[:, :, i]: (batch, in_channels)
            real_part = torch.matmul(x_ft[:, :, i].real, w_real) - torch.matmul(x_ft[:, :, i].imag, w_imag)
            imag_part = torch.matmul(x_ft[:, :, i].real, w_imag) + torch.matmul(x_ft[:, :, i].imag, w_real)
            
            out_ft[:, :, i] = torch.complex(real_part, imag_part)  # (batch, out_channels)
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=seq_len, dim=-1)  # (batch, out_channels, seq_len)
        return x
