import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierNeuralOperatorForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'modes': [16, 64, 256],
            'width': [64, 256, 1024],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.modes = self.config.get('modes', 64)
        self.width = self.config.get('width', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to width dimension
        self.fc_in = nn.Linear(self.n_features, self.width)  # (batch, seq, width)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.fourier_layers.append(
                FourierLayer(self.width, self.modes, self.dropout)
            )
            self.w_layers.append(nn.Conv1d(self.width, self.width, 1))
        
        # Output projection
        self.fc_out = nn.Linear(self.width, self.n_features)  # (batch, seq, n_features)
        
        # Decoder: map from input_len to output_len
        self.decoder = nn.Sequential(
            nn.Linear(self.input_len, self.output_len),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to width dimension
        x = self.fc_in(x)  # (batch, input_len, width)
        
        # Fourier integral operators
        for i in range(self.n_layers):
            x1 = self.fourier_layers[i](x)  # (batch, input_len, width)
            x2 = self.w_layers[i](x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, input_len, width)
            x = x + x1 + x2  # (batch, input_len, width)
            x = F.gelu(x)  # (batch, input_len, width)
            x = F.dropout(x, p=self.dropout, training=self.training)  # (batch, input_len, width)
        
        # Project back to n_features
        x = self.fc_out(x)  # (batch, input_len, n_features)
        
        # Decode to output length
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.decoder(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)


class FourierLayer(nn.Module):
    def __init__(self, width, modes, dropout):
        super().__init__()
        self.width = width
        self.modes = modes
        self.scale = 1.0 / (width * width)
        
        # Complex weights for Fourier modes
        self.weights = nn.Parameter(
            self.scale * torch.randn(modes, width, width, 2)
        )  # (modes, width, width, 2) for complex
        self.dropout = dropout
    
    def forward(self, x):
        # x: (batch, seq_len, width)
        batch_size, seq_len, width = x.shape
        
        # FFT along temporal dimension
        x_ft = torch.fft.rfft(x, dim=1)  # (batch, seq_len//2+1, width), complex
        
        # Truncate to modes
        out_ft = torch.zeros_like(x_ft)  # (batch, seq_len//2+1, width), complex
        
        # Multiply relevant Fourier modes by complex weights
        modes = min(self.modes, x_ft.shape[1])
        
        # Create complex weights
        weights_complex = torch.view_as_complex(self.weights[:modes])  # (modes, width, width)
        
        # Apply spectral convolution: (batch, modes, width) @ (modes, width, width) -> (batch, modes, width)
        x_ft_trunc = x_ft[:, :modes, :]  # (batch, modes, width)
        
        # Batch matrix multiply for each mode
        out_ft[:, :modes, :] = torch.einsum(
            'bmw,mwo->bmo',
            x_ft_trunc,
            weights_complex
        )  # (batch, modes, width)
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=seq_len, dim=1)  # (batch, seq_len, width)
        x = F.dropout(x, p=self.dropout, training=self.training)  # (batch, seq_len, width)
        
        return x  # (batch, seq_len, width)
