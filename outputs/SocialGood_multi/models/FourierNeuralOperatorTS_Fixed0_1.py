import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierNeuralOperatorTS_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'modes': [8, 32, 128],
            'width': [32, 128, 512],
            'n_fno_layers': [2, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.modes = self.config.get('modes', 32)
        self.width = self.config.get('width', 128)
        self.n_fno_layers = self.config.get('n_fno_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Lifting: project to higher dimensional channel space
        self.lifting = nn.Linear(self.n_features, self.width)
        
        # Fourier layers - parameters stored directly, modules in ModuleDict
        self.fourier_weights = nn.ParameterList()
        self.fourier_biases = nn.ParameterList()
        self.freq_gates = nn.ParameterList()
        self.fourier_modules = nn.ModuleList()
        
        for i in range(self.n_fno_layers):
            # Parameters (not modules)
            self.fourier_weights.append(nn.Parameter(torch.randn(self.width, self.width, self.modes, 2) * 0.02))
            self.fourier_biases.append(nn.Parameter(torch.zeros(self.width)))
            self.freq_gates.append(nn.Parameter(torch.ones(self.modes)))
            
            # Actual modules
            self.fourier_modules.append(nn.ModuleDict({
                'w': nn.Conv1d(self.width, self.width, 1),
                'norm': nn.LayerNorm(self.width),
                'dropout': nn.Dropout(self.dropout_rate)
            }))
        
        # Time embedding for position information
        self.time_embed = nn.Parameter(torch.randn(1, self.input_len, self.width) * 0.02)
        
        # Output: project back and decode to prediction length
        self.projection = nn.Sequential(
            nn.Linear(self.width, self.width // 2),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.width // 2, self.n_features)
        )
        
        # Direct path for high frequencies
        self.high_freq_conv = nn.Conv1d(self.width, self.width, 3, padding=1)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Lifting
        x = self.lifting(x)  # (batch, input_len, width)
        x = x + self.time_embed  # (batch, input_len, width)
        
        # Fourier layers
        for i in range(self.n_fno_layers):
            x_in = x
            layer_modules = self.fourier_modules[i]
            weight = self.fourier_weights[i]
            bias = self.fourier_biases[i]
            freq_gate = self.freq_gates[i]
            
            # FFT to frequency domain
            x_ft = torch.fft.rfft(x, dim=1, norm='ortho')  # (batch, input_len//2+1, width)
            
            # Get dimensions
            batch_size, n_freq, ch = x_ft.shape
            
            # Keep only first self.modes frequencies
            n_modes = min(self.modes, n_freq)
            x_ft_trunc = x_ft[:, :n_modes, :]  # (batch, n_modes, width)
            
            # Apply learned spectral weights
            # weight: (width, width, modes, 2) -> complex weights
            weight_complex = torch.view_as_complex(weight)  # (width, width, modes)
            
            # Apply frequency gate
            gate = torch.sigmoid(freq_gate[:n_modes])  # (n_modes,)
            
            # Spectral convolution: for each mode, linear transform of channels
            out_ft = torch.zeros_like(x_ft_trunc)
            for mode_idx in range(n_modes):
                # x_ft_trunc[:, mode_idx, :]: (batch, width)
                # weight_complex[:, :, mode_idx]: (width, width)
                mode_input = x_ft_trunc[:, mode_idx, :]  # (batch, width)
                mode_weight = weight_complex[:, :, mode_idx]  # (width, width)
                mode_out = torch.matmul(mode_input, mode_weight.T)  # (batch, width)
                out_ft[:, mode_idx, :] = gate[mode_idx] * mode_out
            
            # Add bias in frequency
            out_ft = out_ft + bias.unsqueeze(0).unsqueeze(0)  # (batch, n_modes, width)
            
            # Pad back to full frequency range and IFFT
            x_ft_full = torch.zeros_like(x_ft)
            x_ft_full[:, :n_modes, :] = out_ft
            x_fourier = torch.fft.irfft(x_ft_full, n=self.input_len, dim=1, norm='ortho')  # (batch, input_len, width)
            
            # Direct path (W in FNO paper)
            x_direct = layer_modules['w'](x_in.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, input_len, width)
            
            # Combine with residual
            x = layer_modules['norm'](x_fourier + x_direct + x_in)  # (batch, input_len, width)
            x = layer_modules['dropout'](x)
        
        # Decode to output length using linear interpolation in time domain
        x = x.permute(0, 2, 1)  # (batch, width, input_len)
        # Interpolate to output_len
        x = F.interpolate(x, size=self.output_len, mode='linear', align_corners=False)  # (batch, width, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, width)
        
        # Projection to features
        out = self.projection(x)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

