import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveSpectralFNO_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'n_modes_coarse': [4, 16, 64],
            'n_modes_fine': [16, 64, 256],
            'bottleneck_ratio': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.3)
        self.n_modes_coarse = self.config.get('n_modes_coarse', 16)
        self.n_modes_fine = self.config.get('n_modes_fine', 64)
        self.bottleneck_ratio = self.config.get('bottleneck_ratio', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        # (batch, input_len, hidden_dim)
        
        # Positional encoding for temporal information
        self.pos_enc = nn.Parameter(torch.randn(1, self.input_len, self.hidden_dim) * 0.02)
        # (1, input_len, hidden_dim)
        
        # Adaptive spectral layers
        self.spectral_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.gate_nets = nn.ModuleList()
        
        for _ in range(self.n_layers):
            # Spectral convolution with dual-scale modes
            self.spectral_layers.append(
                AdaptiveSpectralConv(
                    self.hidden_dim,
                    self.n_modes_coarse,
                    self.n_modes_fine,
                    self.bottleneck_ratio
                )
            )
            self.norm_layers.append(nn.LayerNorm(self.hidden_dim))
            # Gating network: decides coarse vs fine weighting per sample
            self.gate_nets.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(self.hidden_dim, 2),
                    nn.Softmax(dim=-1)
                )
            )
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        # (batch, output_len, n_features)
        
        # Temporal interpolation for variable length
        self.temporal_mixer = nn.Linear(self.input_len, self.output_len)
        # (batch, hidden_dim, output_len)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')
        # (batch, input_len, n_features)
        
        # Project to hidden dimension
        x = self.input_proj(x)
        # (batch, input_len, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_enc
        # (batch, input_len, hidden_dim)
        
        # Apply spectral layers with adaptive gating
        for i, (spec_layer, norm, gate_net) in enumerate(zip(
            self.spectral_layers, self.norm_layers, self.gate_nets
        )):
            residual = x
            # (batch, input_len, hidden_dim)
            
            # Compute adaptive gate weights from global feature representation
            x_perm = x.permute(0, 2, 1)
            # (batch, hidden_dim, input_len)
            gate_weights = gate_net(x_perm)
            # (batch, 2) - [coarse_weight, fine_weight]
            
            # Apply spectral convolution with gating
            x = spec_layer(x, gate_weights)
            # (batch, input_len, hidden_dim)
            
            x = self.dropout_layer(x)
            # (batch, input_len, hidden_dim)
            
            x = norm(x + residual)
            # (batch, input_len, hidden_dim)
        
        # Temporal dimension adjustment
        x = x.permute(0, 2, 1)
        # (batch, hidden_dim, input_len)
        x = self.temporal_mixer(x)
        # (batch, hidden_dim, output_len)
        x = x.permute(0, 2, 1)
        # (batch, output_len, hidden_dim)
        
        # Project back to feature dimension
        x = self.output_proj(x)
        # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')
        # (batch, output_len, n_features)
        
        return x


class AdaptiveSpectralConv(nn.Module):
    def __init__(self, hidden_dim, n_modes_coarse, n_modes_fine, bottleneck_ratio):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modes_coarse = n_modes_coarse
        self.n_modes_fine = n_modes_fine
        self.bottleneck_dim = hidden_dim // bottleneck_ratio
        
        # Bottleneck projection
        self.bottleneck_down = nn.Linear(hidden_dim, self.bottleneck_dim)
        # (batch, seq_len, bottleneck_dim)
        
        # Coarse frequency weights (low frequencies)
        # Shape: (bottleneck_dim, n_modes_coarse, bottleneck_dim) for einsum 'bdi,dio->bdo'
        self.weight_coarse_real = nn.Parameter(
            torch.randn(self.bottleneck_dim, n_modes_coarse, self.bottleneck_dim) * 0.02
        )
        self.weight_coarse_imag = nn.Parameter(
            torch.randn(self.bottleneck_dim, n_modes_coarse, self.bottleneck_dim) * 0.02
        )
        
        # Fine frequency weights (high frequencies)
        # Shape: (bottleneck_dim, n_modes_fine, bottleneck_dim) for einsum 'bdi,dio->bdo'
        self.weight_fine_real = nn.Parameter(
            torch.randn(self.bottleneck_dim, n_modes_fine, self.bottleneck_dim) * 0.02
        )
        self.weight_fine_imag = nn.Parameter(
            torch.randn(self.bottleneck_dim, n_modes_fine, self.bottleneck_dim) * 0.02
        )
        
        # Bottleneck expansion
        self.bottleneck_up = nn.Linear(self.bottleneck_dim, hidden_dim)
        # (batch, seq_len, hidden_dim)
    
    def forward(self, x, gate_weights):
        # x: (batch, seq_len, hidden_dim)
        # gate_weights: (batch, 2)
        
        batch_size, seq_len, _ = x.shape
        
        # Bottleneck
        x = self.bottleneck_down(x)
        # (batch, seq_len, bottleneck_dim)
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=1)
        # (batch, seq_len//2+1, bottleneck_dim)
        
        out_ft = torch.zeros_like(x_ft)
        # (batch, seq_len//2+1, bottleneck_dim)
        
        # Coarse frequencies (low modes)
        n_c = min(self.n_modes_coarse, x_ft.shape[1])
        coarse_ft = x_ft[:, :n_c, :]
        # (batch, n_c, bottleneck_dim)
        coarse_ft = coarse_ft.permute(0, 2, 1)
        # (batch, bottleneck_dim, n_c)
        
        coarse_out_real = torch.einsum('bdi,dio->bdo', coarse_ft.real, self.weight_coarse_real[:, :n_c, :])
        # (batch, bottleneck_dim, bottleneck_dim)
        coarse_out_imag = torch.einsum('bdi,dio->bdo', coarse_ft.imag, self.weight_coarse_imag[:, :n_c, :])
        # (batch, bottleneck_dim, bottleneck_dim)
        coarse_out = torch.complex(coarse_out_real, coarse_out_imag)
        # (batch, bottleneck_dim, bottleneck_dim)
        coarse_out = coarse_out.permute(0, 2, 1)
        # (batch, bottleneck_dim, bottleneck_dim)
        
        # Note: after einsum 'bdi,dio->bdo', output is (batch, bottleneck_dim, bottleneck_dim)
        # We need to truncate to match n_c modes in the output
        # Actually, let's re-examine: we want output to be (batch, n_c, bottleneck_dim)
        # The einsum 'bdi,dio->bdo' gives (batch, bottleneck_dim, bottleneck_dim)
        # We need to use a different approach: project each mode independently
        
        # Re-approach: use 'bdi,dio->bdo' where i=n_c, o=bottleneck_dim
        # Input: (batch, bottleneck_dim, n_c), Weight: (bottleneck_dim, n_c, bottleneck_dim)
        # Output: (batch, bottleneck_dim, bottleneck_dim) - this is wrong for our needs
        
        # Correct approach: we want to transform each frequency mode
        # Input: (batch, n_c, bottleneck_dim) -> permute to (batch, bottleneck_dim, n_c)
        # Weight should be (bottleneck_dim, bottleneck_dim) per mode, or we use a different einsum
        
        # Let's use: 'bim,imo->bom' where b=batch, i=bottleneck_dim, m=n_c, o=bottleneck_dim
        # Actually simplest: 'bmi,mio->bmo' where m=n_c modes
        
        # Re-do with correct einsum
        coarse_ft = x_ft[:, :n_c, :]
        # (batch, n_c, bottleneck_dim)
        
        coarse_out_real = torch.einsum('bmi,mio->bmo', coarse_ft.real, self.weight_coarse_real[:, :n_c, :])
        # Wait, weight is (bottleneck_dim, n_c, bottleneck_dim), we need (bottleneck_dim, bottleneck_dim, n_c) for 'bmi,ijo->bmj'
        
        # Let me be very careful. We have coarse_ft: (batch, n_c, bottleneck_dim)
        # We want output: (batch, n_c, bottleneck_dim)
        # This is a linear transform on the last dimension, per-mode
        # Use: 'bmi,mij->bmj' where weight is (n_c, bottleneck_dim, bottleneck_dim) or similar
        
        # Actually the cleanest is: for each mode, do matrix multiply
        # coarse_ft: (batch, n_c, bottleneck_dim), weight: (n_c, bottleneck_dim, bottleneck_dim)
        # output: (batch, n_c, bottleneck_dim)
        
        # So weight should be (n_c, bottleneck_dim, bottleneck_dim)
        # But we defined it as (bottleneck_dim, n_modes_coarse, bottleneck_dim)
        # Let's permute: (n_modes_coarse, bottleneck_dim, bottleneck_dim)
        
        # Let me restart with the correct weight shapes
        pass  # will be replaced below
        
        # Actually, let me just fix the einsum properly here:
        # We want: input (batch, n_c, bottleneck_dim), apply linear transform per mode
        # Weight per mode: (bottleneck_dim, bottleneck_dim)
        # Output: (batch, n_c, bottleneck_dim)
        
        # Using einsum 'bmi,ioj->bmj' doesn't work directly
        # Best: 'bmi,mij->bmj' with weight (n_c, bottleneck_dim, bottleneck_dim)
        
        # Let me re-initialize weights correctly and redo
        # I'll change weight shape interpretation in the init and use here
        
        # For now, use a simple loop or reshape approach
        # Actually, let's use bmm after proper reshaping
        
        # coarse_ft: (batch, n_c, bottleneck_dim)
        # weight: (n_c, bottleneck_dim, bottleneck_dim) - need to permute from stored shape
        
        # Stored as (bottleneck_dim, n_modes_coarse, bottleneck_dim)
        # Permute to (n_modes_coarse, bottleneck_dim, bottleneck_dim)
        w_coarse_real = self.weight_coarse_real.permute(1, 0, 2)[:n_c, :, :]
        # (n_c, bottleneck_dim, bottleneck_dim)
        w_coarse_imag = self.weight_coarse_imag.permute(1, 0, 2)[:n_c, :, :]
        # (n_c, bottleneck_dim, bottleneck_dim)
        
        # batch matrix multiply: (batch, n_c, bottleneck_dim) @ (n_c, bottleneck_dim, bottleneck_dim)
        # Need to use einsum or bmm with expansion
        # coarse_ft: (batch, n_c, bottleneck_dim)
        # weight: (n_c, bottleneck_dim, bottleneck_dim)
        
        coarse_out_real = torch.einsum('bmi,mij->bmj', coarse_ft.real, w_coarse_real)
        # (batch, n_c, bottleneck_dim)
        coarse_out_imag = torch.einsum('bmi,mij->bmj', coarse_ft.imag, w_coarse_imag)
        # (batch, n_c, bottleneck_dim)
        coarse_out = torch.complex(coarse_out_real, coarse_out_imag)
        # (batch, n_c, bottleneck_dim)
        
        # Fine frequencies (higher modes)
        n_f = min(self.n_modes_fine, x_ft.shape[1] - n_c)
        if n_f > 0 and n_c < x_ft.shape[1]:
            fine_ft = x_ft[:, n_c:n_c+n_f, :]
            # (batch, n_f, bottleneck_dim)
            
            w_fine_real = self.weight_fine_real.permute(1, 0, 2)[:n_f, :, :]
            # (n_f, bottleneck_dim, bottleneck_dim)
            w_fine_imag = self.weight_fine_imag.permute(1, 0, 2)[:n_f, :, :]
            # (n_f, bottleneck_dim, bottleneck_dim)
            
            fine_out_real = torch.einsum('bmi,mij->bmj', fine_ft.real, w_fine_real)
            # (batch, n_f, bottleneck_dim)
            fine_out_imag = torch.einsum('bmi,mij->bmj', fine_ft.imag, w_fine_imag)
            # (batch, n_f, bottleneck_dim)
            fine_out = torch.complex(fine_out_real, fine_out_imag)
            # (batch, n_f, bottleneck_dim)
        else:
            fine_out = torch.zeros(batch_size, 0, self.bottleneck_dim, device=x.device, dtype=torch.complex64)
            # (batch, 0, bottleneck_dim)
        
        # Apply adaptive gating
        gate_coarse = gate_weights[:, 0:1].unsqueeze(-1)
        # (batch, 1, 1)
        gate_fine = gate_weights[:, 1:2].unsqueeze(-1)
        # (batch, 1, 1)
        
        # Combine with gating
        if n_c > 0:
            out_ft[:, :n_c, :] = gate_coarse * coarse_out
            # (batch, n_c, bottleneck_dim)
        if n_f > 0:
            out_ft[:, n_c:n_c+n_f, :] = gate_fine * fine_out
            # (batch, n_f, bottleneck_dim)
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=seq_len, dim=1)
        # (batch, seq_len, bottleneck_dim)
        
        # Expand bottleneck
        x = self.bottleneck_up(x)
        # (batch, seq_len, hidden_dim)
        
        return x


# Let me rewrite the entire class cleanly with correct weight shapes from the start
class AdaptiveSpectralConv(nn.Module):
    def __init__(self, hidden_dim, n_modes_coarse, n_modes_fine, bottleneck_ratio):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modes_coarse = n_modes_coarse
        self.n_modes_fine = n_modes_fine
        self.bottleneck_dim = hidden_dim // bottleneck_ratio
        
        # Bottleneck projection
        self.bottleneck_down = nn.Linear(hidden_dim, self.bottleneck_dim)
        # (batch, seq_len, bottleneck_dim)
        
        # Coarse frequency weights (low frequencies)
        # Shape: (n_modes_coarse, bottleneck_dim, bottleneck_dim) for per-mode linear transform
        self.weight_coarse_real = nn.Parameter(
            torch.randn(n_modes_coarse, self.bottleneck_dim, self.bottleneck_dim) * 0.02
        )
        self.weight_coarse_imag = nn.Parameter(
            torch.randn(n_modes_coarse, self.bottleneck_dim, self.bottleneck_dim) * 0.02
        )
        
        # Fine frequency weights (high frequencies)
        # Shape: (n_modes_fine, bottleneck_dim, bottleneck_dim) for per-mode linear transform
        self.weight_fine_real = nn.Parameter(
            torch.randn(n_modes_fine, self.bottleneck_dim, self.bottleneck_dim) * 0.02
        )
        self.weight_fine_imag = nn.Parameter(
            torch.randn(n_modes_fine, self.bottleneck_dim, self.bottleneck_dim) * 0.02
        )
        
        # Bottleneck expansion
        self.bottleneck_up = nn.Linear(self.bottleneck_dim, hidden_dim)
        # (batch, seq_len, hidden_dim)
    
    def forward(self, x, gate_weights):
        # x: (batch, seq_len, hidden_dim)
        # gate_weights: (batch, 2)
        
        batch_size, seq_len, _ = x.shape
        
        # Bottleneck
        x = self.bottleneck_down(x)
        # (batch, seq_len, bottleneck_dim)
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=1)
        # (batch, seq_len//2+1, bottleneck_dim)
        
        out_ft = torch.zeros_like(x_ft)
        # (batch, seq_len//2+1, bottleneck_dim)
        
        # Coarse frequencies (low modes)
        n_c = min(self.n_modes_coarse, x_ft.shape[1])
        coarse_ft = x_ft[:, :n_c, :]
        # (batch, n_c, bottleneck_dim)
        
        # Per-mode linear transform using einsum
        # 'bmi,mij->bmj': batch, mode, input -> batch, mode, output
        coarse_out_real = torch.einsum('bmi,mij->bmj', coarse_ft.real, self.weight_coarse_real[:n_c, :, :])
        # (batch, n_c, bottleneck_dim)
        coarse_out_imag = torch.einsum('bmi,mij->bmj', coarse_ft.imag, self.weight_coarse_imag[:n_c, :, :])
        # (batch, n_c, bottleneck_dim)
        coarse_out = torch.complex(coarse_out_real, coarse_out_imag)
        # (batch, n_c, bottleneck_dim)
        
        # Fine frequencies (higher modes)
        n_f = min(self.n_modes_fine, x_ft.shape[1] - n_c)
        if n_f > 0 and n_c < x_ft.shape[1]:
            fine_ft = x_ft[:, n_c:n_c+n_f, :]
            # (batch, n_f, bottleneck_dim)
            
            fine_out_real = torch.einsum('bmi,mij->bmj', fine_ft.real, self.weight_fine_real[:n_f, :, :])
            # (batch, n_f, bottleneck_dim)
            fine_out_imag = torch.einsum('bmi,mij->bmj', fine_ft.imag, self.weight_fine_imag[:n_f, :, :])
            # (batch, n_f, bottleneck_dim)
            fine_out = torch.complex(fine_out_real, fine_out_imag)
            # (batch, n_f, bottleneck_dim)
        else:
            fine_out = torch.zeros(batch_size, 0, self.bottleneck_dim, device=x.device, dtype=torch.complex64)
            # (batch, 0, bottleneck_dim)
        
        # Apply adaptive gating
        gate_coarse = gate_weights[:, 0:1].unsqueeze(-1)
        # (batch, 1, 1)
        gate_fine = gate_weights[:, 1:2].unsqueeze(-1)
        # (batch, 1, 1)
        
        # Combine with gating
        if n_c > 0:
            out_ft[:, :n_c, :] = gate_coarse * coarse_out
            # (batch, n_c, bottleneck_dim)
        if n_f > 0:
            out_ft[:, n_c:n_c+n_f, :] = gate_fine * fine_out
            # (batch, n_f, bottleneck_dim)
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=seq_len, dim=1)
        # (batch, seq_len, bottleneck_dim)
        
        # Expand bottleneck
        x = self.bottleneck_up(x)
        # (batch, seq_len, hidden_dim)
        
        return x


class AdaptiveSpectralFNO(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'n_modes_coarse': [4, 16, 64],
            'n_modes_fine': [16, 64, 256],
            'bottleneck_ratio': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.3)
        self.n_modes_coarse = self.config.get('n_modes_coarse', 16)
        self.n_modes_fine = self.config.get('n_modes_fine', 64)
        self.bottleneck_ratio = self.config.get('bottleneck_ratio', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        # (batch, input_len, hidden_dim)
        
        # Positional encoding for temporal information
        self.pos_enc = nn.Parameter(torch.randn(1, self.input_len, self.hidden_dim) * 0.02)
        # (1, input_len, hidden_dim)
        
        # Adaptive spectral layers
        self.spectral_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.gate_nets = nn.ModuleList()
        
        for _ in range(self.n_layers):
            # Spectral convolution with dual-scale modes
            self.spectral_layers.append(
                AdaptiveSpectralConv(
                    self.hidden_dim,
                    self.n_modes_coarse,
                    self.n_modes_fine,
                    self.bottleneck_ratio
                )
            )
            self.norm_layers.append(nn.LayerNorm(self.hidden_dim))
            # Gating network: decides coarse vs fine weighting per sample
            self.gate_nets.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(self.hidden_dim, 2),
                    nn.Softmax(dim=-1)
                )
            )
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        # (batch, output_len, n_features)
        
        # Temporal interpolation for variable length
        self.temporal_mixer = nn.Linear(self.input_len, self.output_len)
        # (batch, hidden_dim, output_len)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')
        # (batch, input_len, n_features)
        
        # Project to hidden dimension
        x = self.input_proj(x)
        # (batch, input_len, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_enc
        # (batch, input_len, hidden_dim)
        
        # Apply spectral layers with adaptive gating
        for i, (spec_layer, norm, gate_net) in enumerate(zip(
            self.spectral_layers, self.norm_layers, self.gate_nets
        )):
            residual = x
            # (batch, input_len, hidden_dim)
            
            # Compute adaptive gate weights from global feature representation
            x_perm = x.permute(0, 2, 1)
            # (batch, hidden_dim, input_len)
            gate_weights = gate_net(x_perm)
            # (batch, 2) - [coarse_weight, fine_weight]
            
            # Apply spectral convolution with gating
            x = spec_layer(x, gate_weights)
            # (batch, input_len, hidden_dim)
            
            x = self.dropout_layer(x)
            # (batch, input_len, hidden_dim)
            
            x = norm(x + residual)
            # (batch, input_len, hidden_dim)
        
        # Temporal dimension adjustment
        x = x.permute(0, 2, 1)
        # (batch, hidden_dim, input_len)
        x = self.temporal_mixer(x)
        # (batch, hidden_dim, output_len)
        x = x.permute(0, 2, 1)
        # (batch, output_len, hidden_dim)
        
        # Project back to feature dimension
        x = self.output_proj(x)
        # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')
        # (batch, output_len, n_features)
        
        return x


class AdaptiveSpectralConv(nn.Module):
    def __init__(self, hidden_dim, n_modes_coarse, n_modes_fine, bottleneck_ratio):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_modes_coarse = n_modes_coarse
        self.n_modes_fine = n_modes_fine
        self.bottleneck_dim = hidden_dim // bottleneck_ratio
        
        # Bottleneck projection
        self.bottleneck_down = nn.Linear(hidden_dim, self.bottleneck_dim)
        # (batch, seq_len, bottleneck_dim)
        
        # Coarse frequency weights (low frequencies)
        # Shape: (n_modes_coarse, bottleneck_dim, bottleneck_dim) for per-mode linear transform
        self.weight_coarse_real = nn.Parameter(
            torch.randn(n_modes_coarse, self.bottleneck_dim, self.bottleneck_dim) * 0.02
        )
        self.weight_coarse_imag = nn.Parameter(
            torch.randn(n_modes_coarse, self.bottleneck_dim, self.bottleneck_dim) * 0.02
        )
        
        # Fine frequency weights (high frequencies)
        # Shape: (n_modes_fine, bottleneck_dim, bottleneck_dim) for per-mode linear transform
        self.weight_fine_real = nn.Parameter(
            torch.randn(n_modes_fine, self.bottleneck_dim, self.bottleneck_dim) * 0.02
        )
        self.weight_fine_imag = nn.Parameter(
            torch.randn(n_modes_fine, self.bottleneck_dim, self.bottleneck_dim) * 0.02
        )
        
        # Bottleneck expansion
        self.bottleneck_up = nn.Linear(self.bottleneck_dim, hidden_dim)
        # (batch, seq_len, hidden_dim)
    
    def forward(self, x, gate_weights):
        # x: (batch, seq_len, hidden_dim)
        # gate_weights: (batch, 2)
        
        batch_size, seq_len, _ = x.shape
        
        # Bottleneck
        x = self.bottleneck_down(x)
        # (batch, seq_len, bottleneck_dim)
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=1)
        # (batch, seq_len//2+1, bottleneck_dim)
        
        out_ft = torch.zeros_like(x_ft)
        # (batch, seq_len//2+1, bottleneck_dim)
        
        # Coarse frequencies (low modes)
        n_c = min(self.n_modes_coarse, x_ft.shape[1])
        coarse_ft = x_ft[:, :n_c, :]
        # (batch, n_c, bottleneck_dim)
        
        # Per-mode linear transform using einsum
        # 'bmi,mij->bmj': batch, mode, input -> batch, mode, output
        coarse_out_real = torch.einsum('bmi,mij->bmj', coarse_ft.real, self.weight_coarse_real[:n_c, :, :])
        # (batch, n_c, bottleneck_dim)
        coarse_out_imag = torch.einsum('bmi,mij->bmj', coarse_ft.imag, self.weight_coarse_imag[:n_c, :, :])
        # (batch, n_c, bottleneck_dim)
        coarse_out = torch.complex(coarse_out_real, coarse_out_imag)
        # (batch, n_c, bottleneck_dim)
        
        # Fine frequencies (higher modes)
        n_f = min(self.n_modes_fine, x_ft.shape[1] - n_c)
        if n_f > 0 and n_c < x_ft.shape[1]:
            fine_ft = x_ft[:, n_c:n_c+n_f, :]
            # (batch, n_f, bottleneck_dim)
            
            fine_out_real = torch.einsum('bmi,mij->bmj', fine_ft.real, self.weight_fine_real[:n_f, :, :])
            # (batch, n_f, bottleneck_dim)
            fine_out_imag = torch.einsum('bmi,mij->bmj', fine_ft.imag, self.weight_fine_imag[:n_f, :, :])
            # (batch, n_f, bottleneck_dim)
            fine_out = torch.complex(fine_out_real, fine_out_imag)
            # (batch, n_f, bottleneck_dim)
        else:
            fine_out = torch.zeros(batch_size, 0, self.bottleneck_dim, device=x.device, dtype=torch.complex64)
            # (batch, 0, bottleneck_dim)
        
        # Apply adaptive gating
        gate_coarse = gate_weights[:, 0:1].unsqueeze(-1)
        # (batch, 1, 1)
        gate_fine = gate_weights[:, 1:2].unsqueeze(-1)
        # (batch, 1, 1)
        
        # Combine with gating
        if n_c > 0:
            out_ft[:, :n_c, :] = gate_coarse * coarse_out
            # (batch, n_c, bottleneck_dim)
        if n_f > 0:
            out_ft[:, n_c:n_c+n_f, :] = gate_fine * fine_out
            # (batch, n_f, bottleneck_dim)
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=seq_len, dim=1)
        # (batch, seq_len, bottleneck_dim)
        
        # Expand bottleneck
        x = self.bottleneck_up(x)
        # (batch, seq_len, hidden_dim)
        
        return x
