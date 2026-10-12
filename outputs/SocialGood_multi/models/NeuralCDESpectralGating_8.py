import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralCDESpectralGating(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'cde_width': [64, 256, 1024],
            'n_spectral_modes': [8, 32, 128],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.3],
            'bottleneck_ratio': [2, 4, 8],
            'spectral_gating_dim': [16, 64, 256],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cde_width = self.config.get('cde_width', 256)
        n_spectral_modes = self.config.get('n_spectral_modes', 32)
        n_layers = self.config.get('n_layers', 2)
        dropout = self.config.get('dropout', 0.1)
        bottleneck_ratio = self.config.get('bottleneck_ratio', 4)
        spectral_gating_dim = self.config.get('spectral_gating_dim', 64)
        
        self.cde_width = cde_width
        self.n_spectral_modes = min(n_spectral_modes, input_len // 2)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to CDE space
        self.input_proj = nn.Linear(self.n_features, cde_width)  # (batch, input_len, cde_width)
        
        # CDE dynamics function: lightweight MLP
        cde_layers = []
        for i in range(n_layers):
            in_dim = cde_width if i == 0 else cde_width
            out_dim = cde_width
            cde_layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                cde_layers.append(nn.GELU())
                cde_layers.append(nn.Dropout(dropout))
        self.cde_func = nn.Sequential(*cde_layers)
        
        # Initial hidden state
        self.h0 = nn.Parameter(torch.randn(1, cde_width) * 0.01)
        
        # Multi-resolution spectral gating
        # Coarse (low-freq) and fine (high-freq) spectral weights
        self.coarse_real = nn.Parameter(torch.randn(cde_width, cde_width, self.n_spectral_modes) * 0.02)
        self.coarse_imag = nn.Parameter(torch.randn(cde_width, cde_width, self.n_spectral_modes) * 0.02)
        self.fine_real = nn.Parameter(torch.randn(cde_width, cde_width, self.n_spectral_modes) * 0.02)
        self.fine_imag = nn.Parameter(torch.randn(cde_width, cde_width, self.n_spectral_modes) * 0.02)
        
        # Channel-adaptive gating network
        self.gate_net = nn.Sequential(
            nn.Linear(cde_width, spectral_gating_dim),
            nn.GELU(),
            nn.Linear(spectral_gating_dim, cde_width),
            nn.Sigmoid()
        )  # (batch, seq_len, cde_width)
        
        # Bottleneck for attention
        bottleneck_dim = max(cde_width // bottleneck_ratio, 16)
        self.bottleneck_down = nn.Linear(cde_width, bottleneck_dim)  # (batch, seq_len, bottleneck_dim)
        self.bottleneck_up = nn.Linear(bottleneck_dim, cde_width)    # (batch, seq_len, cde_width)
        
        # Cross-attention from query positions to CDE trajectory
        self.attention = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=max(1, bottleneck_dim // 16),
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(cde_width, self.n_features)  # (batch, output_len, n_features)
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def _spectral_conv(self, x, real_w, imag_w):
        # x: (batch, seq_len, cde_width)
        # Apply spectral convolution in Fourier space
        x_ft = torch.fft.rfft(x, dim=1)  # (batch, seq_len//2+1, cde_width)
        
        # Truncate to n_spectral_modes
        x_ft_trunc = x_ft[:, :self.n_spectral_modes, :]  # (batch, n_spectral_modes, cde_width)
        
        # Complex multiplication: (batch, modes, cde_width) @ (cde_width, cde_width, modes)
        # We need to compute for each mode: x_ft[:, m, :] @ (real_w[:, :, m] + i*imag_w[:, :, m])
        out_real = torch.zeros_like(x_ft_trunc.real)
        out_imag = torch.zeros_like(x_ft_trunc.imag)
        
        for m in range(self.n_spectral_modes):
            real_wm = real_w[:, :, m]  # (cde_width, cde_width)
            imag_wm = imag_w[:, :, m]  # (cde_width, cde_width)
            
            x_real = x_ft_trunc[:, m, :].real  # (batch, cde_width)
            x_imag = x_ft_trunc[:, m, :].imag  # (batch, cde_width)
            
            # (a+ib)(c+id) = (ac-bd) + i(ad+bc)
            out_real[:, m, :] = x_real @ real_wm - x_imag @ imag_wm  # (batch, cde_width)
            out_imag[:, m, :] = x_real @ imag_wm + x_imag @ real_wm  # (batch, cde_width)
        
        # Reconstruct full FFT
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :self.n_spectral_modes, :] = torch.complex(out_real, out_imag)
        
        # Inverse FFT
        out = torch.fft.irfft(out_ft, n=x.size(1), dim=1)  # (batch, seq_len, cde_width)
        return out
    
    def _solve_cde(self, x, dt=1.0):
        # Simple Euler integration for CDE: dh/dt = f(h) * dx/dt
        # x: (batch, input_len, cde_width)
        batch_size = x.size(0)
        h = self.h0.expand(batch_size, -1)  # (batch, cde_width)
        
        trajectory = [h]
        for t in range(self.input_len - 1):
            dx = x[:, t+1, :] - x[:, t, :]  # (batch, cde_width)
            dh = self.cde_func(h) * dx  # (batch, cde_width)
            h = h + dh  # (batch, cde_width)
            trajectory.append(h)
        
        trajectory = torch.stack(trajectory, dim=1)  # (batch, input_len, cde_width)
        return trajectory
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.size(0)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to CDE space
        x_cde = self.input_proj(x)  # (batch, input_len, cde_width)
        x_cde = F.gelu(x_cde)  # (batch, input_len, cde_width)
        
        # Solve CDE to get continuous trajectory
        h_trajectory = self._solve_cde(x_cde)  # (batch, input_len, cde_width)
        
        # Multi-resolution spectral gating
        coarse_out = self._spectral_conv(h_trajectory, self.coarse_real, self.coarse_imag)  # (batch, input_len, cde_width)
        fine_out = self._spectral_conv(h_trajectory, self.fine_real, self.fine_imag)  # (batch, input_len, cde_width)
        
        # Channel-adaptive gating
        gate = self.gate_net(h_trajectory)  # (batch, input_len, cde_width)
        h_gated = gate * coarse_out + (1 - gate) * fine_out  # (batch, input_len, cde_width)
        h_gated = self.dropout_layer(h_gated)  # (batch, input_len, cde_width)
        
        # Residual connection
        h_enhanced = h_trajectory + h_gated  # (batch, input_len, cde_width)
        
        # Bottleneck for attention
        h_bottleneck = self.bottleneck_down(h_enhanced)  # (batch, input_len, bottleneck_dim)
        h_bottleneck = F.gelu(h_bottleneck)  # (batch, input_len, bottleneck_dim)
        
        # Create query positions for forecasting (learned temporal embeddings)
        query_positions = torch.linspace(0, 1, self.output_len, device=x.device).unsqueeze(0).unsqueeze(-1)  # (1, output_len, 1)
        query_positions = query_positions.expand(batch_size, -1, -1)  # (batch, output_len, 1)
        query_embed = torch.sin(query_positions * 3.14159 * torch.arange(1, h_bottleneck.size(-1)//2 + 1, device=x.device).view(1, 1, -1))  # (batch, output_len, bottleneck_dim//2)
        query_embed = torch.cat([query_embed, torch.cos(query_positions * 3.14159 * torch.arange(1, h_bottleneck.size(-1)//2 + 1, device=x.device).view(1, 1, -1))], dim=-1)  # (batch, output_len, bottleneck_dim)
        if query_embed.size(-1) < h_bottleneck.size(-1):
            pad = torch.zeros(batch_size, self.output_len, h_bottleneck.size(-1) - query_embed.size(-1), device=x.device)
            query_embed = torch.cat([query_embed, pad], dim=-1)  # (batch, output_len, bottleneck_dim)
        
        # Cross-attention: queries attend to CDE trajectory
        attn_out, _ = self.attention(query_embed, h_bottleneck, h_bottleneck)  # (batch, output_len, bottleneck_dim)
        attn_out = self.dropout_layer(attn_out)  # (batch, output_len, bottleneck_dim)
        
        # Upsample from bottleneck
        h_out = self.bottleneck_up(attn_out)  # (batch, output_len, cde_width)
        h_out = F.gelu(h_out)  # (batch, output_len, cde_width)
        
        # Project to output features
        out = self.output_proj(h_out)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

