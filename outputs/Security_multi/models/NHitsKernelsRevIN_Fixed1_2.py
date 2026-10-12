import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NHitsKernelsRevIN_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_pool': [2, 4, 8],
            'n_kernels': [4, 16, 64],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_pool = self.config.get('n_pool', 4)
        self.n_kernels = self.config.get('n_kernels', 16)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale pooling: extract features at different resolutions
        self.pool_scales = [max(1, input_len // (2 ** i)) for i in range(self.n_pool)]
        self.pool_scales = [s for s in self.pool_scales if s >= 2]  # keep only meaningful scales
        if len(self.pool_scales) == 0:
            self.pool_scales = [input_len]
        
        n_scales = len(self.pool_scales)
        # Ensure projections sum exactly to hidden_dim
        base_dim = self.hidden_dim // n_scales
        remainder = self.hidden_dim - base_dim * (n_scales - 1)
        self.proj_dims = [base_dim] * (n_scales - 1) + [remainder] if n_scales > 1 else [self.hidden_dim]
        
        self.pool_projs = nn.ModuleList([
            nn.Linear(n_features, dim)
            for dim in self.proj_dims
        ])
        
        # Learnable kernel families - store as direct nn.Parameter attributes
        self.kernel_types = ['fourier', 'wavelet', 'polynomial', 'exponential']
        self.n_kernel_types = len(self.kernel_types)
        
        # Fourier: frequencies for periodic components
        self.kernel_fourier = nn.Parameter(torch.randn(self.n_kernels) * 0.1)
        # Wavelet: scales and translations
        self.kernel_wavelet_scale = nn.Parameter(torch.randn(self.n_kernels).abs() + 0.5)
        self.kernel_wavelet_trans = nn.Parameter(torch.randn(self.n_kernels) * 0.1)
        # Polynomial: degrees
        self.kernel_polynomial = nn.Parameter(torch.arange(self.n_kernels).float() * 0.5)
        # Exponential: decay rates
        self.kernel_exponential = nn.Parameter(torch.randn(self.n_kernels).abs() + 0.1)
        
        # Kernel selection network
        self.kernel_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.n_kernel_types)
        )
        
        # Basis expansion for each output time step
        total_kernels = self.n_kernel_types * self.n_kernels
        self.basis_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, total_kernels)
            )
            for _ in range(self.n_layers)
        ])
        
        # Output projection: from kernel basis to features
        self.out_proj = nn.Linear(total_kernels, n_features)
        
        # Temporal embedding for output positions
        self.time_embed = nn.Embedding(output_len, self.hidden_dim // 2)
        
    def evaluate_kernels(self, t, output_len):
        # t: (output_len,) positions in [0, 1]
        t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, output_len)
        
        kernel_values = []
        
        # Fourier: sin(2*pi*freq*t)
        freqs = self.kernel_fourier.unsqueeze(0).unsqueeze(-1)  # (1, n_kernels, 1)
        fourier_vals = torch.sin(2 * 3.14159 * freqs * t)  # (1, n_kernels, output_len)
        kernel_values.append(fourier_vals)
        
        # Wavelet: Gaussian modulated cosine
        scale = self.kernel_wavelet_scale.unsqueeze(0).unsqueeze(-1)  # (1, n_kernels, 1)
        trans = self.kernel_wavelet_trans.unsqueeze(0).unsqueeze(-1)  # (1, n_kernels, 1)
        wavelet_vals = torch.exp(-((t - trans) / scale) ** 2) * torch.cos(5 * (t - trans) / scale)  # (1, n_kernels, output_len)
        kernel_values.append(wavelet_vals)
        
        # Polynomial: t^deg
        degs = self.kernel_polynomial.unsqueeze(0).unsqueeze(-1)  # (1, n_kernels, 1)
        poly_vals = t ** degs  # (1, n_kernels, output_len)
        kernel_values.append(poly_vals)
        
        # Exponential: exp(-rate*t)
        rates = self.kernel_exponential.unsqueeze(0).unsqueeze(-1)  # (1, n_kernels, 1)
        exp_vals = torch.exp(-rates * t)  # (1, n_kernels, output_len)
        kernel_values.append(exp_vals)
        
        # Stack all kernels: (1, n_kernel_types*n_kernels, output_len)
        all_kernels = torch.cat(kernel_values, dim=1)  # (1, total_kernels, output_len)
        return all_kernels
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Multi-scale pooling
        pooled_features = []
        for scale, proj in zip(self.pool_scales, self.pool_projs):
            # Average pool to scale
            if scale < x.shape[1]:
                x_pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), scale).transpose(1, 2)  # (batch, scale, n_features)
            else:
                x_pooled = x  # (batch, input_len, n_features)
            
            # Project
            x_proj = proj(x_pooled)  # (batch, scale, proj_dim)
            # Global average pool
            x_proj = x_proj.mean(dim=1)  # (batch, proj_dim)
            pooled_features.append(x_proj)
        
        # Combine multi-scale features
        h = torch.cat(pooled_features, dim=-1)  # (batch, hidden_dim)
        
        # Kernel selection gate
        gate = F.softmax(self.kernel_gate(h), dim=-1)  # (batch, n_kernel_types)
        
        # Generate output for each time step
        t_vals = torch.linspace(0, 1, self.output_len, device=x.device)  # (output_len,)
        kernel_basis = self.evaluate_kernels(t_vals, self.output_len)  # (1, total_kernels, output_len)
        
        # Apply layers with basis expansion
        all_weights = []
        for layer in self.basis_weights:
            w = layer(h)  # (batch, total_kernels)
            all_weights.append(w)
        
        # Average layer weights
        weights = torch.stack(all_weights, dim=0).mean(dim=0)  # (batch, total_kernels)
        
        # Apply gate: expand gate to per-kernel level
        gate_expanded = gate.repeat_interleave(self.n_kernels, dim=-1)  # (batch, total_kernels)
        weights = weights * gate_expanded  # (batch, total_kernels)
        
        # Compute modulated kernel basis for each time step
        # kernel_basis: (1, total_kernels, output_len)
        # weights: (batch, total_kernels) -> (batch, total_kernels, 1)
        weights_unsqueezed = weights.unsqueeze(-1)  # (batch, total_kernels, 1)
        
        # Expand kernel_basis to batch size
        kb_expanded = kernel_basis.expand(x.shape[0], -1, -1)  # (batch, total_kernels, output_len)
        
        # Modulate: (batch, total_kernels, output_len)
        modulated_basis = kb_expanded * weights_unsqueezed  # (batch, total_kernels, output_len)
        
        # Transpose for projection: (batch, output_len, total_kernels)
        modulated_basis = modulated_basis.transpose(1, 2)  # (batch, output_len, total_kernels)
        
        # Project to features
        out = self.out_proj(modulated_basis)  # (batch, output_len, n_features)
        
        # Residual from last input value
        out = out + x[:, -1:, :]  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

