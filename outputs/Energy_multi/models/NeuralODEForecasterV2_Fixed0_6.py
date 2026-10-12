import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralODEForecasterV2_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [32, 128, 512],
            'hidden_dim': [64, 256, 1024],
            'n_steps': [4, 8, 16],
            'n_scales': [2, 3, 4],
            'dropout': [0.0, 0.1, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 128)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_steps = self.config.get('n_steps', 8)
        self.n_scales = self.config.get('n_scales', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Compute per-scale latent dimensions that sum exactly to latent_dim
        base_latent = self.latent_dim // self.n_scales
        remainder = self.latent_dim % self.n_scales
        self.scale_latent_dims = []
        for i in range(self.n_scales):
            dim = base_latent + (1 if i < remainder else 0)
            self.scale_latent_dims.append(dim)
        
        # Multi-scale temporal encoder with dilated convolutions
        self.scale_encoders = nn.ModuleList()
        for i in range(self.n_scales):
            dilation = 2 ** i
            # Ensure kernel size is valid with dilation
            max_kernel = input_len // dilation if dilation <= input_len else 0
            kernel_size = min(3, max_kernel) if max_kernel >= 1 else 1
            if kernel_size < 1:
                kernel_size = 1
            padding = (kernel_size - 1) * dilation // 2
            self.scale_encoders.append(
                nn.Conv1d(
                    self.n_features, 
                    self.scale_latent_dims[i],
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding
                )
            )
        
        # Feature projection to combine scales
        total_scale_dim = sum(self.scale_latent_dims)
        self.scale_combine = nn.Linear(total_scale_dim, total_scale_dim)
        self.scale_gate = nn.Sequential(
            nn.Linear(total_scale_dim, total_scale_dim),
            nn.Sigmoid()
        )
        
        # ODE dynamics functions for each scale (hierarchical)
        self.ode_funcs = nn.ModuleList()
        for i in range(self.n_scales):
            scale_latent = self.scale_latent_dims[i]
            self.ode_funcs.append(nn.Sequential(
                nn.Linear(scale_latent, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, scale_latent)
            ))
        
        # Cross-scale interaction
        self.cross_scale = nn.Linear(total_scale_dim, total_scale_dim)
        
        # Decoder: project latent to output features per time step
        self.decoder = nn.Sequential(
            nn.Linear(total_scale_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
        # Learnable time embeddings for output positions
        self.time_embed = nn.Parameter(torch.randn(self.output_len, total_scale_dim) * 0.02)
        
    def ode_step(self, z, dt, scale_idx):
        # Simple Euler integration step
        # z: (batch, latent_dim_per_scale)
        dz = self.ode_funcs[scale_idx](z)  # (batch, latent_dim_per_scale)
        return z + dt * dz  # (batch, latent_dim_per_scale)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Multi-scale encoding
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        scale_features = []
        for i in range(self.n_scales):
            sf = self.scale_encoders[i](x_perm)  # (batch, scale_latent_dims[i], input_len)
            # Global average pooling over time
            sf = sf.mean(dim=-1)  # (batch, scale_latent_dims[i])
            scale_features.append(sf)
        
        # Concatenate scales
        z = torch.cat(scale_features, dim=-1)  # (batch, total_scale_dim)
        
        # Gated combination
        gate = self.scale_gate(z)  # (batch, total_scale_dim)
        z = gate * self.scale_combine(z) + (1 - gate) * z  # (batch, total_scale_dim)
        z = F.relu(z)  # (batch, total_scale_dim)
        
        # Split into scales for ODE evolution
        z_scales = torch.split(z, self.scale_latent_dims, dim=-1)  # list of (batch, scale_latent_dims[i])
        
        # Evolve each scale with its own ODE
        dt = 1.0 / self.n_steps
        evolved_scales = []
        for i, zs in enumerate(z_scales):
            zs_evolved = zs
            for _ in range(self.n_steps):
                zs_evolved = self.ode_step(zs_evolved, dt, i)  # (batch, scale_latent_dims[i])
            evolved_scales.append(zs_evolved)
        
        # Recombine evolved scales
        z_evolved = torch.cat(evolved_scales, dim=-1)  # (batch, total_scale_dim)
        
        # Cross-scale interaction
        z_evolved = z_evolved + self.cross_scale(z_evolved)  # (batch, total_scale_dim)
        
        # Decode to forecast: use time embeddings for each horizon
        # Expand latent for all output steps
        z_expanded = z_evolved.unsqueeze(1)  # (batch, 1, total_scale_dim)
        z_expanded = z_expanded.expand(-1, self.output_len, -1)  # (batch, output_len, total_scale_dim)
        
        # Add time embeddings
        time_emb = self.time_embed.unsqueeze(0)  # (1, output_len, total_scale_dim)
        z_time = z_expanded + time_emb  # (batch, output_len, total_scale_dim)
        
        # Decode
        out = self.decoder(z_time)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

