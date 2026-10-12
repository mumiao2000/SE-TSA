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
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 128)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_steps = self.config.get('n_steps', 8)
        self.n_scales = self.config.get('n_scales', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Ensure latent_dim is divisible by n_scales
        self.scale_dim = self.latent_dim // self.n_scales
        self.latent_dim = self.scale_dim * self.n_scales  # Adjust to exact multiple
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to multi-scale latent space
        self.input_proj = nn.Linear(self.n_features, self.latent_dim)
        
        # Multi-scale temporal encodings (learned positional embeddings at different scales)
        # FIX: Use nn.ParameterList instead of nn.ModuleList for parameter storage
        self.scale_encoders = nn.ParameterList([
            nn.Parameter(torch.randn(1, input_len // (2**i) + 1, self.scale_dim) * 0.02)
            for i in range(self.n_scales)
        ])
        
        # ODE dynamics function with channel-aware gating
        self.dynamics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.scale_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.scale_dim),
            )
            for _ in range(self.n_scales)
        ])
        
        # Channel gating: learns per-feature importance for dynamics
        self.channel_gate = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.n_scales),
        )
        
        # Scale mixing with attention
        self.scale_mix = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=max(1, self.latent_dim // 64),
            batch_first=True,
            dropout=self.dropout,
        )
        
        # Output projection layers
        self.output_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features),
        )
        
        # Residual connection for direct path
        self.direct_proj = nn.Linear(input_len, output_len)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def ode_solve(self, z0, dynamics_fn, t_span, n_steps):
        # Simple Euler integration with fixed steps
        # z0: (batch, scale_dim)
        dt = (t_span[1] - t_span[0]) / n_steps
        z = z0
        for _ in range(n_steps):
            dz = dynamics_fn(z)  # (batch, scale_dim)
            z = z + dt * dz  # (batch, scale_dim)
        return z  # (batch, scale_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Direct residual path
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        direct_out = self.direct_proj(x_perm)  # (batch, n_features, output_len)
        direct_out = direct_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Project to latent space
        z = self.input_proj(x)  # (batch, input_len, latent_dim)
        
        # Extract multi-scale initial conditions by average pooling
        z_scales = []
        for i in range(self.n_scales):
            pool_size = 2 ** i
            if pool_size <= x.shape[1]:
                z_pooled = F.avg_pool1d(
                    z[:, :, i*self.scale_dim:(i+1)*self.scale_dim].permute(0, 2, 1),
                    kernel_size=pool_size,
                    stride=pool_size
                ).permute(0, 2, 1)  # (batch, pooled_len, scale_dim)
                # Take last timestep as initial condition
                z0 = z_pooled[:, -1, :]  # (batch, scale_dim)
            else:
                z0 = z[:, -1, i*self.scale_dim:(i+1)*self.scale_dim]  # (batch, scale_dim)
            z_scales.append(z0)
        
        # Compute channel gating weights from input statistics
        x_mean = x.mean(dim=1)  # (batch, n_features)
        gate_weights = F.softmax(self.channel_gate(x_mean), dim=-1)  # (batch, n_scales)
        
        # Evolve each scale with ODE
        t_span = (0.0, float(self.output_len))
        evolved_scales = []
        for i in range(self.n_scales):
            z_evolved = self.ode_solve(
                z_scales[i],
                self.dynamics[i],
                t_span,
                max(1, self.n_steps // (i + 1))  # Coarser steps for slower scales
            )  # (batch, scale_dim)
            evolved_scales.append(z_evolved)
        
        # Stack and apply gating
        evolved_stack = torch.stack(evolved_scales, dim=1)  # (batch, n_scales, scale_dim)
        gate_weights = gate_weights.unsqueeze(-1)  # (batch, n_scales, 1)
        gated = evolved_stack * gate_weights  # (batch, n_scales, scale_dim)
        
        # Flatten to full latent
        z_combined = gated.reshape(batch_size, self.latent_dim)  # (batch, latent_dim)
        
        # Expand to output sequence length with learned temporal pattern
        z_expanded = z_combined.unsqueeze(1).expand(-1, self.output_len, -1)  # (batch, output_len, latent_dim)
        
        # Add temporal encoding for output positions
        t_enc = torch.arange(self.output_len, device=x.device).float().unsqueeze(0).unsqueeze(-1)  # (1, output_len, 1)
        t_enc = t_enc / self.output_len  # Normalize to [0, 1]
        z_expanded = z_expanded + 0.1 * torch.sin(t_enc * 2 * 3.14159 * torch.arange(1, self.latent_dim + 1, device=x.device).float().view(1, 1, -1)[:1, :1, :self.latent_dim])
        
        # Self-attention for temporal dependencies
        z_attended, _ = self.scale_mix(z_expanded, z_expanded, z_expanded)  # (batch, output_len, latent_dim)
        z_attended = z_attended + z_expanded  # Residual
        
        # Project to output features
        out = self.output_proj(z_attended)  # (batch, output_len, n_features)
        
        # Add residual connection with learned weight
        out = out + self.residual_weight * direct_out  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)

