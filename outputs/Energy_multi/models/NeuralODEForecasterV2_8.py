import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralODEForecasterV2(BaseForecastModel):
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
        
        # Config with defaults
        self.latent_dim = self.config.get('latent_dim', 128)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_steps = self.config.get('n_steps', 8)
        self.n_scales = self.config.get('n_scales', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Ensure latent_dim is divisible by n_scales for multi-scale processing
        self.scale_dim = self.latent_dim // self.n_scales
        self.latent_dim = self.scale_dim * self.n_scales  # Adjust to exact multiple
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: (batch, input_len, n_features) -> (batch, latent_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_len * self.n_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )  # (batch, latent_dim)
        
        # Multi-scale temporal encoders: different time constants for different scales
        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.scale_dim),
            ) for _ in range(self.n_scales)
        ])  # Each: (batch, latent_dim) -> (batch, scale_dim)
        
        # Dynamics networks for each scale (Neural ODE functions)
        self.dynamics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.scale_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_dim // 2, self.scale_dim),
            ) for _ in range(self.n_scales)
        ])  # Each: (batch, scale_dim) -> (batch, scale_dim)
        
        # Learnable gating between scales (fast vs slow dynamics)
        self.scale_gates = nn.Parameter(torch.ones(self.n_scales))  # (n_scales,)
        
        # Channel-adaptive evolution rates: each feature evolves at different rate
        self.channel_rates = nn.Parameter(torch.ones(self.n_features))  # (n_features,)
        
        # ODE solver: simple Euler with adaptive step size per scale
        # Time points for forecasting: linearly spaced from 0 to 1
        self.register_buffer('time_points', torch.linspace(0, 1, self.output_len))
        
        # Scale combination: merge multi-scale representations back to latent
        self.scale_combine = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )  # (batch, latent_dim) -> (batch, latent_dim)
        
        # Cross-channel mixing after ODE evolution
        self.channel_mixer = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )  # (batch, latent_dim) -> (batch, latent_dim)
        
        # Output projection: latent -> (output_len, n_features)
        self.output_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_len * self.n_features),
        )  # (batch, output_len * n_features)
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def ode_solve_euler(self, z0, dynamics_fn, t_span, n_steps):
        """
        Simple Euler ODE solver.
        z0: (batch, dim)
        dynamics_fn: callable, computes dz/dt
        t_span: (2,) start and end time
        n_steps: number of integration steps
        Returns: trajectory at t_eval points
        """
        # (batch, dim)
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        
        z = z0  # (batch, dim)
        trajectory = []
        
        for step in range(n_steps):
            dz = dynamics_fn(z)  # (batch, dim)
            z = z + dt * dz  # (batch, dim)
            trajectory.append(z)
        
        # Return final state and sample points
        # For forecasting, we need states at specific output times
        # Interpolate linearly between trajectory points
        return z, trajectory  # z: (batch, dim), trajectory: list of (batch, dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]  # scalar
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store residual for skip connection
        x_last = x[:, -1:, :].repeat(1, self.output_len, 1)  # (batch, output_len, n_features)
        
        # Flatten input and project to latent
        x_flat = x.reshape(batch_size, -1)  # (batch, input_len * n_features)
        z = self.input_proj(x_flat)  # (batch, latent_dim)
        z = F.gelu(z)  # (batch, latent_dim)
        
        # Multi-scale encoding: split latent into scales
        scale_outputs = []
        for i, encoder in enumerate(self.scale_encoders):
            scale_z = encoder(z)  # (batch, scale_dim)
            scale_outputs.append(scale_z)
        
        # Stack scales: (batch, n_scales, scale_dim)
        scale_tensor = torch.stack(scale_outputs, dim=1)  # (batch, n_scales, scale_dim)
        
        # Apply ODE evolution per scale with different effective step sizes
        evolved_scales = []
        for i in range(self.n_scales):
            z_scale = scale_tensor[:, i, :]  # (batch, scale_dim)
            
            # Scale-specific step count: slower scales get more steps (finer resolution)
            # Gate modulates the dynamics magnitude
            gate = torch.sigmoid(self.scale_gates[i])  # scalar
            effective_steps = max(1, int(self.n_steps * (1.0 + 0.5 * i)))  # scalar
            
            # Evolve through ODE
            dynamics_fn = self.dynamics[i]  # nn.Module
            
            # Multiple small steps for stability
            z_evolved = z_scale  # (batch, scale_dim)
            dt = 1.0 / effective_steps  # scalar
            for _ in range(effective_steps):
                dz = dynamics_fn(z_evolved)  # (batch, scale_dim)
                z_evolved = z_evolved + gate * dt * torch.tanh(dz)  # (batch, scale_dim)
            
            evolved_scales.append(z_evolved)
        
        # Concatenate evolved scales back to full latent
        z_evolved = torch.cat(evolved_scales, dim=-1)  # (batch, latent_dim)
        
        # Combine scales with nonlinearity
        z_combined = self.scale_combine(z_evolved)  # (batch, latent_dim)
        z_combined = z_combined + z_evolved  # Residual, (batch, latent_dim)
        
        # Cross-channel mixing: treat latent dimensions as channels
        z_mixed = self.channel_mixer(z_combined)  # (batch, latent_dim)
        z_mixed = z_mixed + z_combined  # Residual, (batch, latent_dim)
        
        # Project to output shape
        out_flat = self.output_proj(z_mixed)  # (batch, output_len * n_features)
        out = out_flat.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Residual connection from last input value (learnable weight)
        out = out + torch.sigmoid(self.residual_weight) * x_last  # (batch, output_len, n_features)
        
        # Denormalize with RevIN
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
