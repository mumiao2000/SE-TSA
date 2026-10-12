import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralOrdinaryForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [32, 128, 512],
            'ode_hidden_dim': [64, 256, 1024],
            'n_ode_layers': [2, 3, 5],
            'solver_steps': [10, 20, 50],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.latent_dim = self.config.get('latent_dim', 128)
        self.ode_hidden_dim = self.config.get('ode_hidden_dim', 256)
        self.n_ode_layers = self.config.get('n_ode_layers', 3)
        self.solver_steps = self.config.get('solver_steps', 20)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: maps (input_len, n_features) to latent initial state
        self.encoder = nn.Sequential(
            nn.Linear(self.n_features, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        # ODE dynamics function: dz/dt = f(z, t)
        ode_layers = []
        for i in range(self.n_ode_layers):
            if i == 0:
                ode_layers.append(nn.Linear(self.latent_dim + 1, self.ode_hidden_dim))  # +1 for time
            else:
                ode_layers.append(nn.Linear(self.ode_hidden_dim, self.ode_hidden_dim))
            ode_layers.append(nn.GELU())
            if i < self.n_ode_layers - 1:
                ode_layers.append(nn.Dropout(self.dropout))
        ode_layers.append(nn.Linear(self.ode_hidden_dim, self.latent_dim))
        
        self.ode_func = nn.Sequential(*ode_layers)
        
        # Decoder: maps latent state to observations
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.n_features)
        )
        
        # Time embedding for irregular sampling
        self.time_embed = nn.Linear(1, self.latent_dim // 4)
        
    def ode_step(self, z, t, dt):
        # z: (batch, latent_dim), t: (batch, 1)
        # Simple Euler step: z_{t+dt} = z_t + dt * f(z_t, t)
        
        # Concatenate state and time
        t_input = t.expand(z.size(0), 1)  # (batch, 1)
        zt = torch.cat([z, t_input], dim=-1)  # (batch, latent_dim + 1)
        
        dz = self.ode_func(zt)  # (batch, latent_dim)
        z_next = z + dt * dz  # (batch, latent_dim)
        
        return z_next
        
    def ode_solve(self, z0, t_span, n_steps):
        # z0: (batch, latent_dim)
        # t_span: (2,) start and end times
        # Returns: (batch, n_steps+1, latent_dim)
        
        t0, t1 = t_span
        dt = (t1 - t0) / n_steps
        
        z = z0
        trajectory = [z]
        
        for i in range(n_steps):
            t = t0 + i * dt
            z = self.ode_step(z, torch.tensor([[t]], device=z.device), dt)
            trajectory.append(z)
        
        return torch.stack(trajectory, dim=1)  # (batch, n_steps+1, latent_dim)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.size(0)
        
        # Encode each time step to latent, then pool to initial state
        # Use last observation as most relevant for forecasting
        x_last = x[:, -1, :]  # (batch, n_features)
        
        # Also use temporal average for context
        x_avg = x.mean(dim=1)  # (batch, n_features)
        
        # Combine for encoding
        x_encode = x_last + 0.1 * x_avg  # (batch, n_features)
        
        z0 = self.encoder(x_encode)  # (batch, latent_dim)
        
        # Add time embedding based on position in sequence
        time_feat = self.time_embed(torch.tensor([[self.input_len]], device=x.device, dtype=torch.float32).expand(batch_size, 1))  # (batch, latent_dim//4)
        z0 = z0 + F.pad(time_feat, (0, self.latent_dim - time_feat.size(-1)))  # (batch, latent_dim)
        
        # Solve ODE from input_len to input_len + output_len
        t_span = torch.tensor([0.0, float(self.output_len)], device=x.device)
        
        # Use fewer steps for efficiency, but enough for stability
        n_solve_steps = min(self.solver_steps, self.output_len * 2)
        
        trajectory = self.ode_solve(z0, t_span, n_solve_steps)  # (batch, n_solve_steps+1, latent_dim)
        
        # Sample at output time points
        # Map output indices to trajectory positions
        output_indices = torch.linspace(0, n_solve_steps, self.output_len, device=x.device).long()
        output_indices = torch.clamp(output_indices, 0, n_solve_steps)
        
        z_pred = trajectory[:, output_indices, :]  # (batch, output_len, latent_dim)
        
        # Decode to observations
        out = self.decoder(z_pred)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

