import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralODEContinuousForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [64, 256, 1024],
            'ode_hidden': [128, 512, 2048],
            'n_ode_layers': [2, 4, 8],
            'solver_steps': [4, 16, 64],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 256)
        self.ode_hidden = self.config.get('ode_hidden', 512)
        self.n_ode_layers = self.config.get('n_ode_layers', 4)
        self.solver_steps = self.config.get('solver_steps', 16)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Normalization
        self.input_norm = nn.LayerNorm(n_features)
        
        # Encoder: RNN-based to get initial state
        # Bidirectional GRU: num_layers=2, hidden=latent_dim//2 per direction
        # Output h_n shape: (num_layers * 2, batch, latent_dim // 2) = (4, batch, latent_dim//2)
        self.encoder_rnn = nn.GRU(n_features, self.latent_dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=self.dropout)
        
        # Initial state projection: takes concatenated last layer forward+backward = latent_dim
        self.h0_proj = nn.Linear(self.latent_dim, self.latent_dim)
        
        # ODE dynamics function: dz/dt = f(z, t)
        ode_layers = []
        in_dim = self.latent_dim + 1  # +1 for time
        for i in range(self.n_ode_layers):
            out_dim = self.ode_hidden if i < self.n_ode_layers - 1 else self.latent_dim
            ode_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < self.n_ode_layers - 1 else nn.Identity(),
                nn.Tanh() if i < self.n_ode_layers - 1 else nn.Identity(),
                nn.Dropout(self.dropout) if i < self.n_ode_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        self.dynamics = nn.Sequential(*ode_layers)
        
        # Decoder: from latent state to output features at each time
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.ode_hidden // 2),
            nn.LayerNorm(self.ode_hidden // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ode_hidden // 2, n_features)
        )
        
        # Time embedding for decoder
        self.time_embed = nn.Linear(1, self.latent_dim // 4)
        
        # Direct trend model for stability
        self.trend = nn.Linear(input_len, output_len)
        
    def ode_step(self, z, t, dt):
        # RK4 integration step
        # z: (batch, latent_dim), t: scalar
        t_batch = torch.full((z.shape[0], 1), t, device=z.device)  # (batch, 1)
        
        # k1
        inp = torch.cat([z, t_batch], dim=-1)  # (batch, latent_dim + 1)
        k1 = self.dynamics(inp)  # (batch, latent_dim)
        
        # k2
        z2 = z + dt * k1 / 2
        t2 = t + dt / 2
        t_batch2 = torch.full((z.shape[0], 1), t2, device=z.device)
        inp2 = torch.cat([z2, t_batch2], dim=-1)
        k2 = self.dynamics(inp2)
        
        # k3
        z3 = z + dt * k2 / 2
        t3 = t + dt / 2
        t_batch3 = torch.full((z.shape[0], 1), t3, device=z.device)
        inp3 = torch.cat([z3, t_batch3], dim=-1)
        k3 = self.dynamics(inp3)
        
        # k4
        z4 = z + dt * k3
        t4 = t + dt
        t_batch4 = torch.full((z.shape[0], 1), t4, device=z.device)
        inp4 = torch.cat([z4, t_batch4], dim=-1)
        k4 = self.dynamics(inp4)
        
        return z + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        B = x.shape[0]
        
        # Normalize
        x = self.input_norm(x)  # (batch, input_len, n_features)
        
        # Encode to initial latent state
        _, h_n = self.encoder_rnn(x)  # h_n: (num_layers * 2, batch, latent_dim // 2) = (4, batch, latent_dim//2)
        
        # Extract last layer's forward and backward hidden states
        # h_n[2] = last layer forward, h_n[3] = last layer backward
        # Each has shape (batch, latent_dim // 2)
        h_n_forward = h_n[2]   # (batch, latent_dim // 2)
        h_n_backward = h_n[3]  # (batch, latent_dim // 2)
        h_n = torch.cat([h_n_forward, h_n_backward], dim=-1)  # (batch, latent_dim)
        
        z0 = self.h0_proj(h_n)  # (batch, latent_dim)
        z0 = torch.tanh(z0)
        
        # Solve ODE from t=0 to t=output_len
        # Time points for prediction
        t_span = 1.0  # normalize prediction horizon
        dt = t_span / self.solver_steps
        
        # Store trajectory
        z = z0
        trajectory = [z]
        t = 0.0
        for _ in range(self.solver_steps):
            z = self.ode_step(z, t, dt)
            trajectory.append(z)
            t += dt
        
        # Sample at output time points
        # Map output_len discrete steps to trajectory indices
        indices = torch.linspace(0, self.solver_steps, self.output_len + 1, device=x.device)[1:].long() - 1
        indices = torch.clamp(indices, 0, self.solver_steps - 1)
        
        # Get states at these indices
        trajectory = torch.stack(trajectory, dim=0)  # (solver_steps+1, batch, latent_dim)
        z_out = trajectory[indices]  # (output_len, batch, latent_dim)
        z_out = z_out.transpose(0, 1)  # (batch, output_len, latent_dim)
        
        # Add time embeddings
        time_vals = torch.arange(self.output_len, device=x.device).float().unsqueeze(0).unsqueeze(-1) / self.output_len  # (1, output_len, 1)
        time_emb = self.time_embed(time_vals)  # (1, output_len, latent_dim // 4)
        time_emb = time_emb.expand(B, -1, -1)
        
        # Pad time_emb to match latent_dim
        if time_emb.shape[-1] < self.latent_dim:
            padding = torch.zeros(B, self.output_len, self.latent_dim - time_emb.shape[-1], device=x.device)
            time_emb = torch.cat([time_emb, padding], dim=-1)
        
        z_out = z_out + time_emb * 0.1  # weak time conditioning
        
        # Decode
        out = self.decoder(z_out.reshape(B * self.output_len, -1))  # (batch * output_len, n_features)
        out = out.reshape(B, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Add trend from direct model
        trend_in = x.transpose(1, 2)  # (batch, n_features, input_len)
        trend_out = self.trend(trend_in)  # (batch, n_features, output_len)
        trend_out = trend_out.transpose(1, 2)  # (batch, output_len, n_features)
        
        out = out + trend_out * 0.5  # residual with trend
        
        return out

