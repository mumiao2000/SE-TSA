import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralODEGatedForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [32, 128, 512],
            'ode_hidden_dim': [64, 256, 1024],
            'n_ode_layers': [1, 3, 6],
            'gate_dim': [16, 32, 64],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 128)
        self.ode_hidden_dim = self.config.get('ode_hidden_dim', 256)
        self.n_ode_layers = self.config.get('n_ode_layers', 3)
        self.gate_dim = self.config.get('gate_dim', 32)
        self.dropout = self.config.get('dropout', 0.3)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: map input sequence to initial latent state
        self.encoder = nn.LSTM(self.n_features, self.latent_dim, batch_first=True, bidirectional=True)
        self.encoder_proj = nn.Linear(self.latent_dim * 2, self.latent_dim)
        
        # ODE dynamics function (latent_dim -> latent_dim)
        self.ode_fcs = nn.ModuleList()
        self.ode_norms = nn.ModuleList()
        for i in range(self.n_ode_layers):
            in_dim = self.latent_dim if i == 0 else self.ode_hidden_dim
            out_dim = self.ode_hidden_dim if i < self.n_ode_layers - 1 else self.latent_dim
            self.ode_fcs.append(nn.Linear(in_dim, out_dim))
            if i < self.n_ode_layers - 1:
                self.ode_norms.append(nn.LayerNorm(out_dim))
        
        # Channel-adaptive gating for ODE dynamics
        self.ode_gate = nn.Sequential(
            nn.Linear(self.latent_dim + 1, self.gate_dim),  # +1 for time
            nn.GELU(),
            nn.Linear(self.gate_dim, self.latent_dim),
            nn.Sigmoid()
        )
        
        # Time embedding
        self.time_embed = nn.Linear(1, self.latent_dim // 4)
        
        # Decoder: latent state to predictions at each time
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.latent_dim // 4, 256),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        
        # Channel-specific output with gating
        self.channel_gate = nn.Sequential(
            nn.Linear(128, self.gate_dim),
            nn.GELU(),
            nn.Linear(self.gate_dim, self.n_features),
            nn.Sigmoid()
        )
        self.channel_out = nn.Linear(128, self.n_features)
        
        # Discrete correction branch
        self.discrete_branch = nn.Sequential(
            nn.Conv1d(self.n_features, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, self.n_features, kernel_size=3, padding=1),
        )
        
        # Direct mapping
        self.direct = nn.Linear(self.input_len, self.output_len)
        
    def ode_dynamics(self, z, t):
        # z: (batch, latent_dim), t: (batch, 1)
        # Gate based on current state and time
        gate_input = torch.cat([z, t], dim=-1)  # (batch, latent_dim + 1)
        gate = self.ode_gate(gate_input)  # (batch, latent_dim)
        
        h = z
        for i, (fc, norm) in enumerate(zip(self.ode_fcs, self.ode_norms)):
            h = fc(h)  # (batch, hidden_dim)
            h = norm(h)  # (batch, hidden_dim)
            h = F.gelu(h)  # (batch, hidden_dim)
        
        # Last layer without norm
        h = self.ode_fcs[-1](h)  # (batch, latent_dim)
        
        # Apply gate to dynamics
        return h * gate  # (batch, latent_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Direct residual
        x_direct = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_direct = self.direct(x_direct)  # (batch, n_features, output_len)
        x_direct = x_direct.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Discrete correction
        x_disc = self.discrete_branch(x.permute(0, 2, 1))  # (batch, n_features, input_len)
        x_disc = F.interpolate(x_disc, size=self.output_len, mode='linear', align_corners=False)  # (batch, n_features, output_len)
        x_disc = x_disc.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Encode to latent
        _, (h_n, _) = self.encoder(x)  # h_n: (2, batch, latent_dim)
        h_n = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch, latent_dim * 2)
        z0 = self.encoder_proj(h_n)  # (batch, latent_dim)
        
        # Solve ODE from t=0 to t=1 (normalized time)
        # Use simple Euler integration with fixed steps
        n_steps = 10
        dt = 1.0 / n_steps
        
        # Generate predictions at output time points
        batch_size = x.size(0)
        predictions = []
        
        # Target times for output
        target_times = torch.linspace(0, 1, self.output_len + 1, device=x.device)[1:]  # (output_len,)
        
        # Integrate and sample
        z = z0  # (batch, latent_dim)
        t = torch.zeros(batch_size, 1, device=x.device)  # (batch, 1)
        
        pred_idx = 0
        for step in range(n_steps):
            t_next = t + dt  # (batch, 1)
            
            # RK4 step
            k1 = self.ode_dynamics(z, t)  # (batch, latent_dim)
            k2 = self.ode_dynamics(z + 0.5 * dt * k1, t + 0.5 * dt)  # (batch, latent_dim)
            k3 = self.ode_dynamics(z + 0.5 * dt * k2, t + 0.5 * dt)  # (batch, latent_dim)
            k4 = self.ode_dynamics(z + dt * k3, t_next)  # (batch, latent_dim)
            
            z = z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)  # (batch, latent_dim)
            t = t_next  # (batch, 1)
            
            # Check if we should output at this time
            while pred_idx < self.output_len and target_times[pred_idx].item() <= t[0, 0].item() + 1e-6:
                # Decode
                t_embed = self.time_embed(t)  # (batch, latent_dim // 4)
                z_decode = torch.cat([z, t_embed], dim=-1)  # (batch, latent_dim + latent_dim // 4)
                h_dec = self.decoder(z_decode)  # (batch, 128)
                
                # Channel gating
                gate = self.channel_gate(h_dec)  # (batch, n_features)
                out_t = self.channel_out(h_dec) * gate  # (batch, n_features)
                
                predictions.append(out_t)
                pred_idx += 1
        
        # Pad if needed (shouldn't happen with proper integration)
        while len(predictions) < self.output_len:
            predictions.append(predictions[-1] if predictions else torch.zeros(batch_size, self.n_features, device=x.device))
        
        out = torch.stack(predictions[:self.output_len], dim=1)  # (batch, output_len, n_features)
        
        # Add discrete correction and residual
        out = out + x_disc * 0.2 + x_direct * 0.1  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

