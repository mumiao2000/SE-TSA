import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralOrdinaryForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'dynamics_hidden': [64, 256, 1024],
            'n_steps': [4, 12, 32],
            'time_embed_dim': [8, 32, 128],
            'dynamics_depth': [1, 3, 6]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.dynamics_hidden = self.config.get('dynamics_hidden', 256)
        self.n_steps = self.config.get('n_steps', 12)
        self.time_embed_dim = self.config.get('time_embed_dim', 32)
        self.dynamics_depth = self.config.get('dynamics_depth', 3)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )
        
        # Initial state encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.n_features, self.dynamics_hidden),
            nn.SiLU(),
            nn.Linear(self.dynamics_hidden, self.dynamics_hidden)
        )
        
        # Dynamics network: dz/dt = f(z, t)
        dynamics_layers = []
        for i in range(self.dynamics_depth):
            in_dim = self.dynamics_hidden + self.time_embed_dim if i == 0 else self.dynamics_hidden
            out_dim = self.dynamics_hidden
            dynamics_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.SiLU() if i < self.dynamics_depth - 1 else nn.Identity()
            ])
        self.dynamics = nn.Sequential(*dynamics_layers)
        
        # Decoder: state to features
        self.decoder = nn.Sequential(
            nn.Linear(self.dynamics_hidden, self.dynamics_hidden // 2),
            nn.SiLU(),
            nn.Linear(self.dynamics_hidden // 2, self.n_features)
        )
        
        # Learnable initial state transformation from input
        self.init_transform = nn.GRU(
            input_size=self.n_features,
            hidden_size=self.dynamics_hidden,
            batch_first=True
        )

    def _step_dynamics(self, z, t_embed, dt):
        # Euler step: z_{t+1} = z_t + dt * f(z_t, t)
        zt = torch.cat([z, t_embed], dim=-1)  # (batch, dynamics_hidden + time_embed_dim)
        dz = self.dynamics(zt)  # (batch, dynamics_hidden)
        return z + dt * dz  # (batch, dynamics_hidden)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Encode input to initial state
        _, h_n = self.init_transform(x)  # h_n: (1, batch, dynamics_hidden)
        z = h_n.squeeze(0)  # (batch, dynamics_hidden)
        
        # Generate predictions by integrating forward in time
        predictions = []
        
        # Time span for prediction [0, 1]
        dt = 1.0 / self.n_steps
        
        for step in range(self.output_len):
            # Sub-steps for each output timestep
            t_start = step / self.output_len
            for sub_step in range(self.n_steps):
                t = t_start + sub_step * dt / self.output_len
                t_tensor = torch.tensor([[t]], device=x.device).expand(batch_size, 1)  # (batch, 1)
                t_embed = self.time_embed(t_tensor)  # (batch, time_embed_dim)
                z = self._step_dynamics(z, t_embed, dt / self.output_len)  # (batch, dynamics_hidden)
            
            # Decode state to prediction
            pred = self.decoder(z)  # (batch, n_features)
            predictions.append(pred)
        
        out = torch.stack(predictions, dim=1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

