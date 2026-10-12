import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralControlledDiffusion_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'vector_field_depth': [2, 4, 8],
            'signature_level': [2, 4, 8],
            'n_coefficients': [16, 64, 256],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.vector_field_depth = self.config.get('vector_field_depth', 4)
        self.signature_level = self.config.get('signature_level', 4)
        self.n_coefficients = self.config.get('n_coefficients', 64)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Compute solve steps for consistent dimensions
        self.n_solve_steps = max(self.input_len, self.output_len * 2)
        
        # Advanced normalization: trend-aware normalization
        self.trend_extractor = nn.Linear(self.input_len, 1)
        self.seasonal_norm = nn.LayerNorm(self.n_features)
        
        # Initial lift: map observations to hidden space
        self.initial_lift = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout)
        )
        
        # Driving signal encoder: RNN that generates the control path
        self.driving_rnn = nn.GRU(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout if self.vector_field_depth > 2 else 0
        )
        
        # Vector field: f(h, t) -> dh/dt, parameterized as MLP
        vector_layers = []
        in_dim = self.hidden_dim + 1  # hidden state + time
        for d in range(self.vector_field_depth):
            out_dim = self.hidden_dim if d == self.vector_field_depth - 1 else self.hidden_dim * 2
            vector_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = out_dim
        self.vector_field = nn.Sequential(*vector_layers)
        
        # Signature approximation: learnable projection of path features
        # Input dim is n_solve_steps * hidden_dim (trajectory excluding initial state)
        self.signature_proj = nn.Sequential(
            nn.Linear(self.n_solve_steps * self.hidden_dim, self.n_coefficients),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_coefficients, self.hidden_dim)
        )
        
        # Readout: map per-timestep hidden state to forecast
        # Output per timestep: n_features
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.n_features)
        )
        
        # Temporal attention for output generation
        self.temporal_query = nn.Parameter(torch.randn(self.output_len, self.hidden_dim))
        self.temporal_attn = nn.MultiheadAttention(self.hidden_dim, num_heads=4, batch_first=True, dropout=self.dropout)
        
    def ode_solve(self, h0, control, n_steps):
        # Simple Euler solver for neural CDE
        # h0: (batch, hidden_dim)
        # control: (batch, input_len, hidden_dim) - driving signal
        batch = h0.shape[0]
        h = h0  # (batch, hidden_dim)
        
        # Interpolate control to finer grid
        control_t = control.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        control_interp = F.interpolate(control_t, size=n_steps, mode='linear', align_corners=False)
        control_interp = control_interp.permute(0, 2, 1)  # (batch, n_steps, hidden_dim)
        
        dt = 1.0 / n_steps
        trajectory = [h]
        
        for t in range(n_steps):
            time_feat = torch.full((batch, 1), t * dt, device=h.device)  # (batch, 1)
            h_time = torch.cat([h, time_feat], dim=-1)  # (batch, hidden_dim + 1)
            
            # Vector field evaluation
            dh = self.vector_field(h_time)  # (batch, hidden_dim)
            
            # Controlled by driving signal
            control_t_step = control_interp[:, t, :]  # (batch, hidden_dim)
            
            # Update: dh = f(h, t) * control
            h = h + dh * control_t_step * dt  # (batch, hidden_dim)
            trajectory.append(h)
        
        return torch.stack(trajectory, dim=1)  # (batch, n_steps + 1, hidden_dim)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch = x.shape[0]
        
        # Trend-aware normalization
        trend = self.trend_extractor(x.transpose(1, 2)).squeeze(-1)  # (batch, n_features)
        x_detrended = x - trend.unsqueeze(1)  # (batch, input_len, n_features)
        x_norm = self.seasonal_norm(x_detrended)  # (batch, input_len, n_features)
        
        # Initial lift
        h0 = self.initial_lift(x_norm[:, 0, :])  # (batch, hidden_dim)
        
        # Generate driving signal
        driving, _ = self.driving_rnn(x_norm)  # (batch, input_len, hidden_dim)
        
        # Solve neural CDE
        trajectory = self.ode_solve(h0, driving, self.n_solve_steps)  # (batch, n_solve_steps + 1, hidden_dim)
        
        # Extract signature-like features from trajectory (exclude initial state)
        traj_flat = trajectory[:, 1:, :].reshape(batch, -1)  # (batch, n_solve_steps * hidden_dim)
        signature = self.signature_proj(traj_flat)  # (batch, hidden_dim)
        
        # Temporal attention for output
        # Use trajectory as keys/values, learned queries for output timesteps
        queries = self.temporal_query.unsqueeze(0).expand(batch, -1, -1)  # (batch, output_len, hidden_dim)
        step_size = max(1, self.n_solve_steps // self.input_len)
        trajectory_attend = trajectory[:, 1::step_size, :]  # (batch, ~input_len, hidden_dim)
        if trajectory_attend.shape[1] < 2:
            trajectory_attend = trajectory[:, -2:, :]  # (batch, 2, hidden_dim)
        
        attn_out, _ = self.temporal_attn(queries, trajectory_attend, trajectory_attend)  # (batch, output_len, hidden_dim)
        
        # Combine with signature
        combined = attn_out + signature.unsqueeze(1)  # (batch, output_len, hidden_dim)
        
        # Readout: directly map each output timestep to n_features
        output = self.readout(combined)  # (batch, output_len, n_features)
        
        # Add back trend
        output = output + trend.unsqueeze(1)  # (batch, output_len, n_features)
        
        return output
