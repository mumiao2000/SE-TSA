import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralControlledDiffusion_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [32, 128, 512],
            'n_layers': [1, 3, 6],
            'solver_steps': [4, 16, 64],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 128)
        self.n_layers = self.config.get('n_layers', 3)
        self.solver_steps = self.config.get('solver_steps', 16)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Initial projection
        self.initial = nn.Linear(self.n_features, self.hidden_dim)
        
        # Vector field network: f(h, x) -> dh/dt
        vector_layers = []
        for i in range(self.n_layers):
            if i == 0:
                vector_layers.append(nn.Linear(self.hidden_dim + self.n_features, self.hidden_dim * 2))
            else:
                vector_layers.append(nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2))
            if i < self.n_layers - 1:
                vector_layers.append(nn.GELU())
                vector_layers.append(nn.Dropout(self.dropout_rate))
        vector_layers.append(nn.Linear(self.hidden_dim * 2, self.hidden_dim))
        self.vector_field = nn.Sequential(*vector_layers)
        
        # Output projection from hidden state
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.n_features)
        )
        
        # Learnable time points for output
        self.output_times = nn.Parameter(torch.linspace(0, 1, self.output_len))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        batch = x.size(0)
        
        # Create natural cubic spline coefficients
        # Simple approach: linear interpolation for derivative estimation
        t = torch.linspace(0, 1, self.input_len, device=x.device)  # (input_len,)
        
        # Estimate derivatives with finite differences
        x_pad = torch.cat([x[:, :1, :], x, x[:, -1:, :]], dim=1)  # (batch, input_len+2, n_features)
        derivs = (x_pad[:, 2:, :] - x_pad[:, :-2, :]) / (2.0 / (self.input_len - 1))  # (batch, input_len, n_features)
        
        # Initial hidden state from last observation
        h0 = self.initial(x[:, -1, :])  # (batch, hidden_dim)
        
        # Solve CDE from t=1 to future times with fixed steps
        # Use simple Euler method with learned vector field
        h = h0  # (batch, hidden_dim)
        
        # We need to integrate from t=1 to t>1 (extrapolation)
        # Scale: input is [0,1], output is [1, 1+alpha]
        t_start = torch.tensor(1.0, device=x.device)  # ()
        t_end = torch.tensor(2.0, device=x.device)  # ()
        dt = (t_end - t_start) / self.solver_steps  # ()
        
        # Store trajectory
        trajectory = [h]
        
        for step in range(self.solver_steps):
            t_curr = t_start + step * dt  # ()
            
            # Interpolate control x at current time
            # Find bracketing indices
            t_scaled = t_curr * (self.input_len - 1)  # ()
            t_scaled = torch.tensor(t_scaled, device=x.device) if isinstance(t_scaled, float) else t_scaled  # ()
            idx = torch.clamp(t_scaled.long(), 0, self.input_len - 2)  # ()
            frac = t_scaled - idx.float()  # ()
            
            # Expand idx to batch dimension for indexing
            idx = idx.long().clamp(0, self.input_len - 2)  # ()
            idx_batch = idx.unsqueeze(0).expand(batch)  # (batch,)
            x_left = x[torch.arange(batch), idx_batch, :]  # (batch, n_features)
            x_right = x[torch.arange(batch), (idx_batch + 1).clamp(0, self.input_len - 1), :]  # (batch, n_features)
            x_t = x_left + frac.view(-1, 1) * (x_right - x_left)  # (batch, n_features)
            
            # For extrapolation beyond input, use last value with trend
            mask = (t_scaled >= self.input_len - 1).float().view(-1, 1)  # (batch, 1)
            trend = derivs[:, -1, :]  # (batch, n_features)
            x_extrap = x[:, -1, :] + (t_scaled.view(-1, 1) - (self.input_len - 1)) * trend / (self.input_len - 1)  # (batch, n_features)
            x_t = x_t * (1 - mask) + x_extrap * mask  # (batch, n_features)
            
            # Compute vector field
            hx = torch.cat([h, x_t], dim=-1)  # (batch, hidden_dim + n_features)
            dh = self.vector_field(hx)  # (batch, hidden_dim)
            
            # Euler step
            h = h + dt * dh  # (batch, hidden_dim)
            trajectory.append(h)
        
        # Sample at output times
        trajectory = torch.stack(trajectory, dim=1)  # (batch, solver_steps+1, hidden_dim)
        traj_times = torch.linspace(t_start.item(), t_end.item(), self.solver_steps + 1, device=x.device)  # (solver_steps+1,)
        
        # Map output_times [0,1] to [t_start, t_end]
        output_times_scaled = t_start + self.output_times * (t_end - t_start)  # (output_len,)
        
        # Interpolate trajectory at output times
        out_h = []
        for ot in output_times_scaled:
            # Find nearest in trajectory
            dists = (traj_times - ot).abs()  # (solver_steps+1,)
            idx = dists.argmin()
            out_h.append(trajectory[:, idx, :])  # (batch, hidden_dim)
        
        out_h = torch.stack(out_h, dim=1)  # (batch, output_len, hidden_dim)
        
        # Project to features
        out = self.readout(out_h)  # (batch, output_len, n_features)
        
        # Add momentum from last observed trend
        last_trend = x[:, -1, :] - x[:, -2, :]  # (batch, n_features)
        momentum = torch.arange(1, self.output_len + 1, device=x.device).float().view(1, self.output_len, 1)
        out = out + last_trend.unsqueeze(1) * momentum * 0.1  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
