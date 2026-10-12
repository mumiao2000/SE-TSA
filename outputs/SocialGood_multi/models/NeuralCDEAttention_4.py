import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralCDEAttention(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_layers': [1, 3, 6],
            'attention_heads': [2, 8, 32],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 3)
        self.attention_heads = self.config.get('attention_heads', 8)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Initial projection: (batch, input_len, n_features) -> coefficients for CDE
        self.initial_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # CDE vector field: f(z, t) -> dz/dt
        # We use a neural network that takes [z, t] and outputs dz/dt
        vector_field_layers = []
        in_dim = self.hidden_dim + 1  # +1 for time
        for i in range(self.n_layers):
            out_dim = self.hidden_dim if i < self.n_layers - 1 else self.hidden_dim * self.n_features
            vector_field_layers.extend([
                nn.Linear(in_dim, self.hidden_dim * 2),
                nn.LayerNorm(self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 2, out_dim),
            ])
            if i < self.n_layers - 1:
                vector_field_layers.extend([
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                ])
            in_dim = out_dim
        self.vector_field = nn.Sequential(*vector_field_layers)
        
        # Attention for reading from hidden state
        self.query_embed = nn.Parameter(torch.randn(self.output_len, self.hidden_dim))
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, 
            self.attention_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
    def _hermite_cubic_coefficients(self, x):
        # x: (batch, input_len, n_features)
        # Compute cubic spline coefficients
        # Simple approach: use finite differences for derivatives
        batch_size = x.shape[0]
        
        # Time points (normalized)
        t = torch.linspace(0, 1, self.input_len, device=x.device)  # (input_len,)
        
        # Compute derivatives using central differences
        dx = torch.zeros_like(x)
        dx[:, 1:-1, :] = (x[:, 2:, :] - x[:, :-2, :]) / 2.0  # (batch, input_len-2, n_features)
        dx[:, 0, :] = x[:, 1, :] - x[:, 0, :]
        dx[:, -1, :] = x[:, -1, :] - x[:, -2, :]
        
        # Coefficients for Hermite cubic: a + b*t + c*t^2 + d*t^3 on each interval
        # We'll use a simpler approach: just store values and use linear interpolation for CDE
        return x, dx, t
    
    def _evaluate_path(self, x, dx, t, t_query):
        # Evaluate path at t_query using linear interpolation
        # x: (batch, input_len, n_features), dx: (batch, input_len, n_features), t: (input_len,)
        # t_query: scalar or (batch,)
        
        # Find bracketing indices
        idx = torch.searchsorted(t, t_query.clamp(t[0], t[-1]))
        idx = idx.clamp(1, len(t) - 1)
        
        t0 = t[idx - 1]
        t1 = t[idx]
        
        # Linear interpolation
        alpha = (t_query - t0) / (t1 - t0 + 1e-8)
        
        if x.dim() == 3:
            x0 = x[:, idx - 1, :]  # (batch, n_features)
            x1 = x[:, idx, :]  # (batch, n_features)
            x_query = x0 + alpha.unsqueeze(-1) * (x1 - x0)  # (batch, n_features)
        else:
            x0 = x[idx - 1]
            x1 = x[idx]
            x_query = x0 + alpha * (x1 - x0)
        
        return x_query
    
    def _ode_solve_rk4(self, z0, x_path, dx_path, t_path, t_span, n_steps=20):
        # Solve dz/dt = f(z, t) * dx/dt using RK4
        # z0: (batch, hidden_dim)
        # Returns z at times in t_span
        
        batch_size = z0.shape[0]
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        
        z = z0
        trajectory = [z]
        
        for step in range(n_steps):
            t_curr = t_start + step * dt
            
            # Evaluate path derivative at current time
            dx_dt = self._evaluate_path(dx_path, dx_path, t_path, torch.tensor(t_curr, device=z.device))
            if dx_dt.dim() == 2:
                dx_dt = dx_dt.unsqueeze(1)  # (batch, 1, n_features)
            
            # RK4 steps
            def f_eval(z_in, t_in):
                t_in_expanded = torch.full((batch_size, 1), t_in, device=z.device)
                zt = torch.cat([z_in, t_in_expanded], dim=-1)  # (batch, hidden_dim + 1)
                vf = self.vector_field(zt)  # (batch, hidden_dim * n_features)
                vf = vf.reshape(batch_size, self.hidden_dim, self.n_features)  # (batch, hidden_dim, n_features)
                # Contract with dx/dt
                dz = torch.bmm(vf, dx_dt.permute(0, 2, 1)).squeeze(-1)  # (batch, hidden_dim)
                return dz
            
            k1 = f_eval(z, t_curr)
            k2 = f_eval(z + dt * k1 / 2, t_curr + dt / 2)
            k3 = f_eval(z + dt * k2 / 2, t_curr + dt / 2)
            k4 = f_eval(z + dt * k3, t_curr + dt)
            
            z = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory.append(z)
        
        return torch.stack(trajectory, dim=1)  # (batch, n_steps+1, hidden_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Compute path and derivatives
        x_path, dx_path, t_path = self._hermite_cubic_coefficients(x)
        
        # Initial condition
        z0 = self.initial_proj(x[:, 0, :])  # (batch, hidden_dim)
        
        # Solve CDE from t=0 to t=1 (normalized time)
        t_span = (0.0, 1.0)
        z_trajectory = self._ode_solve_rk4(z0, x_path, dx_path, t_path, t_span, n_steps=self.input_len)
        # (batch, input_len+1, hidden_dim)
        
        # Use attention to read out forecasts
        # Queries are learned embeddings for each output timestep
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, output_len, hidden_dim)
        
        # Keys and values from CDE trajectory
        keys = z_trajectory[:, 1:, :]  # (batch, input_len, hidden_dim)
        values = keys  # (batch, input_len, hidden_dim)
        
        # Cross-attention
        attn_output, _ = self.attention(queries, keys, values)  # (batch, output_len, hidden_dim)
        
        # Output projection
        output = self.output_norm(attn_output)  # (batch, output_len, hidden_dim)
        output = self.output_proj(output)  # (batch, output_len, n_features)
        
        output = self.revin(output, mode='denorm')  # (batch, output_len, n_features)
        return output
