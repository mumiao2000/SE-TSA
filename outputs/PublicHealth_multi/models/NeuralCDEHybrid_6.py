import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralCDEHybrid(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [32, 128, 512],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3],
            'solver_steps': [4, 16, 64]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 128)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.solver_steps = self.config.get('solver_steps', 16)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Initial lift: map features to hidden dimension
        self.initial = nn.Linear(n_features, self.hidden_dim)
        
        # CDE vector field: f(h, x) defines dh/dt = f(h, x(t))
        # We use a simple MLP: input is [h, x], output is dh/dt
        vector_fields = []
        for _ in range(self.n_layers):
            vector_fields.append(nn.Sequential(
                nn.Linear(self.hidden_dim + n_features, self.hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim)
            ))
        self.vector_fields = nn.ModuleList(vector_fields)
        
        # Layer norms for stability
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layers)])
        
        # Readout: map hidden state to output features
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, n_features * output_len)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size, seq_len, n_feat = x.shape
        
        # Create time points
        t = torch.linspace(0, 1, seq_len, device=x.device)  # (input_len,)
        
        # Simple linear interpolation for path X(t)
        # For CDE, we need to evaluate X at arbitrary points
        def interpolate_x(t_query):
            # t_query: scalar or (batch,)
            # Find surrounding points
            if isinstance(t_query, float) or t_query.dim() == 0:
                t_query = torch.tensor(t_query, device=x.device)
            
            # Clamp to valid range
            t_query = torch.clamp(t_query, 0, 1)
            
            # Find indices
            idx = t_query * (seq_len - 1)  # scale to index space
            idx_low = torch.floor(idx).long()
            idx_high = torch.ceil(idx).long()
            idx_low = torch.clamp(idx_low, 0, seq_len - 1)
            idx_high = torch.clamp(idx_high, 0, seq_len - 1)
            
            # Interpolation weight
            w = idx - idx_low.float()
            w = w.view(-1, 1, 1)  # (batch, 1, 1)
            
            # Get values
            x_low = x[torch.arange(batch_size), idx_low, :]  # (batch, n_features)
            x_high = x[torch.arange(batch_size), idx_high, :]  # (batch, n_features)
            
            return x_low * (1 - w.squeeze()) + x_high * w.squeeze()  # (batch, n_features)
        
        # Initial hidden state
        h = self.initial(x[:, 0, :])  # (batch, hidden_dim)
        
        # Solve CDE: simple Euler method
        dt = 1.0 / self.solver_steps
        
        for step in range(self.solver_steps):
            t_curr = step * dt
            t_next = min((step + 1) * dt, 1.0)
            
            # Get path value at current time
            x_t = interpolate_x(t_curr)  # (batch, n_features)
            
            # Compute derivatives for each layer
            for layer_idx, (vf, ln) in enumerate(zip(self.vector_fields, self.layer_norms)):
                h_ln = ln(h)  # (batch, hidden_dim)
                inp = torch.cat([h_ln, x_t], dim=-1)  # (batch, hidden_dim + n_features)
                dh = vf(inp)  # (batch, hidden_dim)
                
                # Euler step
                h = h + dh * (t_next - t_curr)  # (batch, hidden_dim)
        
        # Read out forecast
        out = self.readout(h)  # (batch, n_features * output_len)
        out = out.view(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

