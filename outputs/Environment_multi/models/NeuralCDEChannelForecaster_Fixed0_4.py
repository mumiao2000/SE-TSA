import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralCDEChannelForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'vector_field_depth': [2, 3, 5],
            'n_solver_steps': [10, 25, 50],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.vf_depth = self.config.get('vector_field_depth', 3)
        self.n_steps = self.config.get('n_solver_steps', 25)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Initial condition network
        self.initial = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Channel-specific vector field: f(h, X_t) for CDE: dh = f(h, X_t) dX_t
        # We use a factorized form: shared processing + channel-specific modulation
        self.vf_shared = nn.ModuleList()
        in_dim = self.hidden_dim + self.n_features
        for i in range(self.vf_depth):
            if i < self.vf_depth - 1:
                # Intermediate layers: output 2*hidden_dim, then chunk
                out_dim = self.hidden_dim * 2
                self.vf_shared.append(nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout)
                ))
                in_dim = self.hidden_dim  # After chunking, take first half
            else:
                # Final layer: output hidden_dim * n_features for the full vector field
                out_dim = self.hidden_dim * self.n_features
                self.vf_shared.append(nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim)
                    # No activation/dropout on final layer
                ))
        
        # Channel modulation: shape (hidden_dim, n_features) for broadcasting
        self.channel_gates = nn.Parameter(torch.ones(self.hidden_dim, self.n_features))
        
        # Decoder: hidden state -> predictions
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
        # Output length handling: predict all at once from final state
        self.output_projector = nn.Linear(self.hidden_dim, self.output_len * self.n_features)
    
    def interpolate(self, x):
        # Simple linear interpolation for continuous path
        # x: (batch, input_len, n_features)
        # Returns coefficients for simple interpolation
        return x  # For simplicity, use observed values directly with Riemann sum
    
    def vector_field(self, h, x_t):
        # h: (batch, hidden_dim), x_t: (batch, n_features)
        # Returns: (batch, hidden_dim, n_features) - derivative w.r.t each channel
        
        # Concatenate
        hx = torch.cat([h, x_t], dim=-1)  # (batch, hidden_dim + n_features)
        
        # Shared processing
        for i, layer in enumerate(self.vf_shared):
            hx = layer(hx)  # (batch, out_dim)
            if i < len(self.vf_shared) - 1:
                # Intermediate layers: chunk and take first half
                hx = hx.chunk(2, dim=-1)[0]  # (batch, hidden_dim)
        
        # Reshape to (batch, hidden_dim, n_features)
        vf_out = hx.reshape(h.shape[0], self.hidden_dim, self.n_features)  # (batch, hidden_dim, n_features)
        
        # Apply channel-specific gating
        # channel_gates: (hidden_dim, n_features), unsqueeze to (1, hidden_dim, n_features)
        vf_out = vf_out * self.channel_gates.unsqueeze(0)  # (batch, hidden_dim, n_features)
        
        return vf_out  # (batch, hidden_dim, n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Initial condition from first observation
        h0 = self.initial(x[:, 0, :])  # (batch, hidden_dim)
        
        # Solve CDE with simple Euler method
        h = h0  # (batch, hidden_dim)
        dt = 1.0 / (self.input_len - 1) if self.input_len > 1 else 1.0
        
        for t in range(self.input_len - 1):
            x_t = x[:, t, :]  # (batch, n_features)
            dx = x[:, t + 1, :] - x_t  # (batch, n_features)
            
            # Vector field
            vf = self.vector_field(h, x_t)  # (batch, hidden_dim, n_features)
            
            # Update: dh = vf @ dx (matrix-vector product over channels)
            dh = torch.bmm(vf, dx.unsqueeze(-1)).squeeze(-1)  # (batch, hidden_dim)
            h = h + dh  # (batch, hidden_dim)
            h = F.gelu(h)  # (batch, hidden_dim)
        
        # Extrapolate to future
        # Use average derivative for prediction
        # Simple approach: project hidden state to all future timesteps
        out = self.output_projector(h)  # (batch, output_len * n_features)
        out = out.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Alternative: evolve with zero control (hold last derivative)
        # For simplicity, use direct projection
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

