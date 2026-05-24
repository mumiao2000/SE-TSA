import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralCDEAttention(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_cde_layers': [1, 2, 4],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_cde_layers = self.config.get('n_cde_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        
        assert self.hidden_dim % self.n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Initial lift: (batch, input_len, n_features) -> (batch, input_len, hidden_dim)
        self.initial_lift = nn.Linear(self.n_features, self.hidden_dim)
        
        # CDE vector field: attention-based
        self.cde_layers = nn.ModuleList()
        for _ in range(self.n_cde_layers):
            self.cde_layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(self.hidden_dim),
                'attn': nn.MultiheadAttention(self.hidden_dim, self.n_heads, dropout=self.dropout, batch_first=True),
                'norm2': nn.LayerNorm(self.hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                    nn.Dropout(self.dropout)
                ),
                'vector_field': nn.Sequential(
                    nn.Linear(self.hidden_dim + self.n_features, self.hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim)
                )
            }))
        
        # Readout
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.n_features)
        )
        
    def natural_cubic_spline_coeffs(self, x, t):
        # Simple linear interpolation for coefficients (simplified CDE)
        # x: (batch, seq_len, features), t: (seq_len,)
        # Returns coefficients for interpolation
        return x, t  # Simplified: just return for linear interpolation
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x_orig = x.clone()
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        B, T, F = x.shape  # (batch, input_len, n_features)
        
        # Create time points
        t = torch.linspace(0, 1, T, device=x.device)  # (input_len,)
        t_out = torch.linspace(1, 1 + self.output_len / T, self.output_len, device=x.device)  # (output_len,)
        
        # Initial lift
        h = self.initial_lift(x)  # (batch, input_len, hidden_dim)
        
        # Simplified CDE: use attention to model interactions, then evolve
        # Treat as discrete steps with attention-based vector field
        
        # Encode input sequence
        for layer in self.cde_layers:
            # Self-attention
            h_norm = layer['norm1'](h)  # (batch, input_len, hidden_dim)
            attn_out, _ = layer['attn'](h_norm, h_norm, h_norm)  # (batch, input_len, hidden_dim)
            h = h + attn_out  # (batch, input_len, hidden_dim)
            
            # FFN
            h = h + layer['ffn'](layer['norm2'](h))  # (batch, input_len, hidden_dim)
        
        # Use final hidden state as initial condition for evolution
        h_final = h[:, -1, :]  # (batch, hidden_dim)
        
        # Predict future: attention-based evolution with exogenous conditioning
        # Create query positions for future
        future_queries = torch.arange(T, T + self.output_len, device=x.device).float() / T  # (output_len,)
        
        # Interpolate/extrapolate using learned dynamics
        # Simple approach: use last known state and apply learned transformation
        
        # Generate future predictions
        out = []
        current_h = h_final  # (batch, hidden_dim)
        
        for step in range(self.output_len):
            # Condition on last known features (use last input or previous prediction)
            if step == 0:
                cond = x[:, -1, :]  # (batch, n_features)
            else:
                # Use previous prediction (closed loop)
                cond = out[-1]  # (batch, n_features)
            
            # Combine hidden state with conditioning
            combined = torch.cat([current_h, cond], dim=-1)  # (batch, hidden_dim + n_features)
            
            # Apply vector field (simplified Euler step)
            delta_h = self.cde_layers[-1]['vector_field'](combined)  # (batch, hidden_dim)
            current_h = current_h + delta_h * 0.1  # Small step size
            
            # Read out prediction
            pred = self.readout(current_h)  # (batch, n_features)
            out.append(pred)
        
        out = torch.stack(out, dim=1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

