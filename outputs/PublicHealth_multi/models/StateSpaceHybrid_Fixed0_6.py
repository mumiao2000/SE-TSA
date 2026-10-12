import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceHybrid_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3],
            'rank': [2, 8, 32]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.rank = self.config.get('rank', 8)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input encoder: project to state space
        self.input_encoder = nn.Linear(self.n_features, self.state_dim)
        
        # Helper class to wrap Parameter as Module
        class ParameterModule(nn.Module):
            def __init__(self, param):
                super().__init__()
                self.param = param
            def forward(self):
                return self.param
        
        # State space layers
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(nn.ModuleDict({
                # Base state transition (diagonal for stability)
                'A_log': ParameterModule(nn.Parameter(torch.randn(self.state_dim) * 0.01)),
                'B': nn.Linear(self.n_features, self.state_dim),  # input-to-state
                'C': nn.Linear(self.state_dim, self.n_features),  # state-to-output
                'D': ParameterModule(nn.Parameter(torch.randn(self.n_features) * 0.01)),  # skip connection
                # Input-dependent modulation (low-rank)
                'mod_U': nn.Linear(self.n_features, self.rank),
                'mod_V': nn.Linear(self.rank, self.state_dim * self.state_dim),
                'ln': nn.LayerNorm(self.n_features),
                'dropout': nn.Dropout(self.dropout_rate)
            }))
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.input_len, self.state_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.state_dim, self.output_len)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size, seq_len, _ = x.shape  # seq_len == self.input_len
        
        # Process through state space layers
        h = x  # (batch, input_len, n_features)
        
        for layer in self.layers:
            h_ln = layer['ln'](h)  # (batch, input_len, n_features)
            
            # Compute input-dependent state transition
            # A is diagonal: A = -exp(A_log) for stability
            A_diag = -torch.exp(layer['A_log']())  # (state_dim,)
            
            # Low-rank modulation of A
            mod = layer['mod_U'](h_ln)  # (batch, input_len, rank)
            mod = layer['mod_V'](mod)  # (batch, input_len, state_dim * state_dim)
            mod = mod.view(batch_size, seq_len, self.state_dim, self.state_dim)  # (batch, input_len, state_dim, state_dim)
            
            # State sequence computation
            # Encode input to state dimension
            Bx = layer['B'](h_ln)  # (batch, input_len, state_dim)
            
            # Simple discretization: h_state[t] = A * h_state[t-1] + B * x[t]
            states = []
            state = torch.zeros(batch_size, self.state_dim, device=x.device)  # (batch, state_dim)
            
            for t in range(seq_len):
                # Use diagonal part for stability
                state = A_diag * state + Bx[:, t]  # (batch, state_dim)
                states.append(state)
            
            states = torch.stack(states, dim=1)  # (batch, input_len, state_dim)
            
            # Decode states
            y = layer['C'](states)  # (batch, input_len, n_features)
            y = y + layer['D']() * h_ln  # (batch, input_len, n_features)
            
            h = h + layer['dropout'](y)  # (batch, input_len, n_features)
        
        # Final output: use the processed sequence
        out = self.decoder(h.permute(0, 2, 1))  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

