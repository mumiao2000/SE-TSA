import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceGating_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.1, 0.3],
            'gating_hidden': [32, 128, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.gating_hidden = self.config.get('gating_hidden', 128)
        
        # Advanced preprocessing: learnable normalization with running stats
        self.norm_weight = nn.Parameter(torch.ones(self.n_features))
        self.norm_bias = nn.Parameter(torch.zeros(self.n_features))
        self.norm_running_mean = nn.Parameter(torch.zeros(self.n_features), requires_grad=False)
        self.norm_running_var = nn.Parameter(torch.ones(self.n_features), requires_grad=False)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.state_dim)
        
        # State-space layers with gating
        self.ss_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.ss_layers.append(nn.ModuleDict({
                # Gating network for input-dependent transition
                'gate': nn.Sequential(
                    nn.Linear(self.n_features, self.gating_hidden),
                    nn.SiLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.gating_hidden, self.state_dim * self.state_dim)
                ),
                # State transition (base) - wrapped in a module to be valid in ModuleDict
                'A_base': nn.Linear(self.state_dim, self.state_dim, bias=False),
                # Input matrix
                'B': nn.Linear(self.n_features, self.state_dim),
                # Output matrix
                'C': nn.Linear(self.state_dim, self.state_dim),
                # Layer norm
                'norm': nn.LayerNorm(self.state_dim),
                'dropout': nn.Dropout(self.dropout)
            }))
            # Initialize A_base to small random values (approximating original intent)
            nn.init.normal_(self.ss_layers[-1]['A_base'].weight, mean=0.0, std=0.01)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim // 2),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.state_dim // 2, self.n_features)
        )
        
        # Temporal aggregation for output
        self.temporal_agg = nn.Linear(self.input_len, self.output_len)
        
    def apply_norm(self, x):
        # x: (batch, seq_len, n_features)
        # Use running stats for normalization
        mean = self.norm_running_mean  # (n_features,)
        std = torch.sqrt(self.norm_running_var + 1e-5)  # (n_features,)
        x = (x - mean) / std  # (batch, seq_len, n_features)
        x = x * self.norm_weight + self.norm_bias  # (batch, seq_len, n_features)
        return x
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        # Update running stats (simplified)
        with torch.no_grad():
            batch_mean = x.mean(dim=[0, 1])  # (n_features,)
            batch_var = x.var(dim=[0, 1], unbiased=False)  # (n_features,)
            self.norm_running_mean.copy_(0.9 * self.norm_running_mean + 0.1 * batch_mean)
            self.norm_running_var.copy_(0.9 * self.norm_running_var + 0.1 * batch_var)
        
        x = self.apply_norm(x)  # (batch, input_len, n_features)
        
        B, T, F = x.shape  # (batch, input_len, n_features)
        
        # Project input to state space
        u = self.input_proj(x)  # (batch, input_len, state_dim)
        
        # Initialize state
        state = torch.zeros(B, self.state_dim, device=x.device)  # (batch, state_dim)
        
        # Process input sequence
        states = []
        for t in range(T):
            u_t = u[:, t, :]  # (batch, state_dim)
            x_t = x[:, t, :]  # (batch, n_features)
            
            for layer in self.ss_layers:
                # Compute gated transition matrix
                gate = layer['gate'](x_t)  # (batch, state_dim * state_dim)
                gate = gate.view(B, self.state_dim, self.state_dim)  # (batch, state_dim, state_dim)
                A_t = layer['A_base'].weight.unsqueeze(0) + gate * 0.1  # (batch, state_dim, state_dim)
                
                # State update: h_{t+1} = A_t @ h_t + B @ u_t
                B_u = layer['B'](x_t)  # (batch, state_dim)
                
                # Apply transition
                state_new = torch.bmm(A_t, state.unsqueeze(-1)).squeeze(-1) + B_u  # (batch, state_dim)
                state = state_new
                
                # Output projection and residual
                out_t = layer['C'](state)  # (batch, state_dim)
                out_t = layer['norm'](out_t + u_t)  # (batch, state_dim)
                out_t = layer['dropout'](out_t)
                
                # Update for next layer
                u_t = out_t
            
            states.append(state)
            u = u.clone()
            u[:, t, :] = out_t
        
        states = torch.stack(states, dim=1)  # (batch, input_len, state_dim)
        
        # Predict future: continue state evolution
        predictions = []
        current_state = states[:, -1, :]  # (batch, state_dim)
        
        for step in range(self.output_len):
            # Use last input features as proxy (or could use predicted features)
            x_proxy = x[:, -1, :]  # (batch, n_features)
            
            u_step = self.input_proj(x_proxy.unsqueeze(1)).squeeze(1)  # (batch, state_dim)
            
            for layer in self.ss_layers:
                gate = layer['gate'](x_proxy)  # (batch, state_dim * state_dim)
                gate = gate.view(B, self.state_dim, self.state_dim)
                A_t = layer['A_base'].weight.unsqueeze(0) + gate * 0.1
                
                B_u = layer['B'](x_proxy)
                current_state = torch.bmm(A_t, current_state.unsqueeze(-1)).squeeze(-1) + B_u
                
                out_step = layer['C'](current_state)
                out_step = layer['norm'](out_step + u_step)
                out_step = layer['dropout'](out_step)
                u_step = out_step
            
            pred = self.output_proj(out_step)  # (batch, n_features)
            predictions.append(pred)
            
            # Update proxy for next step (simple feedback)
            x_proxy = pred
        
        out = torch.stack(predictions, dim=1)  # (batch, output_len, n_features)
        
        # Denormalize
        out = out - self.norm_bias  # (batch, output_len, n_features)
        out = out / self.norm_weight  # (batch, output_len, n_features)
        std = torch.sqrt(self.norm_running_var + 1e-5)  # (n_features,)
        out = out * std + self.norm_running_mean  # (batch, output_len, n_features)
        
        return out

