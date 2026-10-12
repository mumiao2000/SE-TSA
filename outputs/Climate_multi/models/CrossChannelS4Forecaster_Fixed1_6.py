import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelS4Forecaster_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'n_groups': [2, 4, 8],
            'rank': [4, 16, 64],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.n_groups = self.config.get('n_groups', 4)
        self.rank = self.config.get('rank', 16)
        self.dropout = self.config.get('dropout', 0.3)
        
        # Ensure groups divide features reasonably
        self.n_groups = min(self.n_groups, self.n_features)
        self.features_per_group = (self.n_features + self.n_groups - 1) // self.n_groups
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Grouped input projections
        self.group_projs = nn.ModuleList([
            nn.Linear(min(self.features_per_group, self.n_features - g * self.features_per_group), self.state_dim)
            for g in range(self.n_groups)
        ])
        
        # S4 state space per group - use individual Parameter/Module attributes instead of ModuleDict
        self.s4_A_real = nn.ParameterList([nn.Parameter(torch.randn(self.state_dim) * -0.5) for _ in range(self.n_groups)])
        self.s4_A_imag = nn.ParameterList([nn.Parameter(torch.randn(self.state_dim) * 0.5) for _ in range(self.n_groups)])
        self.s4_B = nn.ModuleList([nn.Linear(self.state_dim, self.state_dim) for _ in range(self.n_groups)])
        self.s4_C = nn.ModuleList([nn.Linear(self.state_dim, self.state_dim) for _ in range(self.n_groups)])
        self.s4_D = nn.ParameterList([nn.Parameter(torch.randn(self.state_dim) * 0.01) for _ in range(self.n_groups)])
        self.s4_log_dt = nn.ParameterList([nn.Parameter(torch.ones(self.state_dim) * -3) for _ in range(self.n_groups)])
        
        # Low-rank cross-channel mixer - operates on n_features dimension
        self.cross_u = nn.Parameter(torch.randn(self.n_features, self.rank) * 0.02)
        self.cross_v = nn.Parameter(torch.randn(self.rank, self.n_features) * 0.02)
        self.cross_bias = nn.Parameter(torch.zeros(self.n_features))
        
        # Output projections per group
        self.group_out_projs = nn.ModuleList([
            nn.Linear(self.state_dim, min(self.features_per_group, self.n_features - g * self.features_per_group))
            for g in range(self.n_groups)
        ])
        
        # Global refinement
        self.refine = nn.Sequential(
            nn.Linear(self.n_features, 128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.n_features)
        )
        
        # Direct mapping
        self.direct = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Direct residual
        x_res = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_res = self.direct(x_res)  # (batch, n_features, output_len)
        x_res = x_res.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Split into groups
        group_outputs = []
        for g in range(self.n_groups):
            start = g * self.features_per_group
            end = min(start + self.features_per_group, self.n_features)
            if start >= self.n_features:
                break
            
            x_g = x[:, :, start:end]  # (batch, input_len, features_in_group)
            
            # Project to state space
            h = self.group_projs[g](x_g)  # (batch, input_len, state_dim)
            
            # S4 recurrence
            dt = torch.exp(self.s4_log_dt[g])  # (state_dim,)
            
            # Complex diagonal state transition
            A = self.s4_A_real[g] + 1j * self.s4_A_imag[g]  # (state_dim,)
            
            h_states = []
            state = torch.zeros(x.size(0), self.state_dim, dtype=torch.complex64, device=x.device)  # (batch, state_dim)
            for t in range(h.size(1)):
                h_t = h[:, t, :]  # (batch, state_dim)
                # Discretize: state = state + dt * (A * state + B(h_t))
                state = state + dt.unsqueeze(0) * (A.unsqueeze(0) * state + self.s4_B[g](h_t).to(torch.complex64))  # (batch, state_dim)
                out_t = self.s4_C[g](state.real) + self.s4_D[g].unsqueeze(0) * h_t  # (batch, state_dim)
                h_states.append(out_t)
            
            h_g = torch.stack(h_states, dim=1)  # (batch, input_len, state_dim)
            
            # Interpolate to output length
            h_g = h_g.permute(0, 2, 1)  # (batch, state_dim, input_len)
            h_g = F.interpolate(h_g, size=self.output_len, mode='linear', align_corners=False)  # (batch, state_dim, output_len)
            h_g = h_g.permute(0, 2, 1)  # (batch, output_len, state_dim)
            
            # Project back to feature space
            out_g = self.group_out_projs[g](h_g)  # (batch, output_len, features_in_group)
            group_outputs.append(out_g)
        
        # Concatenate groups
        out = torch.cat(group_outputs, dim=-1)  # (batch, output_len, n_features)
        
        # Low-rank cross-channel interaction on n_features dimension
        out_mixed = out @ self.cross_u @ self.cross_v + self.cross_bias  # (batch, output_len, n_features)
        
        # Residual connection for mixing
        out = out + out_mixed * 0.1  # (batch, output_len, n_features)
        
        # Global refinement
        out = self.refine(out)  # (batch, output_len, n_features)
        
        # Add direct residual
        out = out + x_res * 0.1  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

