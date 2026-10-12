import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HierarchicalStateSpaceNet_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'n_levels': [2, 3, 4],
            'hidden_dim': [64, 256, 512],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.n_levels = self.config.get('n_levels', 3)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection per level: project features to state_dim at each timestep
        self.input_projs = nn.ModuleList()
        for level in range(self.n_levels):
            self.input_projs.append(nn.Linear(self.n_features, self.state_dim))
        
        # State transition matrices (learned, initialized near identity for stability)
        self.transitions = nn.ParameterList([
            nn.Parameter(torch.eye(self.state_dim) * 0.9 + torch.randn(self.state_dim, self.state_dim) * 0.01)
            for _ in range(self.n_levels)
        ])
        
        # Input matrices for state update
        self.input_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(self.state_dim, self.state_dim) * 0.01)
            for _ in range(self.n_levels)
        ])
        
        # Observation matrices (project state to hidden_dim)
        self.observations = nn.ModuleList([
            nn.Linear(self.state_dim, self.hidden_dim)
            for _ in range(self.n_levels)
        ])
        
        # Level-specific decoders: hidden_dim -> output_len
        self.level_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, self.output_len)
            ) for _ in range(self.n_levels)
        ])
        
        # Output projection to combine levels: n_levels -> 1
        self.output_proj = nn.Linear(self.n_levels, 1)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        level_outputs = []
        
        for level in range(self.n_levels):
            # Subsample input for higher levels
            if level == 0:
                x_level = x  # (batch, input_len, n_features)
            else:
                stride = 2 ** level
                effective_len = max(self.input_len // stride, 1)
                indices = torch.linspace(0, self.input_len - 1, effective_len).long()
                x_level = x[:, indices, :]  # (batch, effective_len, n_features)
            
            # Project each timestep to state space: (batch, effective_len, n_features) -> (batch, effective_len, state_dim)
            state = self.input_projs[level](x_level)  # (batch, effective_len, state_dim)
            
            # Get stable transition matrix
            transition = self.transitions[level]  # (state_dim, state_dim)
            # Make transition stable: PSD with spectral radius < 1
            transition_stable = torch.matmul(transition, transition.t())  # (state_dim, state_dim)
            transition_stable = transition_stable / (torch.norm(transition_stable, dim=-1, keepdim=True) + 1e-8)
            transition_stable = transition_stable * 0.99  # Ensure spectral radius < 1
            
            # Evolve state through time using recurrence: s_t = A * s_{t-1} + B * input_t
            # For simplicity, apply transition to all states and add input effect
            state_evolved = torch.matmul(state, transition_stable.t())  # (batch, effective_len, state_dim)
            
            # Add input-driven component
            input_effect = torch.matmul(state, self.input_matrices[level].t())  # (batch, effective_len, state_dim)
            state_final = state_evolved + input_effect  # (batch, effective_len, state_dim)
            
            # Use last state for observation (or aggregate)
            state_last = state_final[:, -1, :]  # (batch, state_dim)
            
            # Observe: project to hidden_dim
            observed = self.observations[level](state_last)  # (batch, hidden_dim)
            
            # Decode to output length for each feature
            # We need (batch, n_features, output_len), so expand observed and decode per feature
            # Actually, let's decode once and expand: (batch, hidden_dim) -> (batch, output_len)
            level_out = self.level_decoders[level](observed)  # (batch, output_len)
            
            # Expand to all features: (batch, 1, output_len) -> (batch, n_features, output_len)
            level_out = level_out.unsqueeze(1).expand(-1, self.n_features, -1)  # (batch, n_features, output_len)
            level_outputs.append(level_out)
        
        # Stack level outputs: (batch, n_features, output_len, n_levels)
        stacked = torch.stack(level_outputs, dim=-1)  # (batch, n_features, output_len, n_levels)
        
        # Combine levels using learned weights
        out = self.output_proj(stacked).squeeze(-1)  # (batch, n_features, output_len)
        
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

