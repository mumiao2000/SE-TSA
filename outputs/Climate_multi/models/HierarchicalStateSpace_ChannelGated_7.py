import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HierarchicalStateSpace_ChannelGated(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [32, 128, 512],
            'n_levels': [2, 4, 8],
            'transition_noise': [0.0, 0.05, 0.2],
            'gate_dim': [16, 32, 64],
            'dropout': [0.0, 0.15, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 128)
        self.n_levels = self.config.get('n_levels', 4)
        self.transition_noise = self.config.get('transition_noise', 0.05)
        self.gate_dim = self.config.get('gate_dim', 32)
        self.dropout_rate = self.config.get('dropout', 0.15)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Per-level state dimensions (deeper levels get smaller states)
        self.level_dims = []
        dim = self.state_dim
        for _ in range(self.n_levels):
            self.level_dims.append(max(dim, 8))
            dim = dim // 2
        
        # Input projection to multi-level states
        self.input_proj = nn.Linear(self.n_features, sum(self.level_dims))
        
        # Channel-adaptive gating for each level
        self.level_gates = nn.ModuleList()
        for level_d in self.level_dims:
            # Gate produces per-feature weights for state transitions
            gate = nn.Sequential(
                nn.Linear(self.n_features, self.gate_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.gate_dim, self.n_features),
                nn.Sigmoid(),
            )
            self.level_gates.append(gate)
        
        # State transition matrices (feature-shared base + feature-specific modulation)
        self.transitions_A = nn.ModuleList()
        self.transitions_B = nn.ModuleList()
        for level_d in self.level_dims:
            # Base transition
            self.transitions_A.append(nn.Linear(level_d, level_d, bias=False))
            # Input modulation
            self.transitions_B.append(nn.Linear(level_d, level_d, bias=False))
        
        # Feature-specific state modulation (low-rank)
        self.feature_modulators = nn.ModuleList()
        for level_d in self.level_dims:
            mod = nn.Sequential(
                nn.Linear(self.n_features, self.gate_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.gate_dim, level_d * level_d),
            )
            self.feature_modulators.append(mod)
        
        # Inter-level connections (coarser levels influence finer levels)
        self.level_up_projs = nn.ModuleList()
        for i in range(self.n_levels - 1):
            self.level_up_projs.append(
                nn.Linear(self.level_dims[i+1], self.level_dims[i])
            )
        
        # Output: project states back to features, predict future
        total_state = sum(self.level_dims)
        self.output_proj = nn.Sequential(
            nn.Linear(total_state, self.state_dim * 2),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.state_dim * 2, self.output_len * self.n_features),
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode="norm")  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Initialize multi-level states from last input timestep
        x_last = x[:, -1, :]  # (batch, n_features)
        state_vec = self.input_proj(x_last)  # (batch, sum(level_dims))
        
        # Split into per-level states
        states = []
        start = 0
        for level_d in self.level_dims:
            states.append(state_vec[:, start:start+level_d])  # (batch, level_d)
            start += level_d
        
        # Recurrent state evolution with channel-adaptive gating
        for t in range(self.input_len):
            x_t = x[:, t, :]  # (batch, n_features)
            new_states = []
            
            for i, (state, level_d) in enumerate(zip(states, self.level_dims)):
                # Compute channel-adaptive gate for this level
                gate = self.level_gates[i](x_t)  # (batch, n_features)
                gate_mean = gate.mean(dim=-1, keepdim=True)  # (batch, 1)
                
                # Base state transition
                base_transition = self.transitions_A[i](state)  # (batch, level_d)
                
                # Feature-specific modulation (low-rank, conditioned on mean gate)
                mod_weights = self.feature_modulators[i](x_t)  # (batch, level_d * level_d)
                mod_weights = mod_weights.view(batch_size, level_d, level_d)  # (batch, level_d, level_d)
                
                # Apply modulation scaled by gate
                modulated = torch.bmm(state.unsqueeze(1), mod_weights).squeeze(1)  # (batch, level_d)
                modulated = modulated * gate_mean  # (batch, level_d)
                
                # Combine base and modulated
                new_state = base_transition + modulated  # (batch, level_d)
                
                # Add inter-level influence from coarser level
                if i < self.n_levels - 1:
                    coarse_influence = self.level_up_projs[i](states[i+1])  # (batch, level_d)
                    new_state = new_state + 0.1 * coarse_influence  # (batch, level_d)
                
                # Add transition noise for regularization during training
                if self.training and self.transition_noise > 0:
                    noise = torch.randn_like(new_state) * self.transition_noise
                    new_state = new_state + noise  # (batch, level_d)
                
                new_states.append(self.dropout(torch.tanh(new_state)))  # (batch, level_d)
            
            states = new_states
        
        # Concatenate final states from all levels
        final_states = torch.cat(states, dim=-1)  # (batch, sum(level_dims))
        
        # Predict output
        out = self.output_proj(final_states)  # (batch, output_len * n_features)
        out = out.view(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode="denorm")  # (batch, output_len, n_features)
        return out

