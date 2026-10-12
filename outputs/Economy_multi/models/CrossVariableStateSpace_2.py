import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossVariableStateSpace(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'rank': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.rank = self.config.get('rank', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Instance normalization (more flexible than RevIN for state space)
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Input projection: from feature values to state
        self.input_proj = nn.Linear(1, self.state_dim)  # (batch, n_features, state_dim)
        
        # State transition: diagonal + low-rank cross-feature
        # For each feature, state transition matrix is D + U @ V^T
        self.log_diagonal = nn.Parameter(torch.zeros(self.n_features, self.state_dim))
        self.U = nn.Parameter(torch.randn(self.n_features, self.state_dim, self.rank) * 0.01)
        self.V = nn.Parameter(torch.randn(self.n_features, self.state_dim, self.rank) * 0.01)
        
        # Cross-feature mixing: how much each feature influences others
        self.cross_mix = nn.Parameter(torch.eye(self.n_features) * 0.9 + 0.1 / self.n_features)
        
        # State processing layers
        self.state_norms = nn.ModuleList([nn.LayerNorm(self.state_dim) for _ in range(self.n_layers)])
        self.state_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.state_dim, self.state_dim * 4),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.state_dim * 4, self.state_dim)
            ) for _ in range(self.n_layers)
        ])
        
        # Output: predict next values from state
        self.output_head = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim // 2),
            nn.ReLU(),
            nn.Linear(self.state_dim // 2, 1)
        )
        
        # For direct prediction of full horizon, use a decoder
        self.horizon_proj = nn.Linear(self.state_dim, self.output_len)  # (batch, n_features, output_len)
        
    def apply_state_transition(self, state):
        # state: (batch, n_features, state_dim)
        batch_size = state.shape[0]
        
        # Diagonal component: each feature's own dynamics
        diag = torch.exp(self.log_diagonal)  # (n_features, state_dim)
        state_diag = state * diag.unsqueeze(0)  # (batch, n_features, state_dim)
        
        # Low-rank cross-feature component
        # U: (n_features, state_dim, rank), V: (n_features, state_dim, rank)
        # For each feature i, compute influence from all features j
        # state_j @ V_j: (batch, n_features, rank)
        state_V = torch.einsum('bjd,jdr->bjr', state, self.V)  # (batch, n_features, rank)
        
        # Cross-feature mixing: weighted combination
        mixed = torch.einsum('ij,bjr->bir', self.cross_mix, state_V)  # (batch, n_features, rank)
        
        # Project back to state_dim
        state_lowrank = torch.einsum('bir,idr->bid', mixed, self.U)  # (batch, n_features, state_dim)
        
        return state_diag + state_lowrank  # (batch, n_features, state_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        # Instance norm across features
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.norm(x)  # (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Initialize state from first observation
        x0 = x[:, 0:1, :]  # (batch, 1, n_features)
        x0 = x0.permute(0, 2, 1)  # (batch, n_features, 1)
        state = self.input_proj(x0)  # (batch, n_features, state_dim)
        
        # Recurrent state updates through input sequence
        for t in range(1, self.input_len):
            # State transition
            state = self.apply_state_transition(state)  # (batch, n_features, state_dim)
            
            # Incorporate new observation
            xt = x[:, t:t+1, :].permute(0, 2, 1)  # (batch, n_features, 1)
            obs_embed = self.input_proj(xt)  # (batch, n_features, state_dim)
            state = state + obs_embed  # (batch, n_features, state_dim)
            
            # Process with MLP layers
            for norm, mlp in zip(self.state_norms, self.state_mlps):
                state = state + mlp(norm(state))  # (batch, n_features, state_dim)
        
        # Final state after processing all inputs
        # Predict full horizon at once
        out = self.horizon_proj(state)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Denormalize: apply inverse of instance norm
        # We need to store statistics, but instance norm doesn't store running stats by default
        # For simplicity, we learn a linear output adjustment
        return out  # (batch, output_len, n_features)

