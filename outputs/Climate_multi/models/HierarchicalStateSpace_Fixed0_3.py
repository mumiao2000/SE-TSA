import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HierarchicalStateSpace_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'n_levels': [2, 3, 4],
            'transition_noise': [0.01, 0.1, 0.5],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.state_dim = self.config.get('state_dim', 64)
        self.n_levels = self.config.get('n_levels', 3)
        self.transition_noise = self.config.get('transition_noise', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Level-specific parameters (coarser time resolution at higher levels)
        self.level_factors = [2 ** i for i in range(self.n_levels)]
        
        # Observation model: project features to state
        self.obs_proj = nn.Linear(self.n_features, self.state_dim)
        
        # Hierarchical transition matrices (learnable)
        self.transitions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.state_dim, self.state_dim),
                nn.Tanh(),
                nn.Linear(self.state_dim, self.state_dim)
            ) for _ in range(self.n_levels)
        ])
        
        # Bottom-up and top-down connections
        self.bottom_up = nn.ModuleList([
            nn.Linear(self.state_dim, self.state_dim) if i > 0 else None
            for i in range(self.n_levels)
        ])
        self.top_down = nn.ModuleList([
            nn.Linear(self.state_dim, self.state_dim) if i < self.n_levels - 1 else None
            for i in range(self.n_levels)
        ])
        
        # Output generation from all levels
        self.level_weights = nn.Parameter(torch.ones(self.n_levels) / self.n_levels)
        self.output_proj = nn.ModuleList([
            nn.Linear(self.state_dim, self.n_features) for _ in range(self.n_levels)
        ])
        
        # Direct prediction MLP
        self.direct_pred = nn.Sequential(
            nn.Linear(self.input_len * self.n_features, self.state_dim * 2),
            nn.ReLU(),
            nn.Linear(self.state_dim * 2, self.output_len * self.n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Project observations to state space
        z = self.obs_proj(x)  # (batch, input_len, state_dim)
        
        # Hierarchical processing
        level_states = []
        
        for level in range(self.n_levels):
            factor = self.level_factors[level]
            
            # Pool to coarser resolution if needed
            if factor > 1:
                # Average pool in time
                seq_len = z.shape[1]
                pool_size = min(factor, seq_len)
                z_pooled = F.avg_pool1d(z.permute(0, 2, 1), kernel_size=pool_size, stride=pool_size)
                z_pooled = z_pooled.permute(0, 2, 1)  # (batch, pooled_len, state_dim)
            else:
                z_pooled = z  # (batch, input_len, state_dim)
            
            # Apply transition
            z_trans = self.transitions[level](z_pooled)  # (batch, pooled_len, state_dim)
            
            # Add bottom-up influence
            if level > 0 and self.bottom_up[level] is not None:
                # Upsample previous level's state
                prev_state = level_states[-1]  # (batch, prev_pooled_len, state_dim)
                # Simple repeat upsampling
                upsampled = prev_state.repeat_interleave(2, dim=1)  # (batch, prev_pooled_len*2, state_dim)
                # Match length
                min_len = min(upsampled.shape[1], z_trans.shape[1])
                z_trans[:, :min_len, :] = z_trans[:, :min_len, :] + self.bottom_up[level](upsampled[:, :min_len, :])
            
            level_states.append(z_trans)
        
        # Top-down refinement
        for level in range(self.n_levels - 2, -1, -1):
            if self.top_down[level] is not None:
                higher = level_states[level + 1]  # (batch, higher_pooled_len, state_dim)
                # Downsample by taking or repeating
                curr_len = level_states[level].shape[1]
                higher_len = higher.shape[1]
                if higher_len < curr_len:
                    higher_expanded = higher.repeat_interleave(curr_len // max(higher_len, 1), dim=1)
                    higher_expanded = higher_expanded[:, :curr_len, :]
                else:
                    higher_expanded = higher[:, :curr_len, :]
                level_states[level] = level_states[level] + self.top_down[level](higher_expanded)
        
        # Generate predictions from each level
        level_preds = []
        for level in range(self.n_levels):
            # Use last state to predict future
            last_state = level_states[level][:, -1:, :]  # (batch, 1, state_dim)
            # Simple: repeat for output_len with small transition
            pred_states = [last_state]
            for _ in range(self.output_len - 1):
                next_state = self.transitions[level](pred_states[-1])
                pred_states.append(next_state)
            pred_states = torch.cat(pred_states, dim=1)  # (batch, output_len, state_dim)
            pred = self.output_proj[level](pred_states)  # (batch, output_len, n_features)
            level_preds.append(pred)
        
        # Weighted combination
        weights = F.softmax(self.level_weights, dim=0)  # (n_levels,)
        out = sum(w * p for w, p in zip(weights, level_preds))  # (batch, output_len, n_features)
        
        # Add direct prediction residual
        direct = self.direct_pred(x.reshape(batch_size, -1))  # (batch, output_len*n_features)
        direct = direct.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        out = out + 0.1 * direct  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

