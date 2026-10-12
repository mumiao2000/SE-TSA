import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HierarchicalStateSpace_ChannelAdaptive(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [32, 128, 512],
            'n_levels': [2, 4, 8],
            'transition_noise': [0.0, 0.05, 0.2],
            'gate_hidden_dim': [16, 64, 256],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Configurable hyperparameters
        self.state_dim = self.config.get('state_dim', 128)
        self.n_levels = self.config.get('n_levels', 4)
        self.transition_noise = self.config.get('transition_noise', 0.05)
        self.gate_hidden_dim = self.config.get('gate_hidden_dim', 64)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        # Ensure state_dim is divisible by n_levels for hierarchical splitting
        assert self.state_dim % self.n_levels == 0, "state_dim must be divisible by n_levels"
        self.level_dim = self.state_dim // self.n_levels
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: map n_features to state_dim
        self.input_proj = nn.Linear(self.n_features, self.state_dim)
        
        # Channel-adaptive gating network: learns feature-specific importance
        self.gate_encoder = nn.Sequential(
            nn.Linear(self.n_features, self.gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.gate_hidden_dim, self.n_features),
            nn.Sigmoid()
        )
        
        # Hierarchical state space parameters - one per level
        # Each level operates at a different temporal resolution
        self.level_transitions = nn.ModuleList()
        self.level_emissions = nn.ModuleList()
        self.level_scales = nn.ParameterList()
        
        for level in range(self.n_levels):
            # Transition matrix for this level (learned, with stability constraint)
            transition = nn.Linear(self.level_dim, self.level_dim, bias=False)
            # Initialize near identity for stable dynamics
            with torch.no_grad():
                transition.weight.copy_(torch.eye(self.level_dim) * 0.9 + torch.randn(self.level_dim, self.level_dim) * 0.1)
            self.level_transitions.append(transition)
            
            # Emission: project state back to feature space for this level
            self.level_emissions.append(nn.Linear(self.level_dim, self.n_features))
            
            # Learnable temporal scale for this level (higher levels = slower dynamics)
            # Level 0: fastest, Level n-1: slowest
            scale_init = 1.0 / (2.0 ** level)
            self.level_scales.append(nn.Parameter(torch.tensor(scale_init)))
        
        # Learned mixture weights for combining level predictions
        self.level_mixture = nn.Linear(self.state_dim, self.n_levels)
        
        # Output projection from concatenated level outputs
        self.output_proj = nn.Sequential(
            nn.Linear(self.n_features * self.n_levels, self.state_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.state_dim, self.n_features)
        )
        
        # Direct skip connection for stable gradient flow
        self.skip_proj = nn.Linear(self.input_len, self.output_len, bias=False)
        
        # Final mixing layer
        self.mix = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x_norm = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-adaptive gating: compute per-feature importance
        # Use last timestep features for gating context
        gate_context = x_norm[:, -1, :]  # (batch, n_features)
        channel_gates = self.gate_encoder(gate_context)  # (batch, n_features)
        channel_gates = channel_gates.unsqueeze(1)  # (batch, 1, n_features)
        
        # Apply channel gating
        x_gated = x_norm * channel_gates  # (batch, input_len, n_features)
        
        # Input projection to state space
        h = self.input_proj(x_gated)  # (batch, input_len, state_dim)
        h = self.dropout(h)
        
        # Initialize hierarchical states
        # Split state into levels
        level_states = list(torch.split(h[:, -1, :], self.level_dim, dim=-1))  # list of (batch, level_dim)
        
        # Store predictions from each level
        level_predictions = []
        
        # Predict forward for output_len steps
        for t in range(self.output_len):
            step_preds = []
            
            for level in range(self.n_levels):
                # Apply transition with learned scale
                # Higher levels have slower transitions (smaller effective step)
                scale = torch.abs(self.level_scales[level])
                
                # Multiple micro-steps for slower levels to simulate different resolutions
                n_micro_steps = max(1, int(1.0 / scale.item())) if scale.item() > 0 else 1
                n_micro_steps = min(n_micro_steps, 4)  # Cap for efficiency
                
                state = level_states[level]
                for _ in range(n_micro_steps):
                    # Transition with optional noise during training
                    state = self.level_transitions[level](state)
                    if self.training and self.transition_noise > 0:
                        noise = torch.randn_like(state) * self.transition_noise
                        state = state + noise
                    # Apply nonlinearity for expressive dynamics
                    state = F.tanh(state)
                
                level_states[level] = state
                
                # Emit prediction for this level
                pred = self.level_emissions[level](state)  # (batch, n_features)
                step_preds.append(pred)
            
            # Stack level predictions: (batch, n_levels, n_features)
            step_stack = torch.stack(step_preds, dim=1)  # (batch, n_levels, n_features)
            level_predictions.append(step_stack)
        
        # Stack over time: (batch, output_len, n_levels, n_features)
        predictions = torch.stack(level_predictions, dim=1)  # (batch, output_len, n_levels, n_features)
        
        # Compute adaptive mixture weights from final states
        final_states = torch.cat(level_states, dim=-1)  # (batch, state_dim)
        mixture_logits = self.level_mixture(final_states)  # (batch, n_levels)
        mixture_weights = F.softmax(mixture_logits, dim=-1)  # (batch, n_levels)
        mixture_weights = mixture_weights.unsqueeze(1).unsqueeze(-1)  # (batch, 1, n_levels, 1)
        
        # Weighted combination of level predictions
        weighted_pred = (predictions * mixture_weights).sum(dim=2)  # (batch, output_len, n_features)
        
        # Also use output projection on flattened level predictions for richer combination
        flat_preds = predictions.view(batch_size, self.output_len, -1)  # (batch, output_len, n_levels * n_features)
        proj_pred = self.output_proj(flat_preds)  # (batch, output_len, n_features)
        
        # Mix the two prediction paths
        mix_weights = F.softmax(self.mix, dim=0)
        out = mix_weights[0] * weighted_pred + mix_weights[1] * proj_pred  # (batch, output_len, n_features)
        
        # Skip connection from input (direct linear forecast)
        x_perm = x_norm.permute(0, 2, 1)  # (batch, n_features, input_len)
        skip = self.skip_proj(x_perm)  # (batch, n_features, output_len)
        skip = skip.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Residual connection
        out = out + skip * 0.1  # scaled skip
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

