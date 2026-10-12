import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'n_scales': [1, 3, 5],
            'emission_hidden': [64, 256, 1024],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.state_dim = self.config.get('state_dim', 64)
        self.n_scales = self.config.get('n_scales', 3)
        self.emission_hidden = self.config.get('emission_hidden', 256)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Input encoder: project observations to state
        self.encoder = nn.Sequential(
            nn.Linear(self.n_features, self.state_dim * self.n_scales),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Multi-scale state transition matrices (discrete time)
        self.transitions = nn.ParameterList([
            nn.Parameter(torch.randn(self.state_dim, self.state_dim) * 0.01)
            for _ in range(self.n_scales)
        ])
        
        # Scale mixing weights
        self.scale_mix = nn.Parameter(torch.ones(self.n_scales) / self.n_scales)
        
        # State evolution over input sequence
        # We use a simple RNN-like accumulation but with structured linear transitions
        
        # Emission: state to output features
        self.emission = nn.Sequential(
            nn.Linear(self.state_dim * self.n_scales, self.emission_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.emission_hidden, self.n_features)
        )
        
        # Direct input-to-output skip for short-term patterns
        self.skip = nn.Linear(self.input_len, self.output_len)
        
        # Layer norms
        self.state_norm = nn.LayerNorm(self.state_dim * self.n_scales)
        self.input_norm = nn.LayerNorm(self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.input_norm(x)  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Encode each timestep to state
        states = self.encoder(x)  # (batch, input_len, state_dim * n_scales)
        states = states.reshape(batch_size, self.input_len, self.n_scales, self.state_dim)
        # (batch, input_len, n_scales, state_dim)
        
        # Initialize final state as weighted combination of last few states
        # with transition evolution
        final_states = []
        for s in range(self.n_scales):
            scale_states = states[:, :, s, :]  # (batch, input_len, state_dim)
            
            # Evolve state: h_t = A_s * h_{t-1} + x_t
            # We approximate by applying transition to mean of recent states
            recent = scale_states[:, -5:, :].mean(dim=1)  # (batch, state_dim)
            
            # Apply transition multiple times for stability
            evolved = recent
            for _ in range(min(self.output_len, 10)):
                evolved = evolved @ self.transitions[s].T  # (batch, state_dim)
                evolved = torch.tanh(evolved)  # stability
            
            final_states.append(evolved)
        
        # Stack and weight scales
        final_state = torch.stack(final_states, dim=1)  # (batch, n_scales, state_dim)
        weights = F.softmax(self.scale_mix, dim=0)  # (n_scales,)
        weights = weights.view(1, self.n_scales, 1)  # (1, n_scales, 1)
        
        mixed_state = (final_state * weights).sum(dim=1)  # (batch, state_dim)
        
        # Expand to output length by continuing evolution
        output_states = []
        current = mixed_state  # (batch, state_dim)
        for t in range(self.output_len):
            # Evolve with blended transition
            blended_A = sum(w * self.transitions[i] for i, w in enumerate(F.softmax(self.scale_mix, dim=0)))
            current = current @ blended_A.T  # (batch, state_dim)
            current = torch.tanh(current)
            output_states.append(current)
        
        output_states = torch.stack(output_states, dim=1)  # (batch, output_len, state_dim)
        
        # For multi-scale, repeat with each scale and combine
        all_scale_outputs = []
        for s in range(self.n_scales):
            scale_out = output_states  # (batch, output_len, state_dim)
            # Pad to full state dimension
            padded = F.pad(scale_out, (s * self.state_dim, (self.n_scales - s - 1) * self.state_dim))
            # Actually, let's just use the mixed state properly
            pass
        
        # Simpler: use the mixed state expanded
        full_state = output_states.repeat(1, 1, self.n_scales)  # (batch, output_len, state_dim * n_scales)
        full_state = self.state_norm(full_state)  # (batch, output_len, state_dim * n_scales)
        
        # Emit output
        out = self.emission(full_state)  # (batch, output_len, n_features)
        
        # Skip connection from input
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        skip_out = self.skip(x_perm)  # (batch, n_features, output_len)
        skip_out = skip_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = out + skip_out  # (batch, output_len, n_features)
        
        return out
