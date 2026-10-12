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
            'n_components': [2, 4, 8],
            'emission_hidden': [64, 256, 512],
            'transition_hidden': [32, 128, 256]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.n_components = self.config.get('n_components', 4)
        self.emission_hidden = self.config.get('emission_hidden', 256)
        self.transition_hidden = self.config.get('transition_hidden', 128)
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Encoder: observation to latent state
        self.encoder = nn.Sequential(
            nn.Linear(self.n_features, self.emission_hidden),
            nn.ReLU(),
            nn.Linear(self.emission_hidden, self.emission_hidden // 2),
            nn.ReLU(),
            nn.Linear(self.emission_hidden // 2, self.state_dim)
        )
        
        # Multiple transition matrices (mixture of dynamics)
        self.transitions = nn.ParameterList([
            nn.Parameter(torch.eye(self.state_dim) + 0.01 * torch.randn(self.state_dim, self.state_dim))
            for _ in range(self.n_components)
        ])
        
        # Transition mixture weights based on current state
        self.transition_gating = nn.Sequential(
            nn.Linear(self.state_dim, self.transition_hidden),
            nn.ReLU(),
            nn.Linear(self.transition_hidden, self.n_components),
            nn.Softmax(dim=-1)
        )
        
        # Process noise (learnable)
        self.process_noise = nn.Parameter(torch.ones(self.state_dim) * 0.1)
        
        # Decoder: latent state to observation
        self.decoder = nn.Sequential(
            nn.Linear(self.state_dim, self.emission_hidden // 2),
            nn.ReLU(),
            nn.Linear(self.emission_hidden // 2, self.emission_hidden),
            nn.ReLU(),
            nn.Linear(self.emission_hidden, self.n_features)
        )
        
        # Observation noise
        self.obs_noise = nn.Parameter(torch.ones(self.n_features) * 0.1)
        
        # Initial state prior
        self.initial_state = nn.Parameter(torch.zeros(self.state_dim))
        self.initial_cov = nn.Parameter(torch.eye(self.state_dim))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Normalize
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_norm = self.norm(x_perm)  # (batch, n_features, input_len)
        x_norm = x_norm.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Store stats
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        x_std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        
        # Encode sequence to get initial state
        # Use last few observations to determine state
        window_size = min(self.input_len, 5)
        x_window = x_norm[:, -window_size:, :]  # (batch, window_size, n_features)
        
        # Average encoding over window
        states = []
        for t in range(window_size):
            z = self.encoder(x_window[:, t, :])  # (batch, state_dim)
            states.append(z)
        
        # Initialize state as weighted combination
        weights = F.softmax(torch.arange(window_size, dtype=torch.float32, device=x.device), dim=0)
        state = sum(w * s for w, s in zip(weights, states))  # (batch, state_dim)
        
        # Run filtering over input to refine state
        for t in range(self.input_len):
            obs = x_norm[:, t, :]  # (batch, n_features)
            
            # Predict
            gate_weights = self.transition_gating(state)  # (batch, n_components)
            
            # Mixture of transitions
            state_pred = torch.zeros_like(state)
            for i, trans in enumerate(self.transitions):
                state_i = torch.matmul(state, trans.t())  # (batch, state_dim)
                state_pred = state_pred + gate_weights[:, i:i+1] * state_i
            
            # Update with observation (simplified Kalman update)
            predicted_obs = self.decoder(state_pred)  # (batch, n_features)
            innovation = obs - predicted_obs  # (batch, n_features)
            
            # Kalman gain approximation
            state = state_pred + self.encoder(innovation) * 0.1  # (batch, state_dim)
        
        # Forecast
        forecasts = []
        for t in range(self.output_len):
            # Predict next state
            gate_weights = self.transition_gating(state)  # (batch, n_components)
            
            state_pred = torch.zeros_like(state)
            for i, trans in enumerate(self.transitions):
                state_i = torch.matmul(state, trans.t())  # (batch, state_dim)
                state_pred = state_pred + gate_weights[:, i:i+1] * state_i
            
            # Add process noise
            state = state_pred + torch.randn_like(state_pred) * self.process_noise.unsqueeze(0)
            
            # Decode
            pred = self.decoder(state)  # (batch, n_features)
            forecasts.append(pred)
        
        out = torch.stack(forecasts, dim=1)  # (batch, output_len, n_features)
        
        # Denormalize
        out = out * x_std + x_mean  # (batch, output_len, n_features)
        
        return out
