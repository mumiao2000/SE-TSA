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
            'n_lags': [4, 16, 64],
            'emission_hidden': [32, 128, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.n_lags = self.config.get('n_lags', 16)
        self.emission_hidden = self.config.get('emission_hidden', 128)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: project observations to state
        self.encoder = nn.Sequential(
            nn.Linear(input_len * n_features, self.emission_hidden),
            nn.GELU(),
            nn.Linear(self.emission_hidden, self.state_dim)
        )
        
        # Structured transition: companion matrix for smooth dynamics
        # Learn AR coefficients for state evolution
        self.ar_coeffs = nn.Parameter(torch.randn(self.state_dim) * 0.01)
        self.transition_bias = nn.Parameter(torch.zeros(self.state_dim))
        
        # Learnable damping factor for stability
        self.damping = nn.Parameter(torch.zeros(self.state_dim))
        
        # Seasonal component: Fourier features
        self.seasonal_freqs = nn.Parameter(torch.randn(self.state_dim // 4) * 0.1)
        
        # Decoder: state to predictions
        self.decoder = nn.Sequential(
            nn.Linear(self.state_dim, self.emission_hidden),
            nn.GELU(),
            nn.Linear(self.emission_hidden, output_len * n_features)
        )
        
        # Direct skip connection for recent lags
        self.lag_proj = nn.Linear(min(self.n_lags, input_len) * n_features, output_len * n_features)
        
    def _build_transition(self, batch_size):
        # Create companion-like transition matrix with damping
        # T[i,i] = damping[i], T[i,i-1] = ar_coeffs[i] (with boundary handling)
        
        T = torch.zeros(batch_size, self.state_dim, self.state_dim, 
                       device=self.ar_coeffs.device, dtype=self.ar_coeffs.dtype)
        
        # Diagonal damping
        damp = torch.sigmoid(self.damping) * 0.99 + 0.01  # Ensure |eigenvalue| < 1
        T[:, torch.arange(self.state_dim), torch.arange(self.state_dim)] = damp.unsqueeze(0)
        
        # Sub-diagonal AR structure
        if self.state_dim > 1:
            T[:, 1:, :-1] = torch.diag(torch.tanh(self.ar_coeffs[1:]))
        
        return T
    
    def _add_seasonal(self, state, t):
        # Add seasonal oscillation based on time step
        # state: (batch, state_dim), t: scalar time step
        seasonal = torch.zeros_like(state)
        n_seasonal = min(len(self.seasonal_freqs), self.state_dim // 2)
        for i in range(n_seasonal):
            freq = self.seasonal_freqs[i]
            seasonal[:, 2*i] += torch.sin(freq * t)
            seasonal[:, 2*i+1] += torch.cos(freq * t)
        return seasonal * 0.1  # Small amplitude
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Flatten for encoding
        x_flat = x.view(batch_size, -1)  # (batch, input_len * n_features)
        
        # Initialize state from history
        state = self.encoder(x_flat)  # (batch, state_dim)
        
        # Build transition matrix
        T = self._build_transition(batch_size)  # (batch, state_dim, state_dim)
        
        # Roll out state for prediction horizon
        states = []
        for t in range(self.output_len):
            # Evolve state: s_{t+1} = T @ s_t + bias + seasonal
            state = torch.bmm(T, state.unsqueeze(-1)).squeeze(-1)  # (batch, state_dim)
            state = state + self.transition_bias  # (batch, state_dim)
            state = state + self._add_seasonal(state, t)  # (batch, state_dim)
            states.append(state)
        
        # Stack future states
        future_states = torch.stack(states, dim=1)  # (batch, output_len, state_dim)
        
        # Decode to predictions
        pred = self.decoder(future_states)  # (batch, output_len, output_len * n_features) -> need reshape
        pred = pred.view(batch_size, self.output_len, self.output_len, self.n_features)
        # Take diagonal: each time step predicts itself
        pred = torch.stack([pred[:, i, i, :] for i in range(self.output_len)], dim=1)  # (batch, output_len, n_features)
        
        # Add skip from recent lags
        lag_len = min(self.n_lags, self.input_len)
        x_lag = x[:, -lag_len:, :].contiguous().view(batch_size, -1)  # (batch, lag_len * n_features)
        skip = self.lag_proj(x_lag).view(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        pred = pred + skip * 0.5  # (batch, output_len, n_features)
        
        x = self.revin(pred, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)
