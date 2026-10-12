import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceDecomposition(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'n_fourier_freq': [4, 16, 64],
            'emission_hidden': [32, 128, 512],
            'transition_lr': [0.01, 0.1, 1.0]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.n_fourier = self.config.get('n_fourier_freq', 16)
        self.emission_hidden = self.config.get('emission_hidden', 128)
        self.transition_lr_scale = self.config.get('transition_lr', 0.1)
        
        # Instance normalization parameters
        self.instance_norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Fourier frequencies for period ~140
        self.register_buffer('fourier_freqs', torch.arange(1, self.n_fourier + 1).float() * 2 * 3.14159 / 140.0)
        
        # Input projection: raw features + Fourier features
        fourier_dim = self.n_fourier * 2  # sin and cos
        self.input_proj = nn.Linear(self.n_features + fourier_dim, self.state_dim)
        
        # Learnable diagonal transition (stable via softplus)
        self.transition_log_diag = nn.Parameter(torch.randn(self.state_dim) * 0.01)
        self.transition_off_diag = nn.Parameter(torch.randn(self.state_dim, self.state_dim) * 0.001)
        
        # Emission network: state -> output features
        self.emission = nn.Sequential(
            nn.Linear(self.state_dim, self.emission_hidden),
            nn.GELU(),
            nn.Linear(self.emission_hidden, self.n_features)
        )
        
        # Direct skip connection
        self.skip = nn.Linear(self.input_len, self.output_len)

    def _compute_fourier_features(self, t):
        # t: (batch, seq_len) or scalar
        if t.dim() == 0:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 1:
            t = t.unsqueeze(1)  # (batch, 1)
        # t: (batch, seq_len) or (batch, 1)
        freqs = self.fourier_freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, n_fourier)
        t_expanded = t.unsqueeze(-1)  # (batch, seq_len, 1) or (batch, 1, 1)
        angles = t_expanded * freqs  # (batch, seq_len, n_fourier)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (batch, seq_len, 2*n_fourier)

    def _transition_matrix(self):
        # Stable transition: near-identity with learned perturbation
        diag = F.softplus(self.transition_log_diag) * self.transition_lr_scale  # (state_dim,)
        A = torch.diag(diag) + self.transition_off_diag * 0.01  # (state_dim, state_dim)
        return A

    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Instance normalization across time
        x_norm = self.instance_norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, input_len, n_features)
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        x_std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        
        # Generate Fourier features for input timesteps
        t_input = torch.arange(self.input_len, device=x.device).float()  # (input_len,)
        fourier_in = self._compute_fourier_features(t_input.unsqueeze(0).expand(batch_size, -1))  # (batch, input_len, 2*n_fourier)
        
        # Concatenate and project to state
        x_augmented = torch.cat([x_norm, fourier_in], dim=-1)  # (batch, input_len, n_features + 2*n_fourier)
        state_inputs = self.input_proj(x_augmented)  # (batch, input_len, state_dim)
        
        # Run state-space model forward
        A = self._transition_matrix()  # (state_dim, state_dim)
        
        # Initialize state
        state = state_inputs[:, 0, :]  # (batch, state_dim)
        
        # Process input sequence
        for t in range(1, self.input_len):
            state = torch.matmul(state, A.T) + state_inputs[:, t, :]  # (batch, state_dim)
        
        # Predict future: roll out state transitions
        predictions = []
        for t in range(self.output_len):
            state = torch.matmul(state, A.T)  # (batch, state_dim)
            pred = self.emission(state)  # (batch, n_features)
            predictions.append(pred)
        
        out = torch.stack(predictions, dim=1)  # (batch, output_len, n_features)
        
        # Add skip connection from raw input
        skip_out = self.skip(x_norm.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, output_len, n_features)
        out = out + skip_out * 0.1
        
        # Denormalize
        out = out * x_std + x_mean  # (batch, output_len, n_features)
        return out

