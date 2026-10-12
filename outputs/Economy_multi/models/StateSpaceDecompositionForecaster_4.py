import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceDecompositionForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'emission_hidden': [64, 256, 512],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.emission_hidden = self.config.get('emission_hidden', 256)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: project observations to state
        self.encoder = nn.Sequential(
            nn.Linear(input_len * n_features, self.emission_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.emission_hidden, self.state_dim),
        )
        
        # Learnable but constrained transition
        # Use a parametrization that ensures stability: A = U * diag(clipped_eigenvalues) * U^T
        self.transition_raw = nn.Parameter(torch.randn(self.state_dim, self.state_dim) * 0.01)
        self.transition_diag = nn.Parameter(torch.ones(self.state_dim) * 0.9)
        
        # Emission network: state -> observations
        emission_layers = []
        for i in range(self.n_layers):
            in_dim = self.state_dim if i == 0 else self.emission_hidden
            out_dim = self.emission_hidden if i < self.n_layers - 1 else output_len * n_features
            emission_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                emission_layers.append(nn.GELU())
                emission_layers.append(nn.Dropout(self.dropout_rate))
        self.emission = nn.Sequential(*emission_layers)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def get_stable_transition(self):
        # Create stable transition matrix with eigenvalues in unit circle
        # A = 0.5 * (A_raw + A_raw^T) for symmetry, then scale
        A_sym = 0.5 * (self.transition_raw + self.transition_raw.t())
        # Eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(A_sym)
        # Clip eigenvalues to ensure stability (magnitude < 1)
        clipped_eigenvalues = torch.tanh(eigenvalues) * 0.99
        # Reconstruct with learned diagonal modulation
        diag_scale = torch.sigmoid(self.transition_diag)
        clipped_eigenvalues = clipped_eigenvalues * diag_scale
        A = eigenvectors @ torch.diag(clipped_eigenvalues) @ eigenvectors.t()
        return A  # (state_dim, state_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Flatten input for encoding
        x_flat = x.reshape(batch_size, -1)  # (batch, input_len * n_features)
        state = self.encoder(x_flat)  # (batch, state_dim)
        
        # Get stable transition
        A = self.get_stable_transition()  # (state_dim, state_dim)
        
        # Predict future states by applying transition
        states = [state]
        for _ in range(self.output_len - 1):
            state = torch.matmul(state, A.t())  # (batch, state_dim)
            states.append(state)
        
        # Stack states: (batch, output_len, state_dim)
        states = torch.stack(states, dim=1)  # (batch, output_len, state_dim)
        
        # Emit observations
        out = self.emission(states)  # (batch, output_len, output_len * n_features) - wait, wrong
        
        # Fix: emission should output n_features per time step
        # Actually emission outputs output_len * n_features total, need to reshape
        # But we want per-timestep output
        
        # Recompute: apply emission per state
        outputs = []
        for t in range(self.output_len):
            s = states[:, t, :]  # (batch, state_dim)
            # Use a shared emission that outputs n_features
            o = self.emission(s)  # (batch, output_len * n_features) - this is wrong
            outputs.append(o)
        
        # The issue is emission was designed for total output. Let me fix.
        # Actually, let's make emission output n_features directly
        
        # Redefine emission in forward or use a different head
        # For simplicity, reshape the output
        out = torch.stack(outputs, dim=1)  # (batch, output_len, output_len * n_features)
        out = out.reshape(batch_size, self.output_len, self.output_len, self.n_features)
        # Take diagonal in time dimension
        out = torch.stack([out[:, t, t, :] for t in range(self.output_len)], dim=1)
        
        # This is hacky. Let me use a proper per-timestep emission.
        
        # Correct approach: redefine emission to output n_features
        self.per_step_emission = nn.Sequential(
            nn.Linear(self.state_dim, self.emission_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.emission_hidden, self.n_features),
        ).to(x.device)
        
        outputs = []
        for t in range(self.output_len):
            s = states[:, t, :]  # (batch, state_dim)
            o = self.per_step_emission(s)  # (batch, n_features)
            outputs.append(o)
        
        out = torch.stack(outputs, dim=1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

