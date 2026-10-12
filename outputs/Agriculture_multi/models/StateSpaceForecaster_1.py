import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [32, 128, 512],
            'emission_dim': [64, 256, 1024],
            'rank': [2, 8, 32],
            'n_layers': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 128)
        self.emission_dim = self.config.get('emission_dim', 256)
        self.rank = self.config.get('rank', 8)
        self.n_layers = self.config.get('n_layers', 4)
        
        # RevIN
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input encoder: maps observations to state space
        self.encoder = nn.Sequential(
            nn.Linear(self.n_features, self.emission_dim),
            nn.LayerNorm(self.emission_dim),
            nn.GELU(),
            nn.Linear(self.emission_dim, self.state_dim)
        )
        
        # Structured transition: A = D + uv^T (diagonal plus low-rank)
        self.transition_diag = nn.Parameter(torch.randn(self.state_dim) * 0.9)
        self.transition_u = nn.Parameter(torch.randn(self.state_dim, self.rank) * 0.1)
        self.transition_v = nn.Parameter(torch.randn(self.rank, self.state_dim) * 0.1)
        
        # Learned innovation (Kalman gain-like)
        self.innovation = nn.Sequential(
            nn.Linear(self.state_dim + self.n_features, self.state_dim),
            nn.Sigmoid()
        )
        
        # Emission network
        self.emission = nn.Sequential(
            nn.Linear(self.state_dim, self.emission_dim),
            nn.LayerNorm(self.emission_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.emission_dim, self.n_features)
        )
        
        # Multi-step predictor (propagates uncertainty)
        self.predictor_layers = nn.ModuleList([
            nn.GRUCell(self.state_dim, self.state_dim) for _ in range(self.n_layers)
        ])
        
        # Output refinement
        self.refiner = nn.Sequential(
            nn.Linear(self.n_features * 2, self.n_features),
            nn.LayerNorm(self.n_features),
            nn.GELU(),
            nn.Linear(self.n_features, self.n_features)
        )

    def transition_matrix(self):
        # A = diag(d) + u @ v
        D = torch.diag(torch.tanh(self.transition_diag))  # (state_dim, state_dim)
        UV = self.transition_u @ self.transition_v  # (state_dim, state_dim)
        return D + UV * 0.1  # scaled low-rank update

    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Initialize state
        state = self.encoder(x[:, 0, :])  # (batch, state_dim)
        
        # Kalman filtering over input sequence
        A = self.transition_matrix()  # (state_dim, state_dim)
        
        for t in range(1, self.input_len):
            # Predict
            state_pred = state @ A.T  # (batch, state_dim)
            
            # Observe
            obs = x[:, t, :]  # (batch, n_features)
            obs_encoded = self.encoder(obs)  # (batch, state_dim)
            
            # Update with learned innovation
            innovation_input = torch.cat([state_pred, obs], dim=-1)  # (batch, state_dim + n_features)
            gain = self.innovation(innovation_input)  # (batch, state_dim)
            
            state = state_pred + gain * (obs_encoded - state_pred)  # (batch, state_dim)
        
        # Multi-step prediction
        outputs = []
        current_state = state
        
        for _ in range(self.output_len):
            # Propagate through transition
            next_state = current_state @ A.T  # (batch, state_dim)
            
            # Refine with GRU layers
            gru_state = next_state
            for gru in self.predictor_layers:
                gru_state = gru(next_state, gru_state)  # (batch, state_dim)
            
            # Emit observation
            pred = self.emission(gru_state)  # (batch, n_features)
            outputs.append(pred)
            
            current_state = gru_state
        
        out = torch.stack(outputs, dim=1)  # (batch, output_len, n_features)
        
        # Refine with input statistics
        x_last = x[:, -1, :]  # (batch, n_features)
        x_mean = x.mean(dim=1)  # (batch, n_features)
        
        for t in range(self.output_len):
            ref_input = torch.cat([out[:, t, :], x_last], dim=-1)  # (batch, n_features * 2)
            out[:, t, :] = out[:, t, :] + self.refiner(ref_input)  # residual refinement
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

