import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'hidden_dim': [64, 256, 512],
            'n_layers': [2, 4, 6],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 64)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: compress input to initial state
        encoder_layers = []
        in_dim = self.n_features * self.input_len
        for i in range(self.n_layers):
            out_dim = self.hidden_dim if i < self.n_layers - 1 else self.state_dim * 2
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # State transition MLP
        trans_layers = []
        in_dim = self.state_dim
        for i in range(2):
            out_dim = self.hidden_dim if i == 0 else self.state_dim
            trans_layers.append(nn.Linear(in_dim, out_dim))
            if i == 0:
                trans_layers.append(nn.ReLU())
                trans_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.transition = nn.Sequential(*trans_layers)
        
        # Observation emission MLP
        emit_layers = []
        in_dim = self.state_dim
        for i in range(2):
            out_dim = self.hidden_dim if i == 0 else self.n_features
            emit_layers.append(nn.Linear(in_dim, out_dim))
            if i == 0:
                emit_layers.append(nn.ReLU())
                emit_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.emission = nn.Sequential(*emit_layers)
        
        # Learnable input injection for stability
        self.input_proj = nn.Linear(self.n_features, self.state_dim)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.size(0)
        
        # Encode input to initial state parameters (mean, logvar)
        x_flat = x.view(batch_size, -1)  # (batch, input_len * n_features)
        state_params = self.encoder(x_flat)  # (batch, state_dim * 2)
        
        # Split into mean and log variance
        state_mean = state_params[:, :self.state_dim]  # (batch, state_dim)
        # Use deterministic state for simplicity
        state = torch.tanh(state_mean)  # (batch, state_dim)
        
        # Get last input for input injection
        x_last = x[:, -1, :]  # (batch, n_features)
        input_inject = self.input_proj(x_last)  # (batch, state_dim)
        
        # Unroll state space model
        outputs = []
        for t in range(self.output_len):
            # Transition with residual and input injection
            state = state + self.transition(state) + 0.1 * input_inject  # (batch, state_dim)
            state = torch.tanh(state)  # stabilize
            
            # Emit observation
            obs = self.emission(state)  # (batch, n_features)
            outputs.append(obs)
        
        # Stack outputs
        forecast = torch.stack(outputs, dim=1)  # (batch, output_len, n_features)
        
        forecast = self.revin(forecast, mode='denorm')  # (batch, output_len, n_features)
        return forecast

