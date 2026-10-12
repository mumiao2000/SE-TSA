import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class LiquidTimeConstantNetwork(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_neurons': [32, 128, 512],
            'tau_min': [0.1, 1.0, 10.0],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_neurons = self.config.get('n_neurons', 128)
        self.tau_min = self.config.get('tau_min', 1.0)
        self.dropout_rate = self.config.get('dropout', 0.0)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input encoding
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # LTC neuron parameters
        # Each neuron has: tau (time constant), A (state weight), B (input weight)
        self.tau_base = nn.Parameter(torch.ones(self.n_neurons) * self.tau_min)
        self.tau_gate = nn.Sequential(
            nn.Linear(self.n_features + self.hidden_dim, self.n_neurons),
            nn.Sigmoid()
        )
        
        # State transition: dh/dt = -h/tau + sigmoid(W*h + U*x + b)
        self.W_h = nn.Parameter(torch.randn(self.n_neurons, self.hidden_dim) * 0.01)
        self.U_x = nn.Parameter(torch.randn(self.n_neurons, self.n_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(self.n_neurons))
        
        # Mapping from hidden to neuron space and back
        self.h_to_neuron = nn.Linear(self.hidden_dim, self.n_neurons)
        self.neuron_to_h = nn.Linear(self.n_neurons, self.hidden_dim)
        
        # Feature-specific time constant modulation
        self.feature_tau = nn.Sequential(
            nn.Linear(self.n_features, self.n_features // 2),
            nn.GELU(),
            nn.Linear(self.n_features // 2, self.n_features),
            nn.Softplus()
        )
        
        # Output: predict sequence
        self.output_gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Learned initial state
        self.h0 = nn.Parameter(torch.zeros(self.hidden_dim))
    
    def ltc_step(self, h, x, dt):
        # h: (batch, hidden_dim), x: (batch, n_features)
        # Compute adaptive time constants
        tau_input = torch.cat([x, h], dim=-1)  # (batch, n_features + hidden_dim)
        tau_gate = self.tau_gate(tau_input)  # (batch, n_neurons)
        tau = self.tau_min + tau_gate * (10.0 - self.tau_min)  # bounded time constants  # (batch, n_neurons)
        
        # Map hidden to neurons
        n = self.h_to_neuron(h)  # (batch, n_neurons)
        
        # Compute nonlinear input
        # W_h @ h needs proper dimensions
        # n is (batch, n_neurons), we need (batch, n_neurons) from W_h @ h
        # W_h is (n_neurons, hidden_dim), h is (batch, hidden_dim)
        Wh = torch.matmul(h, self.W_h.T)  # (batch, n_neurons)
        Ux = torch.matmul(x, self.U_x.T)  # (batch, n_neurons)
        f_input = Wh + Ux + self.b  # (batch, n_neurons)
        f_activation = torch.sigmoid(f_input)  # (batch, n_neurons)
        
        # LTC dynamics: dn/dt = (-n + f_activation) / tau
        dn = (-n + f_activation) / (tau + 1e-6)  # (batch, n_neurons)
        n_new = n + dn * dt  # Euler step  # (batch, n_neurons)
        
        # Map back to hidden
        h_new = self.neuron_to_h(n_new)  # (batch, hidden_dim)
        h_new = torch.tanh(h_new)
        
        return h_new
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch = x.size(0)
        
        # Initialize hidden state
        h = self.h0.unsqueeze(0).expand(batch, -1)  # (batch, hidden_dim)
        
        # Process input sequence with adaptive time steps
        dt = 1.0 / self.input_len  # normalized time step
        
        for t in range(self.input_len):
            x_t = x[:, t, :]  # (batch, n_features)
            h = self.ltc_step(h, x_t, dt)  # (batch, hidden_dim)
            h = self.dropout(h)
        
        # Generate output sequence
        # Use GRU to decode with LTC state as initial condition
        outputs = []
        h_gru = h  # (batch, hidden_dim)
        
        for t in range(self.output_len):
            # Use previous prediction as input for autoregressive generation
            if t == 0:
                inp = self.input_proj(x[:, -1, :])  # (batch, hidden_dim)
            else:
                inp = self.input_proj(out_t)  # (batch, hidden_dim)
            
            h_gru = self.output_gru(inp, h_gru)  # (batch, hidden_dim)
            out_t = self.output_proj(h_gru)  # (batch, n_features)
            outputs.append(out_t)
        
        out = torch.stack(outputs, dim=1)  # (batch, output_len, n_features)
        
        # Feature-specific time constant smoothing
        feature_tau = self.feature_tau(out.mean(dim=1))  # (batch, n_features)
        # Apply exponential smoothing with learned time constants
        alpha = torch.sigmoid(-1.0 / (feature_tau + 0.1))  # (batch, n_features)
        # Simple correction
        out = out * alpha.unsqueeze(1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

