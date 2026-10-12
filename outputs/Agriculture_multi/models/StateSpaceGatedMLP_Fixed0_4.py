import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceGatedMLP_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [32, 128, 512],
            'hidden_dim': [128, 512, 2048],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 128)
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        # Learnable instance normalization parameters
        self.instance_norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.input_len, self.hidden_dim)
        
        # Structured state-space layers
        self.ss_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        self.log_A_real_list = nn.ParameterList()
        self.A_imag_list = nn.ParameterList()
        
        for _ in range(self.n_layers):
            # State-space parameters (diagonal structure for efficiency)
            # Linear layers go in ModuleDict (they are Modules)
            self.ss_layers.append(nn.ModuleDict({
                'B': nn.Linear(self.hidden_dim, self.state_dim),  # input to state
                'C': nn.Linear(self.state_dim, self.hidden_dim),  # state to output
                'D': nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),  # skip
            }))
            
            # Parameters stored separately in ParameterList (not in ModuleDict)
            self.log_A_real_list.append(nn.Parameter(torch.randn(self.state_dim) * 0.01))
            self.A_imag_list.append(nn.Parameter(torch.randn(self.state_dim) * 0.01))
            
            # Gating mechanism
            self.gate_layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Sigmoid()
            ))
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.output_len)
        )
        
        # Channel-wise scale and shift for denormalization
        self.channel_scale = nn.Parameter(torch.ones(self.n_features))
        self.channel_shift = nn.Parameter(torch.zeros(self.n_features))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # Instance normalization across time
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.instance_norm(x)  # (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Store mean for denormalization
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        x_std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        
        # Transpose for feature-wise processing
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project to hidden dim
        h = self.input_proj(x)  # (batch, n_features, hidden_dim)
        
        # Apply state-space layers
        for i, (ss, gate) in enumerate(zip(self.ss_layers, self.gate_layers)):
            # State-space recurrence (simplified, parallelizable version)
            # Treat as structured linear transformation
            
            # Compute state: h_proj = B(h)
            h_proj = ss['B'](h)  # (batch, n_features, state_dim)
            
            # Structured matrix multiplication (diagonal approximation)
            log_A_real = self.log_A_real_list[i]  # (state_dim,)
            A_imag = self.A_imag_list[i]  # (state_dim,)
            A_real = -torch.exp(log_A_real)  # ensure stability (negative real part)
            
            # Apply as element-wise (diagonal structure)
            state = h_proj * A_real.unsqueeze(0).unsqueeze(0)  # (batch, n_features, state_dim)
            state = state + h_proj * 0.1  # simplified discretization
            
            # Output from state
            h_ss = ss['C'](state)  # (batch, n_features, hidden_dim)
            h_skip = ss['D'](h)  # (batch, n_features, hidden_dim)
            
            h_new = h_ss + h_skip  # (batch, n_features, hidden_dim)
            
            # Gating
            gate_input = torch.cat([h, h_new], dim=-1)  # (batch, n_features, hidden_dim*2)
            g = gate(gate_input)  # (batch, n_features, hidden_dim)
            
            h = g * h_new + (1 - g) * h  # (batch, n_features, hidden_dim)
        
        # Output projection
        out = self.output_mlp(h)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Denormalize
        out = out * x_std + x_mean  # (batch, output_len, n_features)
        
        # Apply learned channel scaling
        out = out * self.channel_scale.unsqueeze(0).unsqueeze(0) + self.channel_shift.unsqueeze(0).unsqueeze(0)  # (batch, output_len, n_features)
        
        return out

