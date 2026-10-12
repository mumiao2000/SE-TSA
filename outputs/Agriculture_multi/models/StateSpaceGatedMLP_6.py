import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceGatedMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [64, 256, 1024],
            'n_layers': [2, 4, 8],
            'context_ratio': [0.25, 0.5, 1.0],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.state_dim = self.config.get('state_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.context_ratio = self.config.get('context_ratio', 0.5)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input encoder: compress temporal dimension
        self.input_encoder = nn.Sequential(
            nn.Linear(self.input_len, self.state_dim // 2),
            nn.GELU(),
            nn.Linear(self.state_dim // 2, self.state_dim)
        )
        
        # State transition layers with selective gating
        self.state_layers = nn.ModuleList()
        self.context_encoders = nn.ModuleList()
        for _ in range(self.n_layers):
            context_dim = int(self.state_dim * self.context_ratio)
            self.state_layers.append(nn.ModuleDict({
                'A': nn.Linear(self.state_dim, self.state_dim, bias=False),  # state transition
                'B': nn.Linear(self.state_dim, self.state_dim),  # input modulation
                'C': nn.Sequential(  # output projection with gate
                    nn.Linear(self.state_dim, self.state_dim * 2),
                    nn.GLU(dim=-1)
                ),
                'D': nn.Linear(self.state_dim, self.state_dim),  # skip connection
                'norm': nn.LayerNorm(self.state_dim)
            }))
            # Context encoder for selective mechanism
            self.context_encoders.append(nn.Sequential(
                nn.Linear(self.state_dim, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, self.state_dim * 3)  # for A, B, C modulation
            ))
        
        # Temporal decoder
        self.temporal_decoder = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.state_dim // 2, self.output_len)
        )
        
        # Feature-wise output
        self.feature_proj = nn.Linear(self.n_features, self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Encode each feature's temporal pattern
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        state = self.input_encoder(x)  # (batch, n_features, state_dim)
        
        # Apply state transition layers
        for layer, ctx_enc in zip(self.state_layers, self.context_encoders):
            residual = state  # (batch, n_features, state_dim)
            
            # Compute context-dependent modulation
            context = ctx_enc(state.mean(dim=1, keepdim=True))  # (batch, 1, state_dim*3)
            context = context.expand(-1, self.n_features, -1)  # (batch, n_features, state_dim*3)
            
            # Split into A, B, C modulation
            mod_A, mod_B, mod_C = torch.chunk(context, 3, dim=-1)  # each (batch, n_features, state_dim)
            mod_A = torch.sigmoid(mod_A)
            mod_B = torch.sigmoid(mod_B)
            mod_C = torch.sigmoid(mod_C)
            
            # State transition: h' = A * h + B * input
            A_out = layer['A'](state) * mod_A  # (batch, n_features, state_dim)
            B_out = layer['B'](state) * mod_B  # (batch, n_features, state_dim)
            new_state = torch.tanh(A_out + B_out)  # (batch, n_features, state_dim)
            
            # Output with gate and skip
            C_out = layer['C'](new_state) * mod_C  # (batch, n_features, state_dim)
            D_out = layer['D'](state)  # (batch, n_features, state_dim)
            state = layer['norm'](C_out + D_out + residual)  # (batch, n_features, state_dim)
        
        # Decode to output length
        out = self.temporal_decoder(state)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Feature-wise refinement
        out = self.feature_proj(out)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
