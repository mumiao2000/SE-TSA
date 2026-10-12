import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class StateSpaceDecoder(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [64, 256, 1024],
            'decoder_dim': [128, 512, 2048],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.state_dim = self.config.get('state_dim', 256)
        self.decoder_dim = self.config.get('decoder_dim', 512)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: input_len * n_features -> state_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_len * n_features, self.decoder_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_dim, self.state_dim)
        )
        
        # Structured transition: diagonal + low-rank
        self.diag_transition = nn.Parameter(torch.randn(self.state_dim) * 0.01)
        self.low_rank_u = nn.Parameter(torch.randn(self.state_dim, self.state_dim // 4) * 0.01)
        self.low_rank_v = nn.Parameter(torch.randn(self.state_dim // 4, self.state_dim) * 0.01)
        
        # Evolution over output steps
        self.step_norm = nn.LayerNorm(self.state_dim)
        
        # Decoder: state -> n_features per step
        decoder_layers = []
        for i in range(self.n_layers):
            decoder_layers.append(nn.Linear(self.state_dim if i == 0 else self.decoder_dim, self.decoder_dim))
            decoder_layers.append(nn.GELU())
            decoder_layers.append(nn.Dropout(self.dropout))
        decoder_layers.append(nn.Linear(self.decoder_dim, n_features))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Flatten and encode
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1)  # (batch, input_len * n_features)
        state = self.encoder(x_flat)  # (batch, state_dim)
        state = self.step_norm(state)  # (batch, state_dim)
        
        # Evolve state and decode for each output step
        outputs = []
        for t in range(self.output_len):
            # Decode current state
            out_t = self.decoder(state)  # (batch, n_features)
            outputs.append(out_t)
            
            # Transition: diagonal + low-rank
            diag = self.diag_transition * state  # (batch, state_dim)
            low_rank = state @ self.low_rank_u @ self.low_rank_v  # (batch, state_dim)
            state = diag + low_rank  # (batch, state_dim)
            state = self.step_norm(state)  # (batch, state_dim)
        
        out = torch.stack(outputs, dim=1)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

