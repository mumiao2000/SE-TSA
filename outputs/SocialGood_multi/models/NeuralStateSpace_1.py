import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralStateSpace(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [32, 128, 512],
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.state_dim = self.config.get('state_dim', 128)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: maps input sequence to initial state
        self.encoder = nn.LSTM(n_features, self.hidden_dim, num_layers=self.n_layers, 
                               batch_first=True, dropout=self.dropout_rate if self.n_layers > 1 else 0,
                               bidirectional=True)
        self.encoder_proj = nn.Linear(self.hidden_dim * 2, self.state_dim)
        
        # Neural state transition: f(state) -> state_delta
        self.transition_net = nn.Sequential()
        for i in range(3):
            in_d = self.state_dim if i == 0 else self.hidden_dim
            out_d = self.state_dim if i == 2 else self.hidden_dim
            self.transition_net.add_module(f'layer_{i}', nn.Linear(in_d, out_d))
            if i < 2:
                self.transition_net.add_module(f'act_{i}', nn.SiLU())
                self.transition_net.add_module(f'drop_{i}', nn.Dropout(self.dropout_rate))
        
        # Gating for stable transitions
        self.transition_gate = nn.Linear(self.state_dim, self.state_dim)
        
        # Observation model: state -> output features
        self.observation_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, n_features)
        )
        
        # Learnable time steps for prediction
        self.time_embed = nn.Embedding(output_len, self.state_dim)
        
        # Mixing network for multi-step prediction
        self.mixer = nn.GRUCell(self.state_dim, self.state_dim)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Encode to initial state
        enc_out, (h_n, c_n) = self.encoder(x)  # enc_out: (batch, input_len, hidden_dim*2)
        # Use final hidden states
        h_forward = h_n[-2] if self.n_layers > 1 else h_n[0]  # (batch, hidden_dim)
        h_backward = h_n[-1]  # (batch, hidden_dim)
        h_final = torch.cat([h_forward, h_backward], dim=-1)  # (batch, hidden_dim*2)
        state = self.encoder_proj(h_final)  # (batch, state_dim)
        state = torch.tanh(state)  # bound initial state
        
        # Generate predictions by evolving state
        outputs = []
        current_state = state
        for t in range(self.output_len):
            # Apply transition with gating (residual-like for stability)
            delta = self.transition_net(current_state)  # (batch, state_dim)
            gate = torch.sigmoid(self.transition_gate(current_state))  # (batch, state_dim)
            new_state = gate * current_state + (1 - gate) * delta  # (batch, state_dim)
            
            # Add time embedding
            t_emb = self.time_embed(torch.tensor(t, device=x.device).expand(x.size(0)))  # (batch, state_dim)
            mixed = self.mixer(t_emb, new_state)  # (batch, state_dim)
            
            # Predict
            pred = self.observation_net(mixed)  # (batch, n_features)
            outputs.append(pred)
            
            current_state = mixed
        
        out = torch.stack(outputs, dim=1)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
