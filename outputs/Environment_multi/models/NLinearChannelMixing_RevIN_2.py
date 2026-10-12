import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NLinearChannelMixing_RevIN(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'dropout': [0.0, 0.3],
            'n_mixer_layers': [1, 2, 4],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.n_mixer_layers = self.config.get('n_mixer_layers', 2)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # NLinear: one linear layer per feature, mapping input_len -> output_len
        self.linears = nn.ModuleList([
            nn.Linear(self.input_len, self.output_len) 
            for _ in range(self.n_features)
        ])
        
        # Gated channel mixing MLPs with residual connections
        self.mixer_layers = nn.ModuleList()
        self.mixer_norms = nn.ModuleList()
        for _ in range(self.n_mixer_layers):
            self.mixer_layers.append(nn.Sequential(
                nn.Linear(self.n_features, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.n_features * 2),  # gate + value
            ))
            self.mixer_norms.append(nn.LayerNorm(self.n_features))
        
        # Final projection if needed
        self.output_norm = nn.LayerNorm(self.n_features)
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # NLinear: apply separate linear for each feature
        # Transpose to (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Apply feature-specific linear layers
        outs = []
        for i, linear in enumerate(self.linears):
            feat = x[:, i, :]  # (batch, input_len)
            out = linear(feat)  # (batch, output_len)
            outs.append(out)
        x = torch.stack(outs, dim=2)  # (batch, output_len, n_features)
        
        # Channel mixing with gating and residual connections
        for mixer, norm in zip(self.mixer_layers, self.mixer_norms):
            residual = x  # (batch, output_len, n_features)
            x = norm(x)  # (batch, output_len, n_features)
            
            # Apply mixer
            mixed = mixer(x)  # (batch, output_len, n_features * 2)
            gate, value = mixed.chunk(2, dim=-1)  # each (batch, output_len, n_features)
            gate = torch.sigmoid(gate)  # (batch, output_len, n_features)
            x = gate * value  # (batch, output_len, n_features)
            x = self.dropout(x)
            x = x + residual  # (batch, output_len, n_features)
        
        # Final normalization
        x = self.output_norm(x)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x  # (batch, output_len, n_features)

