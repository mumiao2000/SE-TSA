import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class MLPMixerChannelIndependent(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'expansion_factor': [2, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.expansion_factor = self.config.get('expansion_factor', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Temporal mixing: mix across time dimension for each feature independently
        self.temporal_mlps = nn.ModuleList()
        # Channel mixing: mix across features for each time step
        self.channel_mlps = nn.ModuleList()
        self.temporal_norms = nn.ModuleList()
        self.channel_norms = nn.ModuleList()
        
        for _ in range(self.n_layers):
            # Temporal MLP: operates on (batch, n_features, input_len)
            self.temporal_norms.append(nn.LayerNorm(self.input_len))
            self.temporal_mlps.append(nn.Sequential(
                nn.Linear(self.input_len, self.input_len * self.expansion_factor),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.input_len * self.expansion_factor, self.input_len),
                nn.Dropout(self.dropout)
            ))
            
            # Channel MLP: operates on (batch, input_len, n_features) -> actually we need to think carefully
            # After temporal, we have (batch, n_features, input_len). We transpose to (batch, input_len, n_features) for channel mixing
            self.channel_norms.append(nn.LayerNorm(self.n_features))
            self.channel_mlps.append(nn.Sequential(
                nn.Linear(self.n_features, self.n_features * self.expansion_factor),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.n_features * self.expansion_factor, self.n_features),
                nn.Dropout(self.dropout)
            ))
        
        # Final projection from input_len to output_len for each feature
        self.output_proj = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose to (batch, n_features, input_len) for channel-independent processing
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        for i in range(self.n_layers):
            # Temporal mixing
            residual = x  # (batch, n_features, input_len)
            x = self.temporal_norms[i](x)  # (batch, n_features, input_len)
            x = self.temporal_mlps[i](x) + residual  # (batch, n_features, input_len)
            
            # Channel mixing
            residual = x  # (batch, n_features, input_len)
            x = x.permute(0, 2, 1)  # (batch, input_len, n_features)
            x = self.channel_norms[i](x)  # (batch, input_len, n_features)
            x = self.channel_mlps[i](x)  # (batch, input_len, n_features)
            x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
            x = x + residual  # (batch, n_features, input_len)
        
        # Project to output length
        x = self.output_proj(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x

