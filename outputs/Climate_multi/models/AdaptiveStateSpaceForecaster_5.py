import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveStateSpaceForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'state_dim': [16, 64, 256],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.state_dim = self.config.get('state_dim', 64)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.state_dim)
        
        # Multi-resolution SSM layers
        self.ssm_layers = nn.ModuleList()
        for i in range(self.n_layers):
            # Different dilation for each layer to capture different resolutions
            dilation = 2 ** (i % 3)
            self.ssm_layers.append(
                AdaptiveSSMBlock(self.state_dim, dilation, self.dropout_rate)
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.state_dim // 2, self.n_features)
        )
        
        # Direct skip from input to output for stability
        self.skip_proj = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save for skip connection
        x_orig = x  # (batch, input_len, n_features)
        
        # Project to state space
        h = self.input_proj(x)  # (batch, input_len, state_dim)
        
        # Apply SSM layers with residual connections
        for layer in self.ssm_layers:
            h = h + layer(h)  # (batch, input_len, state_dim)
        
        # Take last state and project to output length
        h_last = h[:, -1:, :]  # (batch, 1, state_dim)
        
        # Expand to output length with learned temporal pattern
        h_expanded = h_last.repeat(1, self.output_len, 1)  # (batch, output_len, state_dim)
        
        # Add temporal position encoding
        pos = torch.arange(self.output_len, device=x.device).float().unsqueeze(0).unsqueeze(-1)  # (1, output_len, 1)
        pos_enc = torch.sin(pos * 0.1) + torch.cos(pos * 0.05)  # (1, output_len, 1)
        h_expanded = h_expanded + pos_enc * 0.1  # (batch, output_len, state_dim)
        
        # Project to features
        out = self.output_proj(h_expanded)  # (batch, output_len, n_features)
        
        # Add skip connection from input
        skip = x_orig.permute(0, 2, 1)  # (batch, n_features, input_len)
        skip = self.skip_proj(skip)  # (batch, n_features, output_len)
        skip = skip.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = out + skip * 0.1  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class AdaptiveSSMBlock(nn.Module):
    def __init__(self, state_dim, dilation, dropout):
        super().__init__()
        self.state_dim = state_dim
        self.dilation = dilation
        
        # Input-dependent state transition
        self.transition_net = nn.Sequential(
            nn.Linear(state_dim, state_dim // 4),
            nn.SiLU(),
            nn.Linear(state_dim // 4, state_dim * state_dim)
        )
        
        self.norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Dilation-aware temporal conv
        self.conv = nn.Conv1d(state_dim, state_dim, kernel_size=3, padding=dilation, dilation=dilation, groups=state_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, state_dim)
        batch, seq_len, _ = x.shape
        
        # Compute input-dependent transition matrix from last timestep
        x_last = x[:, -1, :]  # (batch, state_dim)
        A_flat = self.transition_net(x_last)  # (batch, state_dim * state_dim)
        A = A_flat.view(batch, self.state_dim, self.state_dim)  # (batch, state_dim, state_dim)
        
        # Apply state transition
        x_proj = torch.bmm(x, A.transpose(1, 2))  # (batch, seq_len, state_dim)
        
        # Temporal conv with dilation
        x_conv = x.transpose(1, 2)  # (batch, state_dim, seq_len)
        x_conv = self.conv(x_conv)  # (batch, state_dim, seq_len)
        x_conv = x_conv[:, :, :seq_len].transpose(1, 2)  # (batch, seq_len, state_dim)
        
        out = self.norm(x_proj + x_conv)  # (batch, seq_len, state_dim)
        out = self.dropout(out)  # (batch, seq_len, state_dim)
        return out
