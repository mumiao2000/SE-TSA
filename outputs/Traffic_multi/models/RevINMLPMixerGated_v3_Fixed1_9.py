import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class RevINMLPMixerGated_v3_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_mixer_layers': [1, 4, 8],
            'temporal_kernel_sizes': [[3, 5, 7], [5, 11, 21]],
            'expansion_factor': [4, 8, 16],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        c = self.config or {}
        self.hidden_dim = c.get('hidden_dim', 512)
        self.n_mixer_layers = c.get('n_mixer_layers', 4)
        self.temporal_kernel_sizes = c.get('temporal_kernel_sizes', [5, 11, 21])
        self.expansion_factor = c.get('expansion_factor', 8)
        self.dropout_rate = c.get('dropout', 0.1)
        
        n_kernels = len(self.temporal_kernel_sizes)
        # Ensure divisibility: compute per-kernel output channels
        self.conv_out_channels = self.hidden_dim // n_kernels
        # Actual total channels from all convs (may differ from hidden_dim if not divisible)
        self.conv_total_channels = self.conv_out_channels * n_kernels
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale temporal convolution front-end (channel-independent)
        self.temporal_convs = nn.ModuleList()
        for k in self.temporal_kernel_sizes:
            padding = k // 2
            self.temporal_convs.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=self.conv_out_channels,
                    kernel_size=k,
                    padding=padding,
                    groups=1
                )
            )
        
        # Total channels from multi-scale convs
        self.temporal_out_dim = self.hidden_dim
        
        # Project pooled conv features to hidden_dim
        self.temporal_proj = nn.Linear(self.conv_total_channels, self.temporal_out_dim)
        
        # Temporal mixing: operates on (batch, n_features, hidden_dim)
        self.temporal_mix_layers = nn.ModuleList()
        for _ in range(self.n_mixer_layers):
            self.temporal_mix_layers.append(
                self._create_gated_mlp(self.temporal_out_dim, self.expansion_factor)
            )
        
        # Channel mixing: operates on (batch, hidden_dim, n_features)
        self.channel_mix_layers = nn.ModuleList()
        for _ in range(self.n_mixer_layers):
            self.channel_mix_layers.append(
                self._create_gated_mlp(self.n_features, self.expansion_factor)
            )
        
        # Layer norms for residual paths
        self.temporal_norms = nn.ModuleList([
            nn.LayerNorm(self.temporal_out_dim) for _ in range(self.n_mixer_layers)
        ])
        self.channel_norms = nn.ModuleList([
            nn.LayerNorm(self.n_features) for _ in range(self.n_mixer_layers)
        ])
        
        # Output projection: from hidden_dim to output_len
        self.output_proj = nn.Linear(self.temporal_out_dim, self.output_len)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Residual scaling parameter
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def _create_gated_mlp(self, dim, expansion_factor):
        """Create GELU-GLU gated MLP block."""
        hidden_dim = int(dim * expansion_factor)
        return nn.ModuleDict({
            'fc1': nn.Linear(dim, hidden_dim * 2),  # *2 for gate and value
            'fc2': nn.Linear(hidden_dim, dim),
        })
    
    def _gated_mlp_forward(self, x, mlp_dict):
        """Forward through GELU-GLU gated MLP."""
        # x: (..., dim)
        x_proj = mlp_dict['fc1'](x)  # (..., hidden_dim * 2)
        # Split into gate and value
        gate, value = torch.chunk(x_proj, 2, dim=-1)  # each (..., hidden_dim)
        # GELU-GLU: apply GELU to gate, multiply with value
        gate = F.gelu(gate)
        x_gated = gate * value  # (..., hidden_dim)
        # Project back
        x_out = mlp_dict['fc2'](x_gated)  # (..., dim)
        return x_out
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-independent processing: process each feature separately
        # Reshape to (batch * n_features, 1, input_len) for conv1d
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_reshaped = x_perm.reshape(batch_size * self.n_features, 1, self.input_len)  # (batch*n_features, 1, input_len)
        
        # Multi-scale temporal convolutions
        conv_outputs = []
        for conv in self.temporal_convs:
            conv_out = conv(x_reshaped)  # (batch*n_features, conv_out_channels, input_len)
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features
        x_conv = torch.cat(conv_outputs, dim=1)  # (batch*n_features, conv_total_channels, input_len)
        
        # Global average pooling over temporal dimension
        x_conv = F.adaptive_avg_pool1d(x_conv, 1)  # (batch*n_features, conv_total_channels, 1)
        x_conv = x_conv.squeeze(-1)  # (batch*n_features, conv_total_channels)
        
        # Project to temporal_out_dim
        x_conv = self.temporal_proj(x_conv)  # (batch*n_features, temporal_out_dim)
        
        # Reshape back to (batch, n_features, temporal_out_dim)
        x_mixer = x_conv.reshape(batch_size, self.n_features, self.temporal_out_dim)  # (batch, n_features, hidden_dim)
        
        # MLP-Mixer blocks
        for i in range(self.n_mixer_layers):
            # Temporal mixing: mix across hidden_dim dimension
            # Input: (batch, n_features, hidden_dim)
            x_res = x_mixer  # (batch, n_features, hidden_dim)
            x_mixer = self.temporal_norms[i](x_mixer)  # (batch, n_features, hidden_dim)
            x_mixer = self._gated_mlp_forward(x_mixer, self.temporal_mix_layers[i])  # (batch, n_features, hidden_dim)
            x_mixer = self.dropout(x_mixer)
            x_mixer = x_mixer + self.residual_scale * x_res  # (batch, n_features, hidden_dim)
            
            # Channel mixing: mix across n_features dimension
            # Transpose to (batch, hidden_dim, n_features)
            x_res = x_mixer  # (batch, n_features, hidden_dim)
            x_mixer = x_mixer.permute(0, 2, 1)  # (batch, hidden_dim, n_features)
            x_mixer = self.channel_norms[i](x_mixer)  # (batch, hidden_dim, n_features)
            x_mixer = self._gated_mlp_forward(x_mixer, self.channel_mix_layers[i])  # (batch, hidden_dim, n_features)
            x_mixer = self.dropout(x_mixer)
            x_mixer = x_mixer + self.residual_scale * x_res.permute(0, 2, 1)  # (batch, hidden_dim, n_features)
            x_mixer = x_mixer.permute(0, 2, 1)  # (batch, n_features, hidden_dim)
        
        # Output projection: (batch, n_features, hidden_dim) -> (batch, n_features, output_len)
        out = self.output_proj(x_mixer)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

