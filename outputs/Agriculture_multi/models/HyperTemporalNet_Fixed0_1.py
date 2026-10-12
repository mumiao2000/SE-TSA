import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperTemporalNet_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hyper_dim': [32, 128, 512],
            'n_filters': [16, 64, 256],
            'kernel_size': [3, 7, 15],
            'n_hyper_layers': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hyper_dim = self.config.get('hyper_dim', 128)
        self.n_filters = self.config.get('n_filters', 64)
        self.kernel_size = self.config.get('kernel_size', 7)
        self.n_hyper_layers = self.config.get('n_hyper_layers', 4)
        
        # Ensure kernel size is odd
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Context encoder for hypernetwork
        self.context_encoder = nn.Sequential(
            nn.Linear(self.n_features * 3, self.hyper_dim),  # mean, std, last
            nn.LayerNorm(self.hyper_dim),
            nn.GELU(),
            nn.Linear(self.hyper_dim, self.hyper_dim)
        )
        
        # Hypernetwork generates TCN weights
        # For simplicity: generate a set of filter weights for each layer
        self.hyper_mlp = nn.Sequential(
            nn.Linear(self.hyper_dim, self.hyper_dim * 2),
            nn.GELU(),
            nn.Linear(self.hyper_dim * 2, self.hyper_dim)
        )
        
        # Weight generators (output shapes depend on layer)
        self.filter_gen1 = nn.Linear(self.hyper_dim, self.n_filters * self.n_features * self.kernel_size)
        self.bias_gen1 = nn.Linear(self.hyper_dim, self.n_filters)
        self.filter_gen2 = nn.Linear(self.hyper_dim, self.n_filters * self.n_filters * self.kernel_size)
        self.bias_gen2 = nn.Linear(self.hyper_dim, self.n_filters)
        self.filter_gen3 = nn.Linear(self.hyper_dim, self.n_features * self.n_filters * self.kernel_size)
        self.bias_gen3 = nn.Linear(self.hyper_dim, self.n_features)
        
        # Static fallback convolutions (for stability)
        self.static_conv1 = nn.Conv1d(self.n_features, self.n_filters, self.kernel_size, padding=self.kernel_size//2)
        self.static_conv2 = nn.Conv1d(self.n_filters, self.n_filters, self.kernel_size, padding=self.kernel_size//2)
        self.static_conv3 = nn.Conv1d(self.n_filters, self.n_features, self.kernel_size, padding=self.kernel_size//2)
        
        # Temporal attention for output generation - per-feature linear projection from input_len to output_len
        self.temporal_attn = nn.Sequential(
            nn.Linear(self.input_len, self.input_len // 4),
            nn.GELU(),
            nn.Linear(self.input_len // 4, self.output_len)
        )
        
        # Direct mapping
        self.direct_map = nn.Linear(self.input_len, self.output_len)

    def apply_dynamic_conv(self, x, filters, biases, in_channels, out_channels):
        # x: (batch, in_channels, seq_len)
        # filters: (batch, out_channels * in_channels * kernel_size)
        batch_size = x.shape[0]
        seq_len = x.shape[2]
        
        # Reshape filters to (batch * out_channels, in_channels, kernel_size)
        filters = filters.view(batch_size * out_channels, in_channels, self.kernel_size)
        biases = biases.view(batch_size * out_channels)
        
        # Apply grouped convolution
        # Need to process each batch element separately
        outputs = []
        for b in range(batch_size):
            x_b = x[b:b+1]  # (1, in_channels, seq_len)
            w_b = filters[b * out_channels:(b+1) * out_channels]  # (out_channels, in_channels, kernel_size)
            b_b = biases[b * out_channels:(b+1) * out_channels]  # (out_channels,)
            
            out_b = F.conv1d(x_b, w_b, b_b, padding=self.kernel_size//2)  # (1, out_channels, seq_len)
            outputs.append(out_b)
        
        return torch.cat(outputs, dim=0)  # (batch, out_channels, seq_len)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Instance normalization
        x_norm = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Compute context statistics
        x_mean = x_norm.mean(dim=1)  # (batch, n_features)
        x_std = x_norm.std(dim=1) + 1e-5  # (batch, n_features)
        x_last = x_norm[:, -1, :]  # (batch, n_features)
        
        context = torch.cat([x_mean, x_std, x_last], dim=-1)  # (batch, n_features * 3)
        
        # Generate hypernetwork conditioning
        hyper_input = self.context_encoder(context)  # (batch, hyper_dim)
        hyper_state = self.hyper_mlp(hyper_input)  # (batch, hyper_dim)
        
        # Generate dynamic weights
        w1 = self.filter_gen1(hyper_state)  # (batch, n_filters * n_features * kernel_size)
        b1 = self.bias_gen1(hyper_state)  # (batch, n_filters)
        w2 = self.filter_gen2(hyper_state)  # (batch, n_filters * n_filters * kernel_size)
        b2 = self.bias_gen2(hyper_state)  # (batch, n_filters)
        w3 = self.filter_gen3(hyper_state)  # (batch, n_features * n_filters * kernel_size)
        b3 = self.bias_gen3(hyper_state)  # (batch, n_features)
        
        # Apply dynamic convolutions
        x_perm = x_norm.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Mix dynamic and static for stability
        alpha = torch.sigmoid(hyper_input[:, 0:1]).unsqueeze(-1)  # (batch, 1, 1)
        
        # Layer 1
        h1_dynamic = self.apply_dynamic_conv(x_perm, w1, b1, self.n_features, self.n_filters)
        h1_static = self.static_conv1(x_perm)
        h1 = alpha * h1_dynamic + (1 - alpha) * h1_static  # (batch, n_filters, input_len)
        h1 = F.gelu(h1)
        
        # Layer 2
        h2_dynamic = self.apply_dynamic_conv(h1, w2, b2, self.n_filters, self.n_filters)
        h2_static = self.static_conv2(h1)
        h2 = alpha * h2_dynamic + (1 - alpha) * h2_static  # (batch, n_filters, input_len)
        h2 = F.gelu(h2)
        
        # Layer 3
        h3_dynamic = self.apply_dynamic_conv(h2, w3, b3, self.n_filters, self.n_features)
        h3_static = self.static_conv3(h2)
        h3 = alpha * h3_dynamic + (1 - alpha) * h3_static  # (batch, n_features, input_len)
        
        # Temporal attention to produce output
        # Reshape to (batch * n_features, input_len) and apply linear projection
        h3_reshaped = h3.permute(0, 2, 1).reshape(batch_size * self.n_features, self.input_len)  # (batch * n_features, input_len)
        out = self.temporal_attn(h3_reshaped)  # (batch * n_features, output_len)
        out = out.view(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Direct residual
        direct = self.direct_map(x_perm).permute(0, 2, 1)  # (batch, output_len, n_features)
        out = out + direct * 0.5
        
        # Denormalize
        out = out * x_std.unsqueeze(1) + x_mean.unsqueeze(1)  # (batch, output_len, n_features)
        
        return out

