import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttentionMLP_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'temporal_hidden': [32, 128, 512],
            'n_heads': [2, 8, 32],
            'n_scales': [2, 3, 5],
            'decoder_hidden': [64, 256, 1024]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.temporal_hidden = self.config.get('temporal_hidden', 128)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_scales = self.config.get('n_scales', 3)
        self.decoder_hidden = self.config.get('decoder_hidden', 256)
        
        assert self.temporal_hidden % self.n_heads == 0, "temporal_hidden must be divisible by n_heads"
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Hierarchical temporal convolutions at multiple scales
        self.scale_convs = nn.ModuleList()
        for scale in range(1, self.n_scales + 1):
            kernel_size = 2 * scale + 1
            padding = scale
            conv = nn.Conv1d(
                self.n_features, 
                self.temporal_hidden // self.n_scales,
                kernel_size=kernel_size,
                padding=padding,
                dilation=1
            )
            self.scale_convs.append(conv)
        
        # Projection to ensure exact temporal_hidden dimension after concatenation
        self.scale_projection = nn.Linear(
            (self.temporal_hidden // self.n_scales) * self.n_scales,
            self.temporal_hidden
        )
        
        # Cross-channel attention: features attend to each other
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=self.temporal_hidden,
            num_heads=self.n_heads,
            batch_first=True
        )
        
        # Temporal pooling to compress input
        self.temporal_pool = nn.AdaptiveAvgPool1d(self.output_len)
        
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(self.temporal_hidden, self.decoder_hidden),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden, self.n_features)
        )
        
        # Direct mapping for stability
        self.direct_map = nn.Linear(self.input_len, self.output_len)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        batch_size = x.shape[0]  # (batch, input_len, n_features)
        
        # Hierarchical temporal convolutions
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        scale_outputs = []
        for conv in self.scale_convs:
            conv_out = F.relu(conv(x_perm))  # (batch, temp_hidden//n_scales, input_len)
            scale_outputs.append(conv_out)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_outputs, dim=1)  # (batch, (temp_hidden//n_scales)*n_scales, input_len)
        multi_scale = multi_scale.permute(0, 2, 1)  # (batch, input_len, (temp_hidden//n_scales)*n_scales)
        
        # Project to exact temporal_hidden dimension
        multi_scale = self.scale_projection(multi_scale)  # (batch, input_len, temporal_hidden)
        
        # Self-attention on the temporal sequence
        attn_out, _ = self.feature_attention(multi_scale, multi_scale, multi_scale)  # (batch, input_len, temporal_hidden)
        
        # Add residual
        attn_out = attn_out + multi_scale  # (batch, input_len, temporal_hidden)
        
        # Temporal pooling to output length
        attn_perm = attn_out.permute(0, 2, 1)  # (batch, temporal_hidden, input_len)
        pooled = self.temporal_pool(attn_perm)  # (batch, temporal_hidden, output_len)
        pooled = pooled.permute(0, 2, 1)  # (batch, output_len, temporal_hidden)
        
        # Decode to features
        out = self.decoder(pooled)  # (batch, output_len, n_features)
        
        # Add direct skip
        skip = self.direct_map(x_perm).permute(0, 2, 1)  # (batch, output_len, n_features)
        out = out + skip * 0.5  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

