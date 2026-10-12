import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperTemporalConv_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'embed_dim': [16, 64, 256],
            'n_scales': [2, 4, 8],
            'kernel_size': [3, 7, 15],
            'hyper_hidden': [32, 128, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.embed_dim = self.config.get('embed_dim', 64)
        self.n_scales = self.config.get('n_scales', 4)
        self.kernel_size = self.config.get('kernel_size', 7)
        self.hyper_hidden = self.config.get('hyper_hidden', 128)
        
        # Feature embeddings (learned)
        self.feature_emb = nn.Embedding(self.n_features, self.embed_dim)
        
        # Hypernetwork: feature embedding -> convolution weights
        # Generates weights for depthwise conv: (kernel_size,) per feature per scale
        self.hyper_net = nn.Sequential(
            nn.Linear(self.embed_dim, self.hyper_hidden),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden, self.hyper_hidden // 2),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden // 2, self.n_scales * self.kernel_size)
        )
        
        # Hypernetwork for pointwise mixing weights
        self.hyper_mix = nn.Sequential(
            nn.Linear(self.embed_dim, self.hyper_hidden // 2),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden // 2, self.n_scales)
        )
        
        # Learnable dilation rates per scale
        self.dilations = [2**i for i in range(self.n_scales)]
        
        # Output projection
        # Aggregate multi-scale features and project to output
        self.context_compress = nn.Linear(input_len, output_len)
        
        # Final MLP for nonlinear mapping
        self.output_mlp = nn.Sequential(
            nn.Linear(self.n_features * self.n_scales, self.n_features * 4),
            nn.ReLU(),
            nn.Linear(self.n_features * 4, self.n_features * 2),
            nn.ReLU(),
            nn.Linear(self.n_features * 2, self.n_features)
        )
        
        # Instance norm
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Instance normalize
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.norm(x)  # (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Generate feature indices
        feat_idx = torch.arange(self.n_features, device=x.device)  # (n_features,)
        feat_emb = self.feature_emb(feat_idx)  # (n_features, embed_dim)
        
        # Generate convolution weights via hypernetwork
        conv_weights = self.hyper_net(feat_emb)  # (n_features, n_scales * kernel_size)
        conv_weights = conv_weights.view(self.n_features, self.n_scales, self.kernel_size)
        # Normalize weights
        conv_weights = F.softmax(conv_weights, dim=-1)  # (n_features, n_scales, kernel_size)
        
        # Generate mixing weights
        mix_weights = self.hyper_mix(feat_emb)  # (n_features, n_scales)
        mix_weights = F.softmax(mix_weights, dim=-1)  # (n_features, n_scales)
        
        # Apply depthwise separable convolutions with hyper-generated weights
        # x: (batch, input_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        multi_scale_features = []
        for s in range(self.n_scales):
            dilation = self.dilations[s]
            # Calculate padding to maintain output length = input_len
            # For causal/dilated conv: padding = (kernel_size - 1) * dilation // 2 for same padding
            padding = (self.kernel_size - 1) * dilation // 2
            
            # Prepare weights for conv1d: (out_channels=1, in_channels/groups=1, kernel_size)
            # We need to process each feature separately with its own filter
            out_s = []
            for f in range(self.n_features):
                x_f = x[:, f:f+1, :]  # (batch, 1, input_len)
                w_f = conv_weights[f, s, :]  # (kernel_size,)
                # Reshape weight for conv1d: (out_channels=1, in_channels=1, kernel_size)
                w_f = w_f.view(1, 1, self.kernel_size)  # (1, 1, kernel_size)
                
                # Apply 1D convolution with dilation
                out_f = F.conv1d(x_f, w_f, padding=padding, dilation=dilation)  # (batch, 1, input_len)
                out_s.append(out_f)
            
            out_s = torch.cat(out_s, dim=1)  # (batch, n_features, input_len)
            out_s = out_s.permute(0, 2, 1)  # (batch, input_len, n_features)
            multi_scale_features.append(out_s)
        
        # Stack and mix - all should be (batch, input_len, n_features)
        multi_scale = torch.stack(multi_scale_features, dim=-1)  # (batch, input_len, n_features, n_scales)
        
        # Apply mixing weights
        mix = mix_weights.view(1, 1, self.n_features, self.n_scales)  # (1, 1, n_features, n_scales)
        mixed = (multi_scale * mix).sum(dim=-1)  # (batch, input_len, n_features)
        
        # Compress time dimension to output length
        mixed = mixed.permute(0, 2, 1)  # (batch, n_features, input_len)
        compressed = self.context_compress(mixed)  # (batch, n_features, output_len)
        compressed = compressed.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Pool multi_scale to output_len for richer features
        multi_scale_pooled = F.adaptive_avg_pool1d(
            multi_scale.permute(0, 3, 2, 1).reshape(batch_size * self.n_scales, self.n_features, self.input_len),
            self.output_len
        )  # (batch*n_scales, n_features, output_len)
        multi_scale_pooled = multi_scale_pooled.view(batch_size, self.n_scales, self.n_features, self.output_len)
        multi_scale_pooled = multi_scale_pooled.permute(0, 3, 2, 1)  # (batch, output_len, n_features, n_scales)
        multi_scale_pooled = multi_scale_pooled.reshape(batch_size, self.output_len, self.n_features * self.n_scales)
        
        # Use pooled multi-scale features through MLP
        out = self.output_mlp(multi_scale_pooled.reshape(batch_size * self.output_len, -1))
        out = out.view(batch_size, self.output_len, self.n_features)
        
        # Add residual from compressed path for stability
        out = out + compressed
        
        return out

