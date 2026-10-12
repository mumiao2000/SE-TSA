import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HierarchicalPatchGraphNet_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'base_dim': [64, 256, 1024],
            'n_scales': [2, 3, 4],
            'patch_sizes': [[2, 4, 8], [4, 8, 16], [2, 8, 24]],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.base_dim = self.config.get('base_dim', 256)
        self.n_scales = self.config.get('n_scales', 3)
        self.patch_sizes = self.config.get('patch_sizes', [2, 4, 8])[:self.n_scales]
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Ensure patch sizes work with input length
        self.patch_sizes = [p for p in self.patch_sizes if input_len // p >= 2]
        self.n_scales = len(self.patch_sizes)
        
        # Feature embedding
        self.feature_embed = nn.Linear(1, self.base_dim)
        
        # Encoder: multi-scale patch processing with graph conv
        self.encoders = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        
        dim = self.base_dim
        for i, ps in enumerate(self.patch_sizes):
            self.encoders.append(nn.Sequential(
                nn.Linear(dim if i > 0 else self.base_dim, dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim)
            ))
            # Patch downsampling: average pool then project
            self.downsample.append(nn.Sequential(
                nn.AvgPool1d(ps, stride=ps),  # operates on feature dim after transpose
            ))
            # Graph convolution for cross-feature interaction
            self.graph_convs.append(self._make_graph_conv(dim))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(dim * 2, dim)
        )
        
        # Decoder: upsample with skip connections
        self.decoders = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, ps in enumerate(reversed(self.patch_sizes)):
            self.upsample.append(nn.Upsample(scale_factor=ps, mode='linear', align_corners=False))
            self.decoders.append(nn.Sequential(
                nn.Linear(dim * 2 if i > 0 else dim, dim),  # skip connection
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim)
            ))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, self.output_len)
        )
        
    def _make_graph_conv(self, dim):
        # Simple graph conv: learnable adjacency wrapped in a proper Module
        return GraphConvLayer(self.n_features, dim)
    
    def apply_graph_conv(self, x, graph_conv):
        # x: (batch, seq_len, n_features, dim)
        batch, seq, n_feat, dim = x.shape
        
        # Reshape for graph operation
        x_flat = x.reshape(batch * seq, n_feat, dim)  # (batch*seq, n_features, dim)
        
        # Message passing
        x_msg = graph_conv(x_flat)  # (batch*seq, n_features, dim)
        
        # Residual
        out = x_flat + x_msg  # (batch*seq, n_features, dim)
        return out.reshape(batch, seq, n_feat, dim)  # (batch, seq_len, n_features, dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Embed: (batch, input_len, n_features, 1) -> (batch, input_len, n_features, base_dim)
        x = x.unsqueeze(-1)  # (batch, input_len, n_features, 1)
        x = self.feature_embed(x)  # (batch, input_len, n_features, base_dim)
        
        # Encoder path with downsampling
        skips = []
        h = x  # (batch, input_len, n_features, base_dim)
        
        for i, (enc, down, gconv) in enumerate(zip(self.encoders, self.downsample, self.graph_convs)):
            # Apply graph conv
            h = self.apply_graph_conv(h, gconv)  # (batch, seq, n_features, dim)
            
            # Process features
            batch, seq, n_feat, dim = h.shape
            h_flat = h.reshape(batch * seq, n_feat, dim)  # (batch*seq, n_features, dim)
            h_flat = enc(h_flat)  # (batch*seq, n_features, dim)
            h = h_flat.reshape(batch, seq, n_feat, dim)  # (batch, seq, n_features, dim)
            
            skips.append(h)
            
            # Downsample: pool over time
            # Transpose to (batch, n_features, dim, seq) for pooling
            h = h.permute(0, 2, 3, 1)  # (batch, n_features, dim, seq)
            h = h.reshape(batch * n_feat, dim, seq)  # (batch*n_features, dim, seq)
            h = F.avg_pool1d(h, self.patch_sizes[i], stride=self.patch_sizes[i])  # (batch*n_features, dim, seq//ps)
            new_seq = h.shape[-1]
            h = h.reshape(batch, n_feat, dim, new_seq)  # (batch, n_features, dim, new_seq)
            h = h.permute(0, 3, 1, 2)  # (batch, new_seq, n_features, dim)
        
        # Bottleneck
        batch, seq, n_feat, dim = h.shape
        h_flat = h.reshape(batch * seq, n_feat, dim)
        h_flat = self.bottleneck(h_flat)
        h = h_flat.reshape(batch, seq, n_feat, dim)
        
        # Decoder path with upsampling
        for i, (up, dec) in enumerate(zip(self.upsample, self.decoders)):
            # Upsample time dimension
            h = h.permute(0, 2, 3, 1)  # (batch, n_features, dim, seq)
            h = h.reshape(batch * n_feat, dim, h.shape[-1])  # (batch*n_features, dim, seq)
            h = up(h)  # (batch*n_features, dim, upsampled_seq)
            
            # Match skip connection size
            skip = skips[-(i+1)]  # (batch, skip_seq, n_features, dim)
            target_len = skip.shape[1]
            if h.shape[-1] != target_len:
                h = F.interpolate(h, size=target_len, mode='linear', align_corners=False)
            
            h = h.reshape(batch, n_feat, dim, target_len)
            h = h.permute(0, 3, 1, 2)  # (batch, target_len, n_features, dim)
            
            # Concatenate skip
            if i > 0:
                h = torch.cat([h, skip], dim=-1)  # (batch, target_len, n_features, dim*2)
            
            # Process
            batch_d, seq_d, n_feat_d, dim_d = h.shape
            h_flat = h.reshape(batch_d * seq_d, n_feat_d, dim_d)
            h_flat = dec(h_flat)
            h = h_flat.reshape(batch_d, seq_d, n_feat_d, dim_d // (2 if i > 0 else 1))
        
        # Final output: project to forecast horizon
        # h: (batch, seq, n_features, dim)
        # We need to get to (batch, output_len, n_features)
        # Use the final sequence representation
        h = h.mean(dim=1)  # (batch, n_features, dim) - global average pooling over time
        out = self.output_proj(h)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)


class GraphConvLayer(nn.Module):
    def __init__(self, n_features, dim):
        super().__init__()
        self.n_features = n_features
        self.dim = dim
        self.adj = nn.Parameter(torch.eye(n_features) * 0.5 + 0.5 / n_features)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # x: (batch, n_features, dim)
        adj = F.softmax(self.adj, dim=1)  # (n_features, n_features)
        x_msg = torch.einsum('ij,bjd->bid', adj, x)  # (batch, n_features, dim)
        x_msg = self.proj(x_msg)  # (batch, n_features, dim)
        x_msg = self.norm(x_msg)
        x_msg = F.relu(x_msg)
        return x_msg
