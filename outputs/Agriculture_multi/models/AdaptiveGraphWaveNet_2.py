import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveGraphWaveNet(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'graph_dim': [32, 128, 512],
            'n_graph_layers': [1, 3, 6],
            'wavelet_scales': [2, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.graph_dim = self.config.get('graph_dim', 128)
        self.n_graph_layers = self.config.get('n_graph_layers', 3)
        self.wavelet_scales = self.config.get('wavelet_scales', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Instance normalization for stability
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Adaptive adjacency: node embeddings for graph structure learning
        self.node_emb = nn.Parameter(torch.randn(self.n_features, self.graph_dim))
        
        # Input projection
        self.input_proj = nn.Linear(1, self.graph_dim)  # Process each feature independently
        
        # Graph-wavelet layers
        self.graph_layers = nn.ModuleList()
        for _ in range(self.n_graph_layers):
            self.graph_layers.append(
                GraphWaveletLayer(self.graph_dim, self.graph_dim, self.wavelet_scales, self.dropout)
            )
        
        # Temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output MLP
        self.output_net = nn.Sequential(
            nn.Linear(self.graph_dim * self.n_features, self.graph_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.graph_dim * 4, self.output_len * self.n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Instance normalize: transpose for norm
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.norm(x)  # (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Reshape for per-feature processing: (batch * n_features, input_len, 1)
        x = x.reshape(batch_size * self.n_features, self.input_len, 1)
        
        # Project to graph dimension
        h = self.input_proj(x)  # (batch * n_features, input_len, graph_dim)
        
        # Reshape back: (batch, input_len, n_features, graph_dim)
        h = h.view(batch_size, self.input_len, self.n_features, self.graph_dim)
        
        # Adaptive adjacency from node embeddings
        adj = F.softmax(F.relu(torch.matmul(self.node_emb, self.node_emb.t())), dim=-1)
        # adj: (n_features, n_features)
        
        # Apply graph-wavelet layers
        for layer in self.graph_layers:
            h = layer(h, adj)  # (batch, input_len, n_features, graph_dim)
        
        # Temporal pooling per feature
        h = h.permute(0, 2, 3, 1)  # (batch, n_features, graph_dim, input_len)
        h = h.reshape(batch_size * self.n_features, self.graph_dim, self.input_len)
        h = self.temporal_pool(h).squeeze(-1)  # (batch * n_features, graph_dim)
        h = h.view(batch_size, self.n_features, self.graph_dim)  # (batch, n_features, graph_dim)
        
        # Flatten and predict
        h = h.reshape(batch_size, self.n_features * self.graph_dim)  # (batch, n_features * graph_dim)
        out = self.output_net(h)  # (batch, output_len * n_features)
        out = out.view(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        return out


class GraphWaveletLayer(nn.Module):
    def __init__(self, in_dim, out_dim, wavelet_scales, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.wavelet_scales = wavelet_scales
        
        # Graph convolution weights (Chebyshev approximation)
        self.gcn_weights = nn.Parameter(torch.randn(wavelet_scales, in_dim, out_dim) * 0.01)
        self.gcn_bias = nn.Parameter(torch.zeros(out_dim))
        
        # Wavelet filter banks (learnable)
        self.wavelet_filters = nn.Parameter(torch.randn(wavelet_scales, in_dim) * 0.1)
        
        # Feature gating
        self.gate = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, adj):
        # x: (batch, input_len, n_features, in_dim)
        # adj: (n_features, n_features)
        batch_size, seq_len, n_features, in_dim = x.shape
        
        # Graph convolution: aggregate neighbors
        # x_graph: (batch * seq_len, n_features, in_dim)
        x_graph = x.reshape(batch_size * seq_len, n_features, in_dim)
        
        # Multi-scale wavelet filtering
        outputs = []
        for s in range(self.wavelet_scales):
            # Apply adjacency at scale s (power of adjacency)
            adj_s = torch.matrix_power(adj, 2**s) if s > 0 else adj
            # Normalize
            deg = adj_s.sum(dim=-1, keepdim=True).clamp(min=1)
            adj_norm = adj_s / deg
            
            # Graph conv: A * X * W
            h = torch.matmul(adj_norm, x_graph)  # (batch*seq_len, n_features, in_dim)
            h = torch.matmul(h, self.gcn_weights[s])  # (batch*seq_len, n_features, out_dim)
            
            # Wavelet modulation
            wavelet_mod = torch.sigmoid(self.wavelet_filters[s])  # (in_dim,)
            h = h * wavelet_mod.view(1, 1, -1)  # broadcast
            
            outputs.append(h)
        
        h = sum(outputs) + self.gcn_bias  # (batch*seq_len, n_features, out_dim)
        h = h.view(batch_size, seq_len, n_features, self.out_dim)  # (batch, seq_len, n_features, out_dim)
        
        # Gating
        gate = self.gate(x)  # (batch, seq_len, n_features, out_dim)
        h = h * gate  # (batch, seq_len, n_features, out_dim)
        
        h = self.dropout(h)
        h = self.layer_norm(h)
        
        # Residual
        if self.in_dim == self.out_dim:
            h = h + x  # (batch, seq_len, n_features, out_dim)
        
        return h
