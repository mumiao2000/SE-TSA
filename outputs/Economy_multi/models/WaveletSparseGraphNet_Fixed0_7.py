import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class WaveletSparseGraphNet_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_scales': [2, 4, 8],
            'graph_temp': [0.1, 0.3, 0.5],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_scales = self.config.get('n_scales', 4)
        self.graph_temp = self.config.get('graph_temp', 0.3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale dilated convolutions (wavelet-like)
        # Each conv outputs hidden_dim channels for each of n_features
        self.scale_convs = nn.ModuleList()
        for i in range(self.n_scales):
            dilation = 2 ** i
            kernel_size = min(3, self.input_len)
            padding = (kernel_size - 1) * dilation // 2
            self.scale_convs.append(
                nn.Conv1d(self.n_features, self.hidden_dim * self.n_features, kernel_size=kernel_size, 
                         dilation=dilation, padding=padding, bias=False, groups=1)
            )
        
        # Actually, simpler approach: each conv outputs (batch, hidden_dim, length)
        # We need to reorganize so each feature has its own hidden representation
        # Let me restructure: process each feature independently through scales
        
        # Re-approach: input is (batch, n_features, input_len)
        # For each scale, conv outputs (batch, hidden_dim, pooled_length)
        # But we need per-feature representations...
        
        # Correct approach: treat n_features as batch-like, process each feature's time series
        self.scale_convs = nn.ModuleList()
        for i in range(self.n_scales):
            dilation = 2 ** i
            kernel_size = min(3, self.input_len)
            padding = (kernel_size - 1) * dilation // 2
            # Conv1d: in_channels=1 (single feature at a time), out_channels=hidden_dim
            self.scale_convs.append(
                nn.Conv1d(1, self.hidden_dim, kernel_size=kernel_size, 
                         dilation=dilation, padding=padding, bias=False)
            )
        
        # Scale mixing: combine n_scales representations per feature
        self.scale_mix = nn.Sequential(
            nn.Linear(self.hidden_dim * self.n_scales, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Sparse adaptive graph adjacency
        self.node_emb = nn.Parameter(torch.randn(self.n_features, self.hidden_dim) * 0.02)
        self.graph_temp_param = nn.Parameter(torch.tensor(self.graph_temp))
        
        # Graph convolution
        self.graph_conv = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.graph_norm = nn.LayerNorm(self.hidden_dim)
        self.graph_act = nn.ReLU()
        
        # Temporal decoder: from hidden_dim to output_len per feature
        self.temporal_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.output_len)
        )
        
    def get_sparse_adjacency(self):
        # (n_features, n_features)
        adj = torch.matmul(self.node_emb, self.node_emb.t())  # (n_features, n_features)
        adj = adj / self.hidden_dim ** 0.5
        # Temperature-controlled sparse softmax
        adj = F.softmax(adj / F.softplus(self.graph_temp_param), dim=-1)
        # Sparsify: keep top-k per row
        k = max(3, self.n_features // 4)
        mask = torch.zeros_like(adj)
        topk_vals, topk_idx = torch.topk(adj, k, dim=-1)
        mask.scatter_(-1, topk_idx, 1.0)
        adj = adj * mask
        # Renormalize
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        return adj  # (n_features, n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Process each feature independently through multi-scale convolutions
        # Reshape to (batch * n_features, 1, input_len) for per-feature processing
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_reshaped = x_perm.reshape(batch_size * self.n_features, 1, self.input_len)  # (batch * n_features, 1, input_len)
        
        # Multi-scale decomposition: each scale produces (batch * n_features, hidden_dim, pooled_length)
        scale_features = []
        for conv in self.scale_convs:
            sf = conv(x_reshaped)  # (batch * n_features, hidden_dim, pooled_length)
            # Pool to fixed length of 1 (global average) to get single vector per feature per scale
            sf = F.adaptive_avg_pool1d(sf, 1)  # (batch * n_features, hidden_dim, 1)
            sf = sf.squeeze(-1)  # (batch * n_features, hidden_dim)
            scale_features.append(sf)
        
        # Concatenate scales: (batch * n_features, hidden_dim * n_scales)
        multi_scale = torch.cat(scale_features, dim=-1)  # (batch * n_features, hidden_dim * n_scales)
        
        # Mix scales
        multi_scale = self.scale_mix(multi_scale)  # (batch * n_features, hidden_dim)
        
        # Reshape back to (batch, n_features, hidden_dim)
        multi_scale = multi_scale.view(batch_size, self.n_features, self.hidden_dim)  # (batch, n_features, hidden_dim)
        
        # Sparse graph convolution
        adj = self.get_sparse_adjacency()  # (n_features, n_features)
        # Apply graph convolution: adj (n_features, n_features) @ multi_scale (batch, n_features, hidden_dim)
        # Need to transpose for batch matrix multiply
        multi_scale_t = multi_scale.permute(0, 2, 1)  # (batch, hidden_dim, n_features)
        graph_out = torch.matmul(multi_scale_t, adj.t())  # (batch, hidden_dim, n_features) @ (n_features, n_features) -> (batch, hidden_dim, n_features)
        graph_out = graph_out.permute(0, 2, 1)  # (batch, n_features, hidden_dim)
        
        graph_out = self.graph_conv(graph_out)  # (batch, n_features, hidden_dim)
        graph_out = self.graph_norm(graph_out)  # (batch, n_features, hidden_dim)
        graph_out = self.graph_act(graph_out)  # (batch, n_features, hidden_dim)
        
        # Combine with original features
        combined = multi_scale + graph_out  # (batch, n_features, hidden_dim)
        
        # Temporal decoding per feature: (batch, n_features, hidden_dim) -> (batch, n_features, output_len)
        out = self.temporal_decoder(combined)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)
