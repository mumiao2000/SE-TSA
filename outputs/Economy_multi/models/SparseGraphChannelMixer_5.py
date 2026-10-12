import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SparseGraphChannelMixer(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'temp': [0.1, 0.3, 0.5],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.temp = self.config.get('temp', 0.3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Adaptive adjacency matrix with sparsity control
        self.node_emb = nn.Parameter(torch.randn(self.n_features, self.hidden_dim // 4))
        self.adj_temp = nn.Parameter(torch.tensor(self.temp))
        
        # Input projection
        self.input_proj = nn.Linear(input_len, self.hidden_dim)
        
        # Single graph convolution layer (shallow is better per validated principle)
        self.graph_conv = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Channel-wise MLP mixer
        self.channel_mixer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, output_len)
        
    def get_adjacency(self):
        # (n_features, hidden_dim//4) @ (hidden_dim//4, n_features) -> (n_features, n_features)
        adj = torch.mm(self.node_emb, self.node_emb.t())  # (n_features, n_features)
        # Apply temperature-controlled sparsity via soft thresholding
        adj = F.softmax(adj / F.softplus(self.adj_temp), dim=-1)  # (n_features, n_features)
        # Sparsify: keep only top-k connections per node
        mask = torch.zeros_like(adj)
        k = max(3, self.n_features // 4)
        topk_idx = torch.topk(adj, k, dim=-1)[1]  # (n_features, k)
        mask.scatter_(-1, topk_idx, 1.0)  # (n_features, n_features)
        adj = adj * mask  # (n_features, n_features)
        # Renormalize
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)  # (n_features, n_features)
        return adj
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose to (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project to hidden dim
        h = self.input_proj(x)  # (batch, n_features, hidden_dim)
        
        # Graph convolution: aggregate from neighbors
        adj = self.get_adjacency()  # (n_features, n_features)
        h_agg = torch.einsum('ij,bjd->bid', adj, h)  # (batch, n_features, hidden_dim)
        h = self.graph_conv(h_agg) + h  # (batch, n_features, hidden_dim) - residual
        
        # Channel mixing
        h = h + self.channel_mixer(h)  # (batch, n_features, hidden_dim) - residual
        
        # Output projection
        out = self.output_proj(h)  # (batch, n_features, output_len)
        
        # Transpose back
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
