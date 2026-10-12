import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelGraphNet_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'graph_dim': [64, 256, 1024],
            'n_graph_layers': [1, 3, 6],
            'kernel_size': [3, 7, 15],
            'dropout': [0.0, 0.3],
            'n_adaptive_neighbors': [5, 10, 20]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.graph_dim = self.config.get('graph_dim', 256)
        self.n_graph_layers = self.config.get('n_graph_layers', 3)
        self.kernel_size = self.config.get('kernel_size', 7)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_adaptive_neighbors = self.config.get('n_adaptive_neighbors', 10)
        
        # Instance normalization instead of RevIN for graph stability
        self.inst_norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Adaptive graph topology: node embeddings (n_features, graph_dim) for feature graph
        self.node_embed = nn.Parameter(torch.randn(self.n_features, self.graph_dim) * 0.02)
        
        # Temporal embedding
        self.temp_embed = nn.Parameter(torch.randn(self.input_len, self.graph_dim // 4) * 0.02)
        
        # First projection from n_features to graph_dim
        self.feature_proj = nn.Linear(self.n_features, self.graph_dim)
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList()
        self.temp_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(self.n_graph_layers):
            # Temporal convolution operates on graph_dim channels
            self.temp_convs.append(
                nn.Conv1d(self.graph_dim, self.graph_dim, self.kernel_size, padding=self.kernel_size//2, dilation=2**i)
            )
            
            # Graph convolution on feature space
            self.graph_convs.append(nn.Linear(self.graph_dim, self.graph_dim))
            self.norms.append(nn.LayerNorm(self.graph_dim))
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(self.graph_dim + self.graph_dim // 4, self.graph_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.graph_dim, self.output_len)
        )
        
        # Feature-wise adapter
        self.feature_gate = nn.Linear(self.n_features, self.n_features)
        
    def compute_adaptive_adj(self):
        # Cosine similarity based on node embeddings
        embed = self.node_embed  # (n_features, graph_dim)
        embed_norm = F.normalize(embed, p=2, dim=1)  # (n_features, graph_dim)
        adj = torch.mm(embed_norm, embed_norm.t())  # (n_features, n_features)
        # Keep top-k neighbors per node
        topk_vals, topk_idx = torch.topk(adj, k=min(self.n_adaptive_neighbors, adj.shape[0]), dim=1)
        # Create sparse attention-like adjacency
        adj_sparse = torch.zeros_like(adj)
        adj_sparse.scatter_(1, topk_idx, F.softmax(topk_vals, dim=1))
        return adj_sparse + torch.eye(adj.shape[0], device=adj.device) * 0.5  # Add self-loop
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        # Instance norm across features
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.inst_norm(x)  # (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Adaptive adjacency
        adj = self.compute_adaptive_adj()  # (n_features, n_features)
        
        h = x  # (batch, input_len, n_features)
        
        # Initial projection to graph_dim
        h = torch.einsum('btf,fg->btg', h, adj)  # (batch, input_len, n_features)
        h = self.feature_proj(h)  # (batch, input_len, graph_dim)
        
        for i in range(self.n_graph_layers):
            h_res = h  # (batch, input_len, graph_dim)
            
            # Temporal convolution: operate on graph_dim channels
            h_temp = h.permute(0, 2, 1)  # (batch, graph_dim, input_len)
            h_temp = self.temp_convs[i](h_temp)  # (batch, graph_dim, input_len)
            h_temp = h_temp.permute(0, 2, 1)  # (batch, input_len, graph_dim)
            
            # Apply graph conv and norm
            h_temp = self.graph_convs[i](h_temp)  # (batch, input_len, graph_dim)
            h_temp = self.norms[i](h_temp)  # (batch, input_len, graph_dim)
            h = F.gelu(h_temp)  # (batch, input_len, graph_dim)
            
            # Residual connection
            if h_res.shape == h.shape:
                h = h + h_res  # (batch, input_len, graph_dim)
        
        # Pool temporal and combine with temporal embedding
        h_temporal = h.mean(dim=1)  # (batch, graph_dim)
        t_embed = self.temp_embed.mean(dim=0).unsqueeze(0).expand(h.shape[0], -1)  # (batch, graph_dim//4)
        h_comb = torch.cat([h_temporal, t_embed], dim=-1)  # (batch, graph_dim + graph_dim//4)
        
        # Output projection
        out = self.output_mlp(h_comb)  # (batch, output_len)
        out = out.unsqueeze(-1).expand(-1, -1, self.n_features)  # (batch, output_len, n_features)
        
        # Feature-wise gating
        gate = torch.sigmoid(self.feature_gate(out.mean(dim=1)))  # (batch, n_features)
        out = out * gate.unsqueeze(1)  # (batch, output_len, n_features)
        
        return out

