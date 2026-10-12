import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class GraphTemporalFusion(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'graph_dim': [32, 128, 512],
            'n_graph_layers': [1, 3, 6],
            'temporal_kernel': [3, 7, 15],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3],
            'graph_sparsity': [0.1, 0.5, 0.9]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.graph_dim = cfg.get('graph_dim', 128)
        self.n_graph_layers = cfg.get('n_graph_layers', 3)
        self.temporal_kernel = cfg.get('temporal_kernel', 7)
        self.n_heads = cfg.get('n_heads', 8)
        self.dropout = cfg.get('dropout', 0.1)
        self.graph_sparsity = cfg.get('graph_sparsity', 0.5)
        
        self.revin = RevIN(n_features, affine=True)
        
        # Feature embedding
        self.feature_embed = nn.Linear(1, self.graph_dim)
        
        # Adaptive graph generator: learns node embeddings that define graph
        self.node_embed = nn.Parameter(torch.randn(n_features, self.graph_dim))
        self.graph_gen = nn.Sequential(
            nn.Linear(self.graph_dim * 2, self.graph_dim),
            nn.ReLU(),
            nn.Linear(self.graph_dim, 1)
        )
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList()
        for _ in range(self.n_graph_layers):
            self.graph_convs.append(nn.ModuleDict({
                'gcn': nn.Linear(self.graph_dim, self.graph_dim),
                'temporal': nn.Conv1d(
                    self.graph_dim, self.graph_dim,
                    kernel_size=self.temporal_kernel,
                    padding=self.temporal_kernel // 2
                ),
                'norm': nn.LayerNorm(self.graph_dim),
                'dropout': nn.Dropout(self.dropout)
            }))
        
        # Multi-head self-attention for temporal modeling
        self.temporal_attn = nn.MultiheadAttention(
            self.graph_dim, self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Temporal fusion: combine graph and temporal features
        self.fusion = nn.Sequential(
            nn.Linear(self.graph_dim * 2, self.graph_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Output MLP: predict future
        # Flatten temporal and predict
        self.predictor = nn.Sequential(
            nn.Linear(input_len * self.graph_dim, self.graph_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.graph_dim * 4, self.graph_dim * 2),
            nn.ReLU(),
            nn.Linear(self.graph_dim * 2, output_len * n_features)
        )
        
        # Direct autoregressive head
        self.ar_head = nn.Linear(self.graph_dim, 1)
        
        # Temporal positional encoding
        self.temporal_pe = nn.Parameter(torch.randn(input_len, self.graph_dim))
        
    def _generate_graph(self, batch_size):
        # Generate adaptive adjacency matrix from node embeddings
        n = self.n_features
        
        # Compute pairwise similarities
        node_i = self.node_embed.unsqueeze(1).expand(-1, n, -1)  # (n, n, graph_dim)
        node_j = self.node_embed.unsqueeze(0).expand(n, -1, -1)  # (n, n, graph_dim)
        
        # Concatenate and predict edge weights
        node_pairs = torch.cat([node_i, node_j], dim=-1)  # (n, n, graph_dim*2)
        adj = self.graph_gen(node_pairs).squeeze(-1)  # (n, n)
        
        # Sparsify: keep top-k connections per node
        if self.training:
            # Gumbel-softmax for differentiable sparsification
            mask = torch.zeros_like(adj)
            k = int(n * (1 - self.graph_sparsity))
            _, top_indices = torch.topk(adj, k, dim=-1)
            mask.scatter_(-1, top_indices, 1.0)
            adj = adj * mask
        
        # Normalize (symmetric)
        adj = F.relu(adj)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / torch.sqrt(deg * deg.t())
        
        # Add self-loops
        adj_norm = adj_norm + torch.eye(n, device=adj.device)
        
        return adj_norm  # (n, n)
    
    def _graph_conv(self, x, adj, layer):
        # x: (batch, seq_len, n_features, graph_dim)
        # adj: (n_features, n_features)
        batch, seq_len, n, d = x.shape
        
        # Reshape for graph conv: (batch*seq_len, n, d)
        x_flat = x.view(batch * seq_len, n, d)
        
        # Graph convolution: aggregate neighbors
        x_gcn = torch.bmm(adj.unsqueeze(0).expand(batch * seq_len, -1, -1), x_flat)  # (batch*seq_len, n, d)
        x_gcn = layer['gcn'](x_gcn)  # (batch*seq_len, n, d)
        
        # Reshape back
        x_gcn = x_gcn.view(batch, seq_len, n, d)
        
        # Temporal convolution per node
        x_temp = x_gcn.permute(0, 2, 3, 1)  # (batch, n, d, seq_len)
        x_temp = x_temp.reshape(batch * n, d, seq_len)  # (batch*n, d, seq_len)
        x_temp = layer['temporal'](x_temp)  # (batch*n, d, seq_len)
        x_temp = x_temp.view(batch, n, d, seq_len).permute(0, 3, 1, 2)  # (batch, seq_len, n, d)
        
        # Residual and norm
        x_out = x_gcn + x_temp  # (batch, seq_len, n, d)
        x_out = layer['norm'](x_out)
        x_out = layer['dropout'](x_out)
        
        return x_out
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch, seq_len, n = x.shape
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Embed features: treat each feature as a node with value
        x = x.unsqueeze(-1)  # (batch, input_len, n_features, 1)
        x = self.feature_embed(x)  # (batch, input_len, n_features, graph_dim)
        
        # Add temporal PE
        x = x + self.temporal_pe.unsqueeze(0).unsqueeze(2)  # (batch, input_len, n_features, graph_dim)
        
        # Generate adaptive graph
        adj = self._generate_graph(batch)  # (n_features, n_features)
        
        # Apply graph-temporal layers
        for layer in self.graph_convs:
            x = x + self._graph_conv(x, adj, layer)  # (batch, input_len, n_features, graph_dim)
        
        # Temporal self-attention: attend across time for each feature
        # Reshape: (batch*n_features, seq_len, graph_dim)
        x_attn = x.permute(0, 2, 1, 3).reshape(batch * n, seq_len, self.graph_dim)
        x_attn_out, _ = self.temporal_attn(x_attn, x_attn, x_attn)  # (batch*n, seq_len, graph_dim)
        x_attn_out = x_attn_out.reshape(batch, n, seq_len, self.graph_dim).permute(0, 2, 1, 3)
        
        # Fusion
        x_fused = torch.cat([x, x_attn_out], dim=-1)  # (batch, seq_len, n_features, graph_dim*2)
        x_fused = self.fusion(x_fused)  # (batch, seq_len, n_features, graph_dim)
        
        # Predict: flatten and use MLP
        x_flat = x_fused.permute(0, 2, 1, 3).reshape(batch, n, seq_len * self.graph_dim)
        x_flat = x_flat.reshape(batch, n * seq_len * self.graph_dim)
        
        # Actually, let's do per-feature prediction then combine
        # Simpler: global pooling then predict
        x_pooled = x_fused.mean(dim=2)  # (batch, seq_len, graph_dim)
        x_pooled = x_pooled.reshape(batch, seq_len * self.graph_dim)
        
        out = self.predictor(x_pooled)  # (batch, output_len * n_features)
        out = out.view(batch, self.output_len, self.n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out
