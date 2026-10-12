import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveGraphNeuralForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_gnn_layers': [1, 3, 6],
            'n_temporal_layers': [1, 2, 4],
            'dropout': [0.0, 0.3],
            'k_neighbors': [3, 8, 16]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_gnn_layers = self.config.get('n_gnn_layers', 3)
        self.n_temporal_layers = self.config.get('n_temporal_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.k_neighbors = self.config.get('k_neighbors', 8)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Node embedding: project features to hidden dim
        self.node_embed = nn.Linear(1, self.hidden_dim)  # Each feature is a scalar per time step
        
        # Adaptive adjacency matrix parameters
        self.node_vec1 = nn.Parameter(torch.randn(self.n_features, 10))
        self.node_vec2 = nn.Parameter(torch.randn(10, self.n_features))
        
        # Graph convolution layers
        self.gnn_convs = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(self.n_gnn_layers):
            self.gnn_convs.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.gnn_norms.append(nn.LayerNorm(self.hidden_dim))
        
        # Temporal modeling after graph conv
        self.temporal_convs = nn.ModuleList()
        self.temporal_norms = nn.ModuleList()
        for _ in range(self.n_temporal_layers):
            self.temporal_convs.append(
                nn.Conv1d(self.hidden_dim * self.n_features, self.hidden_dim * self.n_features, 
                         kernel_size=3, padding=1)
            )
            self.temporal_norms.append(nn.LayerNorm(self.hidden_dim * self.n_features))
        
        # Output MLP: from input_len to output_len
        self.output_mlp = nn.Sequential(
            nn.Linear(self.input_len, self.input_len // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.input_len // 2, self.output_len)
        )
        
        # Final projection to scalar per feature
        self.final_proj = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Reshape for node embedding: (batch, input_len, n_features, 1)
        x = x.unsqueeze(-1)  # (batch, input_len, n_features, 1)
        
        # Embed each feature to hidden_dim: (batch, input_len, n_features, hidden_dim)
        x = self.node_embed(x)  # (batch, input_len, n_features, hidden_dim)
        
        # Compute adaptive adjacency matrix
        adj = F.softmax(F.relu(self.node_vec1 @ self.node_vec2), dim=1)  # (n_features, n_features)
        # Keep top-k neighbors for sparsity
        mask = torch.zeros_like(adj)
        topk_indices = torch.topk(adj, self.k_neighbors, dim=1)[1]  # (n_features, k_neighbors)
        mask.scatter_(1, topk_indices, 1)
        adj = adj * mask
        # Normalize
        row_sums = adj.sum(dim=1, keepdim=True) + 1e-8
        adj = adj / row_sums  # (n_features, n_features)
        
        # Apply graph convolutions per time step
        for i in range(self.n_gnn_layers):
            residual = x  # (batch, input_len, n_features, hidden_dim)
            # Graph conv: aggregate neighbor features
            # x: (batch, input_len, n_features, hidden_dim)
            # adj: (n_features, n_features)
            # Contract n_features dimension (n) with adj's first dim (n), output target feature (m)
            x = torch.einsum('btnh,nm->btmh', x, adj)  # (batch, input_len, n_features, hidden_dim)
            x = self.gnn_convs[i](x)  # (batch, input_len, n_features, hidden_dim)
            x = self.gnn_norms[i](x)  # (batch, input_len, n_features, hidden_dim)
            x = F.gelu(x)  # (batch, input_len, n_features, hidden_dim)
            x = x + residual  # (batch, input_len, n_features, hidden_dim)
        
        # Flatten features for temporal convolution
        x = x.reshape(batch_size, self.input_len, -1)  # (batch, input_len, n_features * hidden_dim)
        x = x.permute(0, 2, 1)  # (batch, n_features * hidden_dim, input_len)
        
        # Temporal convolutions
        for i in range(self.n_temporal_layers):
            residual = x  # (batch, n_features * hidden_dim, input_len)
            x = self.temporal_convs[i](x)  # (batch, n_features * hidden_dim, input_len)
            x = x.permute(0, 2, 1)  # (batch, input_len, n_features * hidden_dim)
            x = self.temporal_norms[i](x)  # (batch, input_len, n_features * hidden_dim)
            x = F.gelu(x)  # (batch, input_len, n_features * hidden_dim)
            x = x.permute(0, 2, 1)  # (batch, n_features * hidden_dim, input_len)
            x = x + residual  # (batch, n_features * hidden_dim, input_len)
        
        # Back to (batch, input_len, n_features * hidden_dim)
        x = x.permute(0, 2, 1)  # (batch, input_len, n_features * hidden_dim)
        
        # Reshape to separate features
        x = x.reshape(batch_size, self.input_len, self.n_features, self.hidden_dim)  # (batch, input_len, n_features, hidden_dim)
        
        # Project each feature back to scalar
        x = self.final_proj(x).squeeze(-1)  # (batch, input_len, n_features)
        
        # Temporal projection to output_len
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.output_mlp(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x

