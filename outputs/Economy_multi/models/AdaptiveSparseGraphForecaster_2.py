import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveSparseGraphForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'graph_dim': [64, 256, 1024],
            'n_graph_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'sparsity_threshold': [0.01, 0.1, 0.5]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.graph_dim = self.config.get('graph_dim', 256)
        self.n_graph_layers = self.config.get('n_graph_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.sparsity_threshold = self.config.get('sparsity_threshold', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Feature embedding
        self.feature_embed = nn.Linear(1, self.graph_dim)  # (batch, n_features, graph_dim)
        
        # Learnable sparse adjacency matrix
        self.adj_logits = nn.Parameter(torch.randn(self.n_features, self.n_features) * 0.01)
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        for i in range(self.n_graph_layers):
            self.graph_convs.append(nn.Linear(self.graph_dim, self.graph_dim))
            self.graph_norms.append(nn.LayerNorm(self.graph_dim))
        
        self.graph_dropout = nn.Dropout(self.dropout)
        
        # Temporal aggregation and projection
        self.temporal_agg = nn.Linear(input_len, 1)  # (batch, n_features, 1)
        
        # Output projection: from graph representation to forecast
        self.output_proj = nn.Sequential(
            nn.Linear(self.graph_dim, self.graph_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.graph_dim // 2, output_len)  # (batch, n_features, output_len)
        )
        
    def get_sparse_adjacency(self):
        # Apply soft thresholding for adaptive sparsity
        adj = torch.sigmoid(self.adj_logits)  # (n_features, n_features)
        # Soft threshold: weights below threshold are suppressed
        adj = F.relu(adj - self.sparsity_threshold) / (1 - self.sparsity_threshold)
        # Add identity for self-loops
        adj = adj + torch.eye(self.n_features, device=adj.device)
        # Normalize
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
        adj = adj / degree
        return adj  # (n_features, n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose to (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Embed each feature's time series to graph dimension
        # Reshape for linear: (batch * n_features, input_len) -> no, we need per-timepoint
        # Actually: treat input_len as sequence, embed each value
        x = x.unsqueeze(-1)  # (batch, n_features, input_len, 1)
        
        # Process: for each time step, we have feature values
        # Reshape to process all at once
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1, 3)  # (batch, input_len, n_features, 1)
        x = x.reshape(batch_size * self.input_len, self.n_features, 1)  # (batch*input_len, n_features, 1)
        
        # Feature embedding
        h = self.feature_embed(x)  # (batch*input_len, n_features, graph_dim)
        
        # Get sparse adjacency
        adj = self.get_sparse_adjacency()  # (n_features, n_features)
        
        # Graph convolution with residual connections
        for i, (conv, norm) in enumerate(zip(self.graph_convs, self.graph_norms)):
            h_in = h
            # Message passing: adj @ h @ W
            h = torch.einsum('ij,bjf->bif', adj, h)  # (batch*input_len, n_features, graph_dim)
            h = conv(h)  # (batch*input_len, n_features, graph_dim)
            h = norm(h)
            h = F.relu(h)
            h = self.graph_dropout(h)
            if i > 0:  # Residual after first layer
                h = h + h_in  # (batch*input_len, n_features, graph_dim)
        
        # Reshape back
        h = h.reshape(batch_size, self.input_len, self.n_features, self.graph_dim)  # (batch, input_len, n_features, graph_dim)
        
        # Temporal aggregation: aggregate across time for each feature
        h = h.permute(0, 2, 3, 1)  # (batch, n_features, graph_dim, input_len)
        h = self.temporal_agg(h).squeeze(-1)  # (batch, n_features, graph_dim)
        
        # Output projection to forecast horizon
        out = self.output_proj(h)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

