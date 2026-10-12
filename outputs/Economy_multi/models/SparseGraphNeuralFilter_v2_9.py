import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SparseGraphNeuralFilter_v2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'temp': [0.1, 0.3, 0.5],
            'dropout': [0.0, 0.2],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.temp = self.config.get('temp', 0.3)
        self.dropout = self.config.get('dropout', 0.2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Node embeddings for adaptive adjacency
        self.node_emb = nn.Parameter(torch.randn(self.n_features, self.hidden_dim))
        nn.init.xavier_uniform_(self.node_emb)
        
        # Feature projection
        self.feat_proj = nn.Linear(input_len, self.hidden_dim)
        
        # Graph convolution - single shallow layer as validated
        self.graph_conv = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, output_len)
        
    def get_adjacency(self):
        # Adaptive adjacency with temperature-controlled sparsity
        # (n_features, hidden_dim) @ (hidden_dim, n_features) -> (n_features, n_features)
        adj = torch.mm(self.node_emb, self.node_emb.t())
        adj = adj / self.hidden_dim**0.5
        # Low temperature for sparsity
        adj = F.softmax(adj / self.temp, dim=-1)
        return adj
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose to (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project features: (batch, n_features, input_len) -> (batch, n_features, hidden_dim)
        h = self.feat_proj(x)  # (batch, n_features, hidden_dim)
        
        # Get adaptive adjacency
        adj = self.get_adjacency()  # (n_features, n_features)
        
        # Graph convolution: (batch, n_features, hidden_dim) @ (n_features, n_features).T
        # Actually: adj is (n_features, n_features), we want to aggregate neighbors
        # h_graph = adj @ h, where adj[i,j] is weight from j to i
        h_graph = torch.einsum('ij,bjf->bif', adj, h)  # (batch, n_features, hidden_dim)
        
        # Apply graph conv MLP
        h_out = self.graph_conv(h_graph)  # (batch, n_features, hidden_dim)
        
        # Residual connection
        h_out = h_out + h_graph  # (batch, n_features, hidden_dim)
        
        # Project to output length
        out = self.output_proj(h_out)  # (batch, n_features, output_len)
        
        # Transpose back: (batch, output_len, n_features)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

