import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SparseGraphNeuralForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'sparsity_temp': [0.1, 0.5, 1.0],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.sparsity_temp = self.config.get('sparsity_temp', 0.5)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(input_len, self.hidden_dim)  # (batch, n_features, hidden_dim)
        
        # Learnable sparse adjacency with Gumbel-softmax
        self.adj_logits = nn.Parameter(torch.zeros(n_features, n_features))
        self.adj_bias = nn.Parameter(torch.eye(n_features) * 2.0)  # Encourage self-connections
        
        # Graph convolution layers
        self.gc_layers = nn.ModuleList()
        self.gc_norms = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gc_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.gc_norms.append(nn.LayerNorm(self.hidden_dim))
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Temporal processing
        self.temporal_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, output_len)  # (batch, n_features, output_len)
        
    def get_sparse_adjacency(self):
        # Gumbel-softmax for sparse differentiable adjacency
        logits = self.adj_logits + self.adj_bias
        # Use hard sigmoid with temperature for sparsity
        adj = torch.sigmoid(logits / self.sparsity_temp)
        # Apply straight-through estimator for sparsity
        adj_hard = (adj > 0.5).float()
        adj = adj_hard - adj.detach() + adj  # Straight-through
        # Normalize adjacency
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj = adj / degree
        return adj  # (n_features, n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.input_proj(x)  # (batch, n_features, hidden_dim)
        
        adj = self.get_sparse_adjacency()  # (n_features, n_features)
        
        for i in range(self.n_layers):
            # Graph convolution: aggregate neighbor features
            x_agg = torch.einsum('nf,bfh->bnh', adj, x)  # (batch, n_features, hidden_dim)
            x = self.gc_layers[i](x_agg)  # (batch, n_features, hidden_dim)
            x = self.gc_norms[i](x)  # (batch, n_features, hidden_dim)
            x = F.gelu(x)  # (batch, n_features, hidden_dim)
            x = self.dropout(x)  # (batch, n_features, hidden_dim)
        
        x = self.temporal_mlp(x)  # (batch, n_features, hidden_dim)
        x = self.output_proj(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)

