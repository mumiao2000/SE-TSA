import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class GraphNeuralFourier_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_graph_layers': [1, 3, 6],
            'graph_order': [2, 8, 32],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_graph_layers = self.config.get('n_graph_layers', 3)
        self.graph_order = self.config.get('graph_order', 8)
        self.dropout = self.config.get('dropout', 0.1)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Adaptive graph learning
        self.node_embed = nn.Parameter(torch.randn(self.n_features, self.graph_order))
        self.graph_proj = nn.Linear(self.graph_order, self.n_features)
        
        # Temporal encoding with Fourier features
        self.temporal_freqs = nn.Parameter(torch.randn(self.hidden_dim // 2) * 0.1)
        
        # Input projection: from input_len to hidden_dim
        self.input_proj = nn.Linear(self.input_len, self.hidden_dim)
        
        # Graph-Fourier layers
        self.graph_layers = nn.ModuleList()
        for _ in range(self.n_graph_layers):
            self.graph_layers.append(nn.ModuleDict({
                'spectral_filter': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                ),
                'temporal_filter': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                ),
                'norm1': nn.LayerNorm(self.hidden_dim),
                'norm2': nn.LayerNorm(self.hidden_dim)
            }))
        
        # Output
        self.output_proj = nn.Linear(self.hidden_dim, self.output_len)
        
    def get_adjacency(self):
        # Adaptive graph from node embeddings
        adj = torch.mm(self.node_embed, self.node_embed.t())  # (n_features, n_features)
        adj = F.softmax(adj, dim=-1)  # (n_features, n_features)
        return adj
    
    def spectral_conv(self, x, adj):
        # x: (batch, n_features, hidden_dim)
        # adj: (n_features, n_features)
        # Graph convolution via message passing
        out = torch.einsum('ij,bjf->bif', adj, x)  # (batch, n_features, hidden_dim)
        return out
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Create Fourier temporal encodings
        t = torch.arange(self.input_len, device=x.device).float()  # (input_len,)
        temporal_emb = torch.outer(t, self.temporal_freqs)  # (input_len, hidden_dim//2)
        temporal_emb = torch.cat([torch.sin(temporal_emb), torch.cos(temporal_emb)], dim=-1)  # (input_len, hidden_dim)
        
        # Expand to batch: (batch, input_len, hidden_dim)
        temporal_emb = temporal_emb.unsqueeze(0).expand(x.size(0), -1, -1)  # (batch, input_len, hidden_dim)
        
        # Permute input: (batch, n_features, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project each feature from input_len to hidden_dim
        h = self.input_proj(x_perm)  # (batch, n_features, hidden_dim)
        
        # Add temporal encoding: mean pool temporal_emb over input_len, expand to n_features
        temporal_mean = temporal_emb.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        temporal_expanded = temporal_mean.expand(-1, self.n_features, -1)  # (batch, n_features, hidden_dim)
        h = h + temporal_expanded  # (batch, n_features, hidden_dim)
        
        # Graph-Fourier layers
        adj = self.get_adjacency()  # (n_features, n_features)
        
        for layer in self.graph_layers:
            residual = h  # (batch, n_features, hidden_dim)
            
            # Spectral graph convolution
            h_graph = self.spectral_conv(h, adj)  # (batch, n_features, hidden_dim)
            h_graph = layer['spectral_filter'](h_graph)  # (batch, n_features, hidden_dim)
            h = layer['norm1'](residual + h_graph)  # (batch, n_features, hidden_dim)
            
            # Temporal filtering
            residual2 = h  # (batch, n_features, hidden_dim)
            h_temp = layer['temporal_filter'](h)  # (batch, n_features, hidden_dim)
            h = layer['norm2'](residual2 + h_temp)  # (batch, n_features, hidden_dim)
        
        # Output projection per feature
        out = self.output_proj(h)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

