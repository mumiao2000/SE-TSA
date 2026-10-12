import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveGraphNeuralForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3],
            'graph_dim': [16, 64, 256],
            'kernel_size': [3, 5, 7]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        self.graph_dim = self.config.get('graph_dim', 64)
        self.kernel_size = self.config.get('kernel_size', 3)
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Adaptive graph generator: learns node embeddings to construct graph
        self.node_embeddings = nn.Parameter(torch.randn(self.n_features, self.graph_dim))
        
        # Input projection
        self.input_proj = nn.Conv1d(self.n_features, self.hidden_dim, 1)
        
        # STGNN layers
        self.stgnn_layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.stgnn_layers.append(
                STGNNLayer(
                    in_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    kernel_size=self.kernel_size,
                    dropout=self.dropout
                )
            )
        
        # Output projection
        self.output_proj = nn.Conv1d(self.hidden_dim, self.n_features, 1)
        
        # Temporal decoder
        self.temporal_decoder = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        # Instance normalization
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.norm(x)  # (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Generate adaptive adjacency matrix
        # node_embeddings: (n_features, graph_dim)
        adj = F.softmax(
            F.relu(torch.mm(self.node_embeddings, self.node_embeddings.t())),
            dim=1
        )  # (n_features, n_features)
        
        # Input projection
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.input_proj(x)  # (batch, hidden_dim, input_len)
        
        # Apply STGNN layers
        for layer in self.stgnn_layers:
            x = layer(x, adj)  # (batch, hidden_dim, input_len)
        
        # Output projection
        x = self.output_proj(x)  # (batch, n_features, input_len)
        
        # Temporal decoder
        x = self.temporal_decoder(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        return x


class STGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dropout):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(
            in_dim, out_dim, (1, kernel_size), padding=(0, self.padding)
        )
        
        # Graph convolution (using adaptive adjacency)
        self.graph_conv = nn.Linear(in_dim, out_dim)
        
        # Output gate
        self.gate = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        
        # Residual connection
        self.residual = nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x, adj):
        # x: (batch, hidden_dim, input_len)
        # adj: (n_features, n_features) - but we operate on hidden_dim, need to handle carefully
        
        batch_size, hidden_dim, seq_len = x.shape
        
        # Reshape for 2D conv: (batch, hidden_dim, 1, seq_len)
        x_2d = x.unsqueeze(2)  # (batch, hidden_dim, 1, seq_len)
        
        # Temporal convolution
        h_temp = self.temporal_conv(x_2d)  # (batch, out_dim, 1, seq_len)
        h_temp = h_temp.squeeze(2)  # (batch, out_dim, seq_len)
        
        # Graph convolution: treat hidden_dim as nodes
        # We need to adapt: transpose to (batch, seq_len, hidden_dim)
        x_t = x.permute(0, 2, 1)  # (batch, seq_len, hidden_dim)
        h_graph = self.graph_conv(x_t)  # (batch, seq_len, out_dim)
        h_graph = h_graph.permute(0, 2, 1)  # (batch, out_dim, seq_len)
        
        # Combine
        h = h_temp + h_graph  # (batch, out_dim, seq_len)
        
        # Gate
        h_2d = h.unsqueeze(2)  # (batch, out_dim, 1, seq_len)
        gate = self.gate(h_2d).squeeze(2)  # (batch, out_dim, seq_len)
        h = h * gate  # (batch, out_dim, seq_len)
        
        # Residual
        res = self.residual(x_2d).squeeze(2)  # (batch, out_dim, seq_len)
        h = h + res  # (batch, out_dim, seq_len)
        
        h = self.dropout(h)
        
        # Layer norm on feature dim
        h = h.permute(0, 2, 1)  # (batch, seq_len, out_dim)
        h = self.layer_norm(h)  # (batch, seq_len, out_dim)
        h = h.permute(0, 2, 1)  # (batch, out_dim, seq_len)
        
        return h
