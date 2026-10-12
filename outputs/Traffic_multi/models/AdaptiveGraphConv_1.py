import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveGraphConv(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_layers': [2, 4, 6],
            'graph_dim': [8, 16, 32],
            'dropout': [0.0, 0.3],
            'kernel_size': [3, 5, 7],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.graph_dim = self.config.get('graph_dim', 16)
        self.dropout = self.config.get('dropout', 0.1)
        self.kernel_size = self.config.get('kernel_size', 3)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Learnable node embeddings for graph structure
        self.node_embed1 = nn.Parameter(torch.randn(self.n_features, self.graph_dim))
        self.node_embed2 = nn.Parameter(torch.randn(self.n_features, self.graph_dim))
        
        # Input projection
        self.input_proj = nn.Conv2d(1, self.hidden_dim, kernel_size=1)
        
        # Temporal convolution layers with graph convolution
        self.temporal_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(self.n_layers):
            # Temporal convolution: (batch, hidden_dim, n_features, input_len)
            padding = (self.kernel_size - 1) // 2
            self.temporal_convs.append(nn.Conv2d(
                self.hidden_dim, self.hidden_dim, 
                kernel_size=(1, self.kernel_size), 
                padding=(0, padding)
            ))
            
            # Graph convolution: operates on (batch, hidden_dim, n_features, time)
            # We use a simple linear transformation per time step
            self.graph_convs.append(nn.Conv2d(
                self.hidden_dim, self.hidden_dim,
                kernel_size=(1, 1)
            ))
            
            self.bns.append(nn.BatchNorm2d(self.hidden_dim))
        
        # Output layers
        self.output_temporal = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        
        # Length projection
        self.length_proj = nn.Linear(self.input_len, self.output_len)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def get_adjacency(self):
        # Compute adaptive adjacency from node embeddings
        # A = softmax(ReLU(E1 @ E2^T))
        adj = torch.mm(self.node_embed1, self.node_embed2.t())  # (n_features, n_features)
        adj = F.relu(adj)
        adj = F.softmax(adj, dim=-1)
        return adj  # (n_features, n_features)
    
    def graph_conv(self, x, adj):
        # x: (batch, hidden_dim, n_features, time_len)
        # adj: (n_features, n_features)
        # Output: (batch, hidden_dim, n_features, time_len)
        batch_size, hidden_dim, n_features, time_len = x.shape
        
        # Reshape for matrix multiplication
        x_reshaped = x.permute(0, 1, 3, 2)  # (batch, hidden_dim, time_len, n_features)
        x_reshaped = x_reshaped.reshape(batch_size * hidden_dim * time_len, n_features)
        
        # Apply adjacency: (B*T*H, n_features) @ (n_features, n_features)
        out = torch.mm(x_reshaped, adj)  # (batch*hidden_dim*time_len, n_features)
        
        # Reshape back
        out = out.view(batch_size, hidden_dim, time_len, n_features)
        out = out.permute(0, 1, 3, 2)  # (batch, hidden_dim, n_features, time_len)
        return out
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Rearrange to (batch, 1, n_features, input_len) for 2D conv
        x = x.permute(0, 2, 1).unsqueeze(1)  # (batch, 1, n_features, input_len)
        
        # Input projection
        h = self.input_proj(x)  # (batch, hidden_dim, n_features, input_len)
        
        # Get adaptive adjacency
        adj = self.get_adjacency()  # (n_features, n_features)
        
        # Apply temporal and graph convolutions
        for i in range(self.n_layers):
            residual = h
            
            # Temporal convolution
            h_temp = self.temporal_convs[i](h)  # (batch, hidden_dim, n_features, input_len)
            
            # Graph convolution
            h_graph = self.graph_conv(h, adj)  # (batch, hidden_dim, n_features, input_len)
            h_graph = self.graph_convs[i](h_graph)  # (batch, hidden_dim, n_features, input_len)
            
            # Combine
            h = h_temp + h_graph  # (batch, hidden_dim, n_features, input_len)
            h = self.bns[i](h)
            h = F.relu(h)
            h = self.dropout_layer(h)
            
            # Residual connection
            h = h + residual
        
        # Output processing
        out = self.output_temporal(h)  # (batch, hidden_dim, n_features, input_len)
        out = F.relu(out)
        out = self.dropout_layer(out)
        out = self.output_proj(out)  # (batch, 1, n_features, input_len)
        out = out.squeeze(1)  # (batch, n_features, input_len)
        
        # Project to output length
        out = self.length_proj(out)  # (batch, n_features, output_len)
        
        # Transpose back
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

