import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelGraphNet(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'graph_dim': [32, 128, 512],
            'cheb_order': [2, 4, 8],
            'n_graph_layers': [2, 4, 8],
            'temporal_hidden': [64, 256, 1024]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.graph_dim = self.config.get('graph_dim', 128)
        self.cheb_order = self.config.get('cheb_order', 4)
        self.n_graph_layers = self.config.get('n_graph_layers', 4)
        self.temporal_hidden = self.config.get('temporal_hidden', 256)
        
        # RevIN
        self.revin = RevIN(self.n_features, affine=True)
        
        # Learnable feature embeddings (nodes in graph)
        self.node_embed = nn.Parameter(torch.randn(self.n_features, self.graph_dim))
        
        # Attention-based graph construction
        self.graph_attn = nn.Sequential(
            nn.Linear(self.graph_dim * 2, self.graph_dim),
            nn.Tanh(),
            nn.Linear(self.graph_dim, 1)
        )
        
        # Chebyshev polynomial coefficients per layer
        self.cheb_weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.cheb_order, self.graph_dim, self.graph_dim) * 0.1)
            for _ in range(self.n_graph_layers)
        ])
        self.cheb_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(self.graph_dim))
            for _ in range(self.n_graph_layers)
        ])
        
        # Feature projection to graph space
        self.feat_proj = nn.Linear(1, self.graph_dim)  # Each feature value projected
        
        # Temporal modeling after graph aggregation
        self.temporal_lstm = nn.LSTM(
            self.graph_dim * self.n_features,  # flattened graph state
            self.temporal_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Output generation
        self.output_gen = nn.Sequential(
            nn.Linear(self.temporal_hidden, self.temporal_hidden // 2),
            nn.GELU(),
            nn.Linear(self.temporal_hidden // 2, self.n_features)
        )
        
        # Multi-step decoder
        self.decoder = nn.LSTMCell(self.n_features, self.temporal_hidden)
        self.decoder_out = nn.Linear(self.temporal_hidden, self.n_features)

    def compute_adjacency(self):
        # Compute pairwise attention scores
        # node_embed: (n_features, graph_dim)
        n = self.n_features
        
        # Expand for all pairs
        src = self.node_embed.unsqueeze(1).expand(n, n, self.graph_dim)  # (n, n, graph_dim)
        dst = self.node_embed.unsqueeze(0).expand(n, n, self.graph_dim)  # (n, n, graph_dim)
        
        pairs = torch.cat([src, dst], dim=-1)  # (n, n, graph_dim * 2)
        scores = self.graph_attn(pairs).squeeze(-1)  # (n, n)
        
        # Sparse softmax (only keep top-k connections)
        adj = F.softmax(scores, dim=-1)
        
        # Symmetrize
        adj = (adj + adj.T) / 2
        
        # Add self-loops
        adj = adj + torch.eye(n, device=adj.device) * 0.5
        
        # Normalize
        degree = adj.sum(dim=-1, keepdim=True) + 1e-6
        adj_norm = adj / degree
        
        return adj_norm

    def chebyshev_conv(self, x, adj, weights, bias):
        # x: (batch, n_features, graph_dim)
        # adj: (n_features, n_features)
        # weights: (cheb_order, graph_dim, graph_dim)
        
        # Compute Laplacian: L = I - D^{-1/2} A D^{-1/2}
        # Simplified: use normalized adjacency directly
        
        # Chebyshev polynomials: T_0(x) = x, T_1(x) = 2x - 1, ...
        # For graph: T_k(L) x
        
        # Simplified implementation: use powers of adjacency
        out = torch.zeros_like(x)
        current = x  # T_0
        
        for k in range(self.cheb_order):
            # Apply adjacency propagation
            propagated = torch.bmm(adj.unsqueeze(0).expand(x.shape[0], -1, -1), current)  # (batch, n, graph_dim)
            
            # Apply learned weight
            out = out + propagated @ weights[k]  # (batch, n, graph_dim)
            
            # Update for next order
            current = propagated
        
        return out + bias

    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Compute graph adjacency (shared across batch)
        adj = self.compute_adjacency()  # (n_features, n_features)
        
        # Process each time step through graph
        graph_states = []
        for t in range(self.input_len):
            x_t = x[:, t, :]  # (batch, n_features)
            
            # Project to graph space: each feature becomes a node
            x_t_expanded = x_t.unsqueeze(-1)  # (batch, n_features, 1)
            node_feats = self.feat_proj(x_t_expanded)  # (batch, n_features, graph_dim)
            
            # Add node embeddings
            node_feats = node_feats + self.node_embed.unsqueeze(0)  # (batch, n_features, graph_dim)
            
            # Graph convolution layers
            h = node_feats
            for i in range(self.n_graph_layers):
                h_new = self.chebyshev_conv(h, adj, self.cheb_weights[i], self.cheb_biases[i])
                h = h + torch.tanh(h_new)  # residual + gating
            
            # Flatten graph state
            h_flat = h.view(batch_size, -1)  # (batch, n_features * graph_dim)
            graph_states.append(h_flat)
        
        graph_seq = torch.stack(graph_states, dim=1)  # (batch, input_len, n_features * graph_dim)
        
        # Temporal modeling
        lstm_out, (h_n, c_n) = self.temporal_lstm(graph_seq)  # lstm_out: (batch, input_len, temporal_hidden)
        
        # Decode to output sequence
        hidden = h_n[-1]  # (batch, temporal_hidden)
        cell = c_n[-1]  # (batch, temporal_hidden)
        
        # Initialize with last prediction
        last_pred = self.output_gen(hidden)  # (batch, n_features)
        
        outputs = []
        for _ in range(self.output_len):
            # LSTM step
            hidden, cell = self.decoder(last_pred, (hidden, cell))  # (batch, temporal_hidden)
            pred = self.decoder_out(hidden)  # (batch, n_features)
            outputs.append(pred)
            last_pred = pred
        
        out = torch.stack(outputs, dim=1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

