import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveGraphDecomp(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'graph_dim': [16, 64, 256],
            'n_layers': [1, 4, 8],
            'alpha': [0.1, 0.5, 0.9],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.graph_dim = self.config.get('graph_dim', 64)
        self.n_layers = self.config.get('n_layers', 4)
        self.alpha = self.config.get('alpha', 0.5)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Adaptive graph adjacency (learnable)
        self.node_embed = nn.Parameter(torch.randn(n_features, self.graph_dim))
        self.graph_bias = nn.Parameter(torch.zeros(n_features, n_features))
        
        # Trend-seasonality decomposition (shared across features initially)
        self.trend_extract = nn.Sequential(
            nn.Linear(input_len, self.graph_dim),
            nn.ReLU(),
            nn.Linear(self.graph_dim, input_len)
        )
        
        # Graph convolution layers for trend and seasonality
        self.trend_gnns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_len if i == 0 else self.graph_dim, self.graph_dim),
                nn.ReLU(),
                nn.LayerNorm(self.graph_dim),
                nn.Dropout(0.1)
            ) for i in range(self.n_layers)
        ])
        
        self.season_gnns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_len if i == 0 else self.graph_dim, self.graph_dim),
                nn.ReLU(),
                nn.LayerNorm(self.graph_dim),
                nn.Dropout(0.1)
            ) for i in range(self.n_layers)
        ])
        
        # Output projections
        self.trend_out = nn.Linear(self.graph_dim, output_len)
        self.season_out = nn.Linear(self.graph_dim, output_len)
        
        # Feature-wise residual
        self.residual = nn.Linear(input_len, output_len)
        
    def get_adjacency(self):
        # Compute adaptive adjacency from node embeddings
        adj = F.softmax(F.relu(torch.matmul(self.node_embed, self.node_embed.t()) + self.graph_bias), dim=-1)
        return adj  # (n_features, n_features)
    
    def graph_conv(self, x, adj, layers):
        # x: (batch, n_features, seq_len)
        # adj: (n_features, n_features)
        h = x  # (batch, n_features, seq_len)
        for i, layer in enumerate(layers):
            # Message passing: aggregate neighbor features
            h_in = h  # (batch, n_features, dim)
            # Apply adjacency: (batch, n_features, dim) = adj @ (batch, n_features, dim)
            h_msg = torch.einsum('nf,bfd->bnd', adj, h_in)  # (batch, n_features, dim)
            # Combine with self
            h = self.alpha * h_in + (1 - self.alpha) * h_msg  # (batch, n_features, dim)
            # MLP transformation
            h = layer(h)  # (batch, n_features, graph_dim)
        return h  # (batch, n_features, graph_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Trend extraction (shared MLP per feature)
        trend = self.trend_extract(x_perm)  # (batch, n_features, input_len)
        season = x_perm - trend  # (batch, n_features, input_len)
        
        # Get adaptive adjacency
        adj = self.get_adjacency()  # (n_features, n_features)
        
        # Graph convolutions
        trend_h = self.graph_conv(trend, adj, self.trend_gnns)  # (batch, n_features, graph_dim)
        season_h = self.graph_conv(season, adj, self.season_gnns)  # (batch, n_features, graph_dim)
        
        # Output projections
        trend_pred = self.trend_out(trend_h)  # (batch, n_features, output_len)
        season_pred = self.season_out(season_h)  # (batch, n_features, output_len)
        
        out = trend_pred + season_pred  # (batch, n_features, output_len)
        
        # Residual
        res = self.residual(x_perm)  # (batch, n_features, output_len)
        out = out + res  # (batch, n_features, output_len)
        
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

