import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelGraphNet_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'graph_dim': [32, 128, 512],
            'n_graph_layers': [1, 3, 6],
            'temporal_kernel': [3, 7, 15],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.graph_dim = self.config.get('graph_dim', 128)
        self.n_graph_layers = self.config.get('n_graph_layers', 3)
        self.temporal_kernel = self.config.get('temporal_kernel', 7)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Learnable adjacency matrix (soft graph structure)
        self.node_embed = nn.Parameter(torch.randn(self.n_features, self.graph_dim))
        self.adj_temperature = nn.Parameter(torch.ones(1))
        
        # Feature projection to graph space
        self.feat_proj = nn.Linear(1, self.graph_dim)  # Each feature treated separately
        
        # Graph convolution layers
        self.graph_layers = nn.ModuleList()
        for _ in range(self.n_graph_layers):
            self.graph_layers.append(GraphConvBlock(self.graph_dim, self.dropout_rate))
        
        # Temporal modeling
        self.temporal_convs = nn.ModuleList()
        for _ in range(self.n_graph_layers):
            pad = (self.temporal_kernel - 1) // 2
            self.temporal_convs.append(
                nn.Conv1d(self.graph_dim, self.graph_dim, self.temporal_kernel, padding=pad, groups=self.graph_dim)
            )
        
        # Output MLP
        self.output_net = nn.Sequential(
            nn.Linear(self.graph_dim, self.graph_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.graph_dim // 2, 1)
        )
        
        # Direct linear predictor
        self.direct_pred = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save for direct prediction
        x_direct = x  # (batch, input_len, n_features)
        
        # Compute adaptive adjacency from node embeddings
        # node_embed: (n_features, graph_dim)
        sim = torch.mm(self.node_embed, self.node_embed.t())  # (n_features, n_features)
        adj = F.softmax(sim / self.adj_temperature.abs(), dim=-1)  # (n_features, n_features)
        
        # Treat each feature as a node, time as feature
        # x: (batch, input_len, n_features) -> want (batch, n_features, input_len, 1)
        x_nodes = x.permute(0, 2, 1).unsqueeze(-1)  # (batch, n_features, input_len, 1)
        
        # Project to graph dimension per time step
        batch, n_nodes, seq_len, _ = x_nodes.shape
        x_nodes = x_nodes.reshape(batch * n_nodes, seq_len, 1)  # (batch*n_features, input_len, 1)
        h = self.feat_proj(x_nodes)  # (batch*n_features, input_len, graph_dim)
        h = h.reshape(batch, n_nodes, seq_len, self.graph_dim)  # (batch, n_features, input_len, graph_dim)
        
        # Alternate graph and temporal convolutions
        for i, (g_layer, t_conv) in enumerate(zip(self.graph_layers, self.temporal_convs)):
            # Graph conv: aggregate across features
            # h: (batch, n_features, input_len, graph_dim)
            h_perm = h.permute(0, 2, 3, 1)  # (batch, input_len, graph_dim, n_features)
            h_perm = h_perm.contiguous()  # ensure contiguous for reshape
            h_flat = h_perm.reshape(batch * seq_len, self.graph_dim, n_nodes)  # (batch*input_len, graph_dim, n_features)
            
            # Apply adjacency: (batch*input_len, graph_dim, n_features) @ (n_features, n_features)
            adj_expanded = adj.unsqueeze(0).expand(batch * seq_len, -1, -1)  # (batch*input_len, n_features, n_features)
            h_graph = torch.bmm(h_flat, adj_expanded)  # (batch*input_len, graph_dim, n_features)
            h_graph = h_graph.view(batch, seq_len, self.graph_dim, n_nodes).permute(0, 3, 1, 2)  # (batch, n_features, input_len, graph_dim)
            
            h = g_layer(h, h_graph)  # (batch, n_features, input_len, graph_dim)
            
            # Temporal conv
            h_temp = h.reshape(batch * n_nodes, self.graph_dim, seq_len)  # (batch*n_features, graph_dim, input_len)
            h_temp = t_conv(h_temp)  # (batch*n_features, graph_dim, input_len)
            # Ensure output matches input length (padding should handle this, but slice to be safe)
            h_temp = h_temp[:, :, :seq_len]  # (batch*n_features, graph_dim, input_len)
            h_temp = h_temp.view(batch, n_nodes, self.graph_dim, seq_len).permute(0, 1, 3, 2)  # (batch, n_features, input_len, graph_dim)
            
            h = h + h_temp  # residual  # (batch, n_features, input_len, graph_dim)
        
        # Predict: use last temporal state, project to output
        h_last = h[:, :, -1, :]  # (batch, n_features, graph_dim)
        
        # Expand to output length with simple MLP
        h_out = h_last.unsqueeze(2).repeat(1, 1, self.output_len, 1)  # (batch, n_features, output_len, graph_dim)
        
        # Add temporal bias
        time_bias = torch.arange(self.output_len, device=x.device).float()
        time_bias = time_bias.view(1, 1, self.output_len, 1) / self.output_len  # (1, 1, output_len, 1)
        h_out = h_out * (1 + time_bias * 0.1)  # (batch, n_features, output_len, graph_dim)
        
        # Project to scalar per feature
        out = self.output_net(h_out).squeeze(-1)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Add direct prediction
        direct = self.direct_pred(x_direct.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, output_len, n_features)
        out = out + direct * 0.5  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class GraphConvBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.gate = nn.Linear(dim * 2, dim)
        
    def forward(self, x, x_graph):
        # x, x_graph: (batch, n_features, seq_len, dim)
        combined = torch.cat([x, x_graph], dim=-1)  # (batch, n_features, seq_len, dim*2)
        gate = torch.sigmoid(self.gate(combined))  # (batch, n_features, seq_len, dim)
        out = gate * x_graph + (1 - gate) * x  # (batch, n_features, seq_len, dim)
        out = self.norm(out)  # (batch, n_features, seq_len, dim)
        out = out + self.mlp(out)  # (batch, n_features, seq_len, dim)
        return out

