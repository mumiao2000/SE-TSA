import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveGraphNet(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'graph_dim': [32, 128, 512],
            'n_graph_layers': [1, 3, 6],
            'temporal_kernel': [3, 7, 15],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.graph_dim = self.config.get('graph_dim', 128)
        self.n_graph_layers = self.config.get('n_graph_layers', 3)
        self.temporal_kernel = self.config.get('temporal_kernel', 7)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Node embedding: project features to graph space
        self.node_embed = nn.Linear(1, self.graph_dim)  # Each feature becomes a node
        
        # Adaptive graph generator: learns edge weights from global features
        self.graph_gen = nn.Sequential(
            nn.Linear(n_features * 2, self.graph_dim),  # mean and std of features
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.graph_dim, n_features * n_features)  # full adjacency
        )
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList()
        for i in range(self.n_graph_layers):
            self.graph_convs.append(nn.ModuleDict({
                'self_loop': nn.Linear(self.graph_dim, self.graph_dim),
                'neighbor': nn.Linear(self.graph_dim, self.graph_dim),
                'gate': nn.Linear(self.graph_dim * 2, 1),
                'norm': nn.LayerNorm(self.graph_dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.graph_dim, self.graph_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.graph_dim * 2, self.graph_dim)
                ),
                'ffn_norm': nn.LayerNorm(self.graph_dim)
            }))
        
        # Temporal convolution for each node
        self.temporal_convs = nn.ModuleList()
        for i in range(self.n_graph_layers):
            self.temporal_convs.append(nn.Conv1d(
                self.graph_dim, self.graph_dim, 
                kernel_size=self.temporal_kernel,
                padding=self.temporal_kernel // 2,
                groups=self.graph_dim  # depthwise
            ))
        
        # Output: project graph nodes back to features, then temporal to output
        self.output_temporal = nn.Sequential(
            nn.Conv1d(self.graph_dim, self.graph_dim, self.temporal_kernel, padding=self.temporal_kernel // 2),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.AdaptiveAvgPool1d(output_len),
            nn.Conv1d(self.graph_dim, 1, 1)  # back to single value per feature
        )
        
        self.feature_proj = nn.Linear(self.graph_dim, 1)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Generate adaptive graph
        global_mean = x.mean(dim=1)  # (batch, n_features)
        global_std = x.std(dim=1) + 1e-5  # (batch, n_features)
        global_stats = torch.cat([global_mean, global_std], dim=-1)  # (batch, n_features*2)
        adj_logits = self.graph_gen(global_stats)  # (batch, n_features*n_features)
        adj = adj_logits.view(-1, self.n_features, self.n_features)  # (batch, n_features, n_features)
        adj = F.softmax(adj, dim=-1)  # row-normalized
        
        # Make symmetric and add self-loops
        adj = (adj + adj.transpose(-2, -1)) / 2  # (batch, n_features, n_features)
        eye = torch.eye(self.n_features, device=x.device).unsqueeze(0)  # (1, n_features, n_features)
        adj = adj + eye  # (batch, n_features, n_features)
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-5)  # renormalize
        
        # Prepare node features: (batch, input_len, n_features) -> nodes over time
        # We treat each time step as a graph snapshot
        b, t, f = x.shape
        
        # Embed each feature to graph dim: reshape to (batch*time, n_features, 1)
        x_reshaped = x.reshape(b * t, f, 1)  # (batch*time, n_features, 1)
        node_feats = self.node_embed(x_reshaped)  # (batch*time, n_features, graph_dim)
        
        # Apply graph convolutions
        for i, (gconv, tconv) in enumerate(zip(self.graph_convs, self.temporal_convs)):
            # Graph convolution: aggregate neighbors
            # node_feats: (batch*time, n_features, graph_dim)
            self_loop = gconv['self_loop'](node_feats)  # (batch*time, n_features, graph_dim)
            
            # Neighbor aggregation: (batch*time, n_features, graph_dim) @ (batch*time, n_features, n_features).T doesn't work directly
            # Need to use batch matrix multiply: for each sample in batch*time
            neighbor = torch.bmm(adj.repeat(t, 1, 1), node_feats)  # (batch*time, n_features, graph_dim)
            neighbor = gconv['neighbor'](neighbor)  # (batch*time, n_features, graph_dim)
            
            # Gating
            gate_input = torch.cat([self_loop, neighbor], dim=-1)  # (batch*time, n_features, graph_dim*2)
            gate = torch.sigmoid(gconv['gate'](gate_input))  # (batch*time, n_features, 1)
            
            updated = gate * self_loop + (1 - gate) * neighbor  # (batch*time, n_features, graph_dim)
            updated = gconv['norm'](updated + node_feats)  # residual + norm
            
            # FFN
            ffn_out = gconv['ffn'](updated)  # (batch*time, n_features, graph_dim)
            node_feats = gconv['ffn_norm'](ffn_out + updated)  # (batch*time, n_features, graph_dim)
        
        # Reshape back to temporal: (batch, input_len, n_features, graph_dim)
        node_feats = node_feats.view(b, t, f, self.graph_dim)  # (batch, input_len, n_features, graph_dim)
        
        # Process each feature's temporal sequence
        # (batch, n_features, graph_dim, input_len)
        node_feats = node_feats.permute(0, 2, 3, 1)  # (batch, n_features, graph_dim, input_len)
        
        # Apply temporal conv per feature
        out_feats = []
        for fi in range(f):
            feat_seq = node_feats[:, fi, :, :]  # (batch, graph_dim, input_len)
            # Temporal processing with adaptive pooling to output_len
            out_seq = F.adaptive_avg_pool1d(feat_seq, self.output_len)  # (batch, graph_dim, output_len)
            out_val = self.feature_proj(out_seq.permute(0, 2, 1))  # (batch, output_len, 1)
            out_feats.append(out_val)
        
        out = torch.cat(out_feats, dim=-1)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
