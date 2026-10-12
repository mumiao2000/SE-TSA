import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SparseGraphNeuralForecasterV2_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'graph_dim': [64, 256, 1024],
            'n_graph_layers': [1, 4, 8],
            'temporal_kernel': [3, 7, 15],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3],
            'graph_sparsity': [0.1, 0.5, 0.9],
            'mixture_dim': [32, 128, 512],
            'temporal_temperature': [0.1, 1.0, 10.0],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.graph_dim = self.config.get('graph_dim', 256)
        self.n_graph_layers = self.config.get('n_graph_layers', 4)
        self.temporal_kernel = self.config.get('temporal_kernel', 7)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout = self.config.get('dropout', 0.3)
        self.graph_sparsity = self.config.get('graph_sparsity', 0.5)
        self.mixture_dim = self.config.get('mixture_dim', 128)
        self.temporal_temperature = self.config.get('temporal_temperature', 1.0)
        
        # Ensure graph_dim is divisible by n_heads
        self.graph_dim = (self.graph_dim // self.n_heads) * self.n_heads
        if self.graph_dim == 0:
            self.graph_dim = self.n_heads
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to graph dimension
        self.input_proj = nn.Linear(self.input_len, self.graph_dim)
        
        # Temporal gating: learned attention over input time steps
        self.temporal_query = nn.Parameter(torch.randn(1, 1, 1, self.graph_dim))
        self.temporal_key_proj = nn.Linear(1, self.graph_dim)
        self.temporal_value_proj = nn.Linear(1, self.graph_dim)
        self.temp_gate_proj = nn.Sequential(
            nn.Linear(self.graph_dim, self.mixture_dim),
            nn.GELU(),
            nn.Linear(self.mixture_dim, self.input_len)
        )
        
        # Learnable sparse adjacency with temperature-controlled sparsity
        self.adj_logits = nn.Parameter(torch.zeros(self.n_features, self.n_features))
        self.adj_temperature = nn.Parameter(torch.tensor(0.1))
        
        # Graph convolution layers with multi-head feature transformation
        self.graph_layers = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        for i in range(self.n_graph_layers):
            self.graph_layers.append(
                nn.ModuleDict({
                    'msg': nn.Linear(self.graph_dim, self.graph_dim),
                    'gate': nn.Linear(self.graph_dim * 2, self.graph_dim),
                    'ff_update': nn.Sequential(
                        nn.Linear(self.graph_dim, self.graph_dim * 2),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.graph_dim * 2, self.graph_dim)
                    ),
                    'feat_transform': nn.ModuleList([
                        nn.Linear(self.graph_dim, self.graph_dim // self.n_heads)
                        for _ in range(self.n_heads)
                    ])
                })
            )
            self.graph_norms.append(nn.LayerNorm(self.graph_dim))
        
        # Channel-independent pathway (separate MLP per feature)
        self.channel_independent = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.graph_dim, self.mixture_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.mixture_dim, self.graph_dim)
            )
            for _ in range(self.n_features)
        ])
        
        # Adaptive mixture weights: per-feature blending of CI vs CD
        self.mixture_weights = nn.Sequential(
            nn.Linear(self.graph_dim * 2, self.mixture_dim),
            nn.GELU(),
            nn.Linear(self.mixture_dim, 1),
            nn.Sigmoid()
        )
        
        # Temporal convolution for local patterns
        self.temporal_conv = nn.Conv1d(
            self.graph_dim, self.graph_dim, 
            kernel_size=self.temporal_kernel,
            padding=self.temporal_kernel // 2,
            groups=1
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.graph_dim, self.graph_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.graph_dim, self.output_len)
        )
        
        # Residual connection scaler
        self.residual_scale = nn.Parameter(torch.ones(1))
        
    def get_sparse_adjacency(self):
        # Gumbel-softmax inspired sparsity with learned temperature
        adj = F.softmax(self.adj_logits / F.softplus(self.adj_temperature), dim=-1)
        # Apply sparsity threshold: keep top-k connections per node
        k = max(1, int(self.n_features * (1 - self.graph_sparsity)))
        mask = torch.zeros_like(adj)
        topk_indices = torch.topk(adj, k, dim=-1)[1]
        mask.scatter_(-1, topk_indices, 1.0)
        adj = adj * mask
        # Add self-loops
        adj = adj + torch.eye(self.n_features, device=adj.device)
        # Normalize
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj = adj / deg
        return adj
        
    def temporal_gating(self, x):
        # x: (batch, n_features, input_len)
        batch = x.size(0)
        
        # Expand x to 4D for per-time-step projection: (batch, n_features, input_len, 1)
        x_4d = x.unsqueeze(-1)  # (batch, n_features, input_len, 1)
        
        # Project to graph dim per time step
        keys = self.temporal_key_proj(x_4d)  # (batch, n_features, input_len, graph_dim)
        values = self.temporal_value_proj(x_4d)  # (batch, n_features, input_len, graph_dim)
        
        # Multi-head temporal attention: query shared across features and time
        query = self.temporal_query.expand(batch, self.n_features, 1, self.graph_dim)  # (batch, n_features, 1, graph_dim)
        
        # Compute attention scores with temperature scaling
        scores = (query * keys).sum(dim=-1, keepdim=True)  # (batch, n_features, input_len, 1)
        scores = scores / (self.graph_dim ** 0.5)
        scores = scores / self.temporal_temperature  # Apply learned temperature
        weights = F.softmax(scores, dim=2)  # (batch, n_features, input_len, 1)
        
        # Apply gating: weighted combination of values
        gated = (weights * values).sum(dim=2)  # (batch, n_features, graph_dim)
        
        # Additional learned gate modulation
        gate_mod = torch.sigmoid(self.temp_gate_proj(gated.mean(dim=1)))  # (batch, input_len)
        
        return gated, gate_mod  # (batch, n_features, graph_dim), (batch, input_len)
        
    def graph_propagation(self, x, adj):
        # x: (batch, n_features, graph_dim)
        for layer, norm in zip(self.graph_layers, self.graph_norms):
            residual = x
            
            # Multi-head feature transformation
            head_outputs = []
            for head_transform in layer['feat_transform']:
                head_out = head_transform(x)  # (batch, n_features, graph_dim//n_heads)
                head_outputs.append(head_out)
            x_transformed = torch.cat(head_outputs, dim=-1)  # (batch, n_features, graph_dim)
            
            # Message passing with sparse adjacency
            messages = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), x_transformed)
            messages = layer['msg'](messages)
            
            # Gated update
            gate_input = torch.cat([x, messages], dim=-1)
            gate = torch.sigmoid(layer['gate'](gate_input))
            updated = gate * messages + (1 - gate) * x
            
            # Feedforward update
            x = norm(residual + layer['ff_update'](updated))
            
        return x
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.size(0)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for feature-first processing
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Temporal gating with learned attention
        x_graph, temporal_gate = self.temporal_gating(x)  # (batch, n_features, graph_dim), (batch, input_len)
        
        # Apply temporal gate modulation to input
        x = x * temporal_gate.unsqueeze(1)  # (batch, n_features, input_len)
        
        # Get sparse adjacency
        adj = self.get_sparse_adjacency()  # (n_features, n_features)
        
        # Graph propagation (channel-dependent pathway)
        x_cd = self.graph_propagation(x_graph, adj)  # (batch, n_features, graph_dim)
        
        # Channel-independent pathway
        x_ci = torch.stack([
            self.channel_independent[i](x_graph[:, i, :]) 
            for i in range(self.n_features)
        ], dim=1)  # (batch, n_features, graph_dim)
        
        # Adaptive mixture
        mix_input = torch.cat([x_cd, x_ci], dim=-1)  # (batch, n_features, graph_dim*2)
        mix_weights = self.mixture_weights(mix_input)  # (batch, n_features, 1)
        x_mixed = mix_weights * x_cd + (1 - mix_weights) * x_ci  # (batch, n_features, graph_dim)
        
        # Temporal convolution for local pattern refinement
        x_conv = self.temporal_conv(x_mixed.transpose(1, 2)).transpose(1, 2)  # (batch, n_features, graph_dim)
        x_mixed = x_mixed + self.residual_scale * x_conv  # Residual connection
        
        # Output projection
        out = self.output_proj(x_mixed)  # (batch, n_features, output_len)
        
        # Transpose back
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

