import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SparseGraphNeuralFilter(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_filters': [4, 8, 16],
            'graph_temp': [0.1, 0.3, 0.5],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_filters = self.config.get('n_filters', 8)
        self.graph_temp = self.config.get('graph_temp', 0.3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Neural filter bank: learnable FIR filters with different frequency responses
        self.filter_lengths = [3, 5, 7, 9][:self.n_filters]
        if len(self.filter_lengths) < self.n_filters:
            self.filter_lengths = self.filter_lengths * (self.n_filters // len(self.filter_lengths) + 1)
        self.filter_lengths = self.filter_lengths[:self.n_filters]
        
        self.filter_bank = nn.ModuleList()
        for fl in self.filter_lengths:
            # Ensure filter length doesn't exceed input
            actual_fl = min(fl, self.input_len)
            if actual_fl % 2 == 0:
                actual_fl -= 1  # Make odd for symmetric padding
            actual_fl = max(actual_fl, 1)
            self.filter_bank.append(
                nn.Conv1d(self.n_features, self.hidden_dim // self.n_filters, 
                         kernel_size=actual_fl, padding=actual_fl // 2, bias=False)
            )
        
        # Feature projection after filtering
        self.filter_combine = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Sparse adaptive adjacency
        self.node_emb = nn.Parameter(torch.randn(self.n_features, self.hidden_dim) * 0.02)
        self.temp_param = nn.Parameter(torch.tensor(self.graph_temp))
        
        # Single graph convolution layer (shallow as validated)
        self.graph_conv = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Temporal pooling and prediction
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output MLP
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.output_len)
        )
        
    def get_sparse_adjacency(self):
        adj = torch.matmul(self.node_emb, self.node_emb.t())  # (n_features, n_features)
        adj = adj / self.hidden_dim ** 0.5
        adj = F.softmax(adj / F.softplus(self.temp_param), dim=-1)
        # Sparsify with top-k
        k = max(3, self.n_features // 4)
        mask = torch.zeros_like(adj)
        topk_vals, topk_idx = torch.topk(adj, k, dim=-1)
        mask.scatter_(-1, topk_idx, 1.0)
        adj = adj * mask
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
        return adj  # (n_features, n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Apply neural filter bank
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        filtered = []
        for conv in self.filter_bank:
            f = conv(x_perm)  # (batch, hidden_dim//n_filters, input_len)
            # Pool to fixed length if needed
            if f.shape[-1] != self.input_len:
                f = F.interpolate(f, size=self.input_len, mode='linear', align_corners=False)
            filtered.append(f)
        
        # Concatenate filter responses
        filtered = torch.cat(filtered, dim=1)  # (batch, hidden_dim, input_len)
        filtered = filtered.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
        
        # Combine across time
        # Use last timestep and pooled representation
        last = filtered[:, -1, :]  # (batch, hidden_dim)
        pooled = filtered.mean(dim=1)  # (batch, hidden_dim)
        combined = last + pooled  # (batch, hidden_dim)
        
        # Expand to per-feature representation
        # We need (batch, n_features, hidden_dim) for graph conv
        # Use a feature-specific projection
        feature_repr = combined.unsqueeze(1).expand(-1, self.n_features, -1)  # (batch, n_features, hidden_dim)
        
        # Add feature-specific bias
        feature_bias = nn.Parameter(torch.randn(self.n_features, self.hidden_dim) * 0.01)
        feature_repr = feature_repr + feature_bias.unsqueeze(0)  # (batch, n_features, hidden_dim)
        
        # Sparse graph convolution
        adj = self.get_sparse_adjacency()  # (n_features, n_features)
        
        # Message passing
        messages = torch.matmul(adj.unsqueeze(0), feature_repr)  # (batch, n_features, hidden_dim)
        
        # Combine with self
        graph_input = feature_repr + messages  # (batch, n_features, hidden_dim)
        graph_out = self.graph_conv(graph_input)  # (batch, n_features, hidden_dim)
        
        # Predict per feature
        out = self.predictor(graph_out)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)
