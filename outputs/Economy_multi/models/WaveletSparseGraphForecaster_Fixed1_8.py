import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class WaveletSparseGraphForecaster_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'wavelet_dim': [32, 128, 512],
            'n_scales': [2, 4, 8],
            'graph_dim': [64, 256, 1024],
            'n_graph_layers': [1, 3, 6],
            'temporal_kernel': [3, 7, 15],
            'dropout': [0.0, 0.3],
            'graph_sparsity': [0.1, 0.5, 0.9],
            'n_heads': [2, 8, 32],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.wavelet_dim = self.config.get('wavelet_dim', 128)
        self.n_scales = self.config.get('n_scales', 4)
        self.graph_dim = self.config.get('graph_dim', 256)
        self.n_graph_layers = self.config.get('n_graph_layers', 3)
        self.temporal_kernel = self.config.get('temporal_kernel', 7)
        self.dropout = self.config.get('dropout', 0.3)
        self.graph_sparsity = self.config.get('graph_sparsity', 0.5)
        self.n_heads = self.config.get('n_heads', 8)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Learnable wavelet decomposition filters (per-feature)
        self.scale_filters = nn.ModuleList()
        for s in range(self.n_scales):
            kernel_size = self.temporal_kernel * (2 ** s)
            # Use groups=1 for standard conv, output wavelet_dim total channels
            self.scale_filters.append(
                nn.Conv1d(self.n_features, self.wavelet_dim, kernel_size, groups=1, bias=True)
            )
        
        # Feature projection per scale: from wavelet_dim to graph_dim
        self.feature_proj = nn.Linear(self.wavelet_dim, self.graph_dim)
        
        # Adaptive sparse adjacency per scale
        self.adj_logits = nn.ParameterList([
            nn.Parameter(torch.randn(self.n_features, self.n_features) * 0.01)
            for _ in range(self.n_scales)
        ])
        
        # Graph convolution layers per scale
        self.graph_layers = nn.ModuleList()
        for _ in range(self.n_scales):
            scale_layers = nn.ModuleList()
            for l in range(self.n_graph_layers):
                scale_layers.append(nn.Sequential(
                    nn.Linear(self.graph_dim, self.graph_dim),
                    nn.LayerNorm(self.graph_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                ))
            self.graph_layers.append(scale_layers)
        
        # Cross-scale fusion
        self.scale_fusion = nn.MultiheadAttention(
            embed_dim=self.graph_dim,
            num_heads=self.n_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Temporal aggregation: linear projection from input_len to output_len
        self.temporal_agg = nn.Linear(self.input_len, self.output_len)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.graph_dim * self.n_scales, self.graph_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.graph_dim, self.n_features)
        )
        
    def get_sparse_adjacency(self, logits, temperature=0.1):
        # Gumbel-softmax style sparsity
        adj = F.softmax(logits / temperature, dim=-1)  # (n_features, n_features)
        # Hard sparsity: keep top-k connections
        k = max(1, int(self.n_features * (1 - self.graph_sparsity)))
        mask = torch.zeros_like(adj)
        topk_vals, topk_idx = torch.topk(adj, k, dim=-1)
        mask.scatter_(-1, topk_idx, 1.0)
        adj = adj * mask
        # Symmetrize and normalize
        adj = (adj + adj.transpose(-2, -1)) / 2
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj = adj / deg
        return adj  # (n_features, n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Prepare for conv: (batch, n_features, input_len)
        x_conv = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Multi-scale wavelet decomposition with zero padding (no size restrictions)
        scale_features = []  # List of (batch, input_len, n_features, graph_dim)
        for s in range(self.n_scales):
            kernel_size = self.temporal_kernel * (2 ** s)
            # Calculate padding to ensure output length = input_len
            # For Conv1d: output_len = input_len + 2*padding - kernel_size + 1
            # We want output_len = input_len, so pad_total = kernel_size - 1
            pad_total = kernel_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            
            # Use zero padding (mode='constant', value=0) - no restriction on padding size
            x_padded = F.pad(x_conv, (pad_left, pad_right), mode='constant', value=0)  # (batch, n_features, input_len + pad_total)
            
            # Apply learnable filter
            x_scale = self.scale_filters[s](x_padded)  # (batch, wavelet_dim, input_len)
            x_scale = x_scale.permute(0, 2, 1)  # (batch, input_len, wavelet_dim)
            
            # Expand to per-feature representation
            x_scale = x_scale.unsqueeze(2).expand(-1, -1, self.n_features, -1)  # (batch, input_len, n_features, wavelet_dim)
            x_scale = self.feature_proj(x_scale)  # (batch, input_len, n_features, graph_dim)
            scale_features.append(x_scale)
        
        # Apply graph convolution at each scale
        fused_scales = []
        for s in range(self.n_scales):
            x_scale = scale_features[s]  # (batch, input_len, n_features, graph_dim)
            adj = self.get_sparse_adjacency(self.adj_logits[s])  # (n_features, n_features)
            
            # Graph conv: for each time step, mix features according to adjacency
            batch_size, seq_len, n_feat, g_dim = x_scale.shape
            x_flat = x_scale.reshape(batch_size * seq_len, n_feat, g_dim)  # (batch*seq, n_features, graph_dim)
            
            for l in range(self.n_graph_layers):
                # Message passing: adj @ features
                adj_batch = adj.unsqueeze(0).expand(batch_size * seq_len, -1, -1)  # (batch*seq, n_features, n_features)
                x_msg = torch.bmm(adj_batch, x_flat)  # (batch*seq, n_features, graph_dim)
                x_flat = x_flat + x_msg  # residual connection before MLP
                x_flat = self.graph_layers[s][l](x_flat)  # (batch*seq, n_features, graph_dim)
            
            x_scale = x_flat.reshape(batch_size, seq_len, n_feat, g_dim)  # (batch, input_len, n_features, graph_dim)
            
            # Pool over features for this scale representation
            x_scale_pooled = x_scale.mean(dim=2)  # (batch, input_len, graph_dim)
            fused_scales.append(x_scale_pooled)
        
        # Stack scales: all should be (batch, input_len, graph_dim)
        fused = torch.stack(fused_scales, dim=2)  # (batch, input_len, n_scales, graph_dim)
        
        # Cross-scale attention
        batch_size, seq_len, n_scales, g_dim = fused.shape
        fused = fused.reshape(batch_size * seq_len, n_scales, g_dim)  # (batch*seq, n_scales, graph_dim)
        fused_att, _ = self.scale_fusion(fused, fused, fused)  # (batch*seq, n_scales, graph_dim)
        fused = fused + fused_att  # residual  # (batch*seq, n_scales, graph_dim)
        fused = fused.reshape(batch_size, seq_len, n_scales * g_dim)  # (batch, input_len, n_scales*graph_dim)
        
        # Temporal aggregation: project from input_len to output_len
        # fused: (batch, input_len, n_scales*graph_dim)
        fused = fused.permute(0, 2, 1)  # (batch, n_scales*graph_dim, input_len)
        out = self.temporal_agg(fused)  # (batch, n_scales*graph_dim, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_scales*graph_dim)
        
        # Final projection to n_features
        out = self.output_proj(out)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

