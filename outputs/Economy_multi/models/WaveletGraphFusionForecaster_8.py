import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class WaveletGraphFusionForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'wavelet_dim': [32, 128, 512],
            'n_scales': [2, 4, 8],
            'graph_dim': [64, 256, 1024],
            'n_graph_layers': [1, 3, 6],
            'graph_sparsity': [0.1, 0.5, 0.9],
            'dropout': [0.0, 0.3],
            'temporal_kernel': [3, 7, 15],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.wavelet_dim = self.config.get('wavelet_dim', 128)
        self.n_scales = self.config.get('n_scales', 4)
        self.graph_dim = self.config.get('graph_dim', 256)
        self.n_graph_layers = self.config.get('n_graph_layers', 3)
        self.graph_sparsity = self.config.get('graph_sparsity', 0.5)
        self.dropout_rate = self.config.get('dropout', 0.3)
        self.temporal_kernel = self.config.get('temporal_kernel', 7)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Haar wavelet decomposition filters (learnable)
        self.register_buffer('haar_lo', torch.tensor([0.5, 0.5]).float())
        self.register_buffer('haar_hi', torch.tensor([0.5, -0.5]).float())
        
        # Ensure input_len is divisible by 2^n_scales for proper decomposition
        self.effective_input_len = input_len
        # Pad if necessary
        self.pad_len = 0
        temp_len = input_len
        for _ in range(self.n_scales):
            if temp_len % 2 == 1:
                self.pad_len += 1
                temp_len += 1
            temp_len = temp_len // 2
        
        # Wavelet projection layers for each scale
        self.wavelet_projs = nn.ModuleList()
        for i in range(self.n_scales + 1):
            self.wavelet_projs.append(nn.Linear(self.n_features, self.wavelet_dim))
        
        # Temporal convolution for each scale
        self.temporal_convs = nn.ModuleList()
        for i in range(self.n_scales + 1):
            padding = (self.temporal_kernel - 1) // 2
            self.temporal_convs.append(
                nn.Conv1d(self.wavelet_dim, self.wavelet_dim, kernel_size=self.temporal_kernel, padding=padding)
            )
        
        # Adaptive graph layers for cross-feature interaction at each scale
        self.graph_layers = nn.ModuleList()
        for i in range(self.n_scales + 1):
            scale_graph_layers = nn.ModuleList()
            for _ in range(self.n_graph_layers):
                scale_graph_layers.append(
                    AdaptiveSparseGraphLayer(self.wavelet_dim, self.graph_dim, self.graph_sparsity, self.dropout_rate)
                )
            self.graph_layers.append(scale_graph_layers)
        
        # Scale fusion: combine information across scales
        self.scale_fusion = nn.Linear((self.n_scales + 1) * self.wavelet_dim, self.wavelet_dim)
        
        # Feature-specific output projection
        self.feature_proj = nn.Sequential(
            nn.Linear(self.wavelet_dim, self.wavelet_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.wavelet_dim * 2, self.n_features)
        )
        
        # Temporal decoder: from compressed wavelet representation to output sequence
        # We need to reconstruct or directly predict the output
        self.decoder = nn.Linear(self.effective_input_len + self.pad_len, self.output_len)
        
    def haar_decompose(self, x, n_levels):
        # x: (batch, seq_len, n_features)
        coeffs = []
        current = x
        for level in range(n_levels):
            batch, seq_len, feat = current.shape
            if seq_len % 2 == 1:
                current = F.pad(current, (0, 0, 0, 1), mode='replicate')  # (batch, seq_len+1, n_features)
                seq_len += 1
            # Reshape for pairwise operation
            current = current.reshape(batch, seq_len // 2, 2, feat)  # (batch, seq_len//2, 2, n_features)
            lo = current[:, :, 0, :] + current[:, :, 1, :]  # (batch, seq_len//2, n_features)
            hi = current[:, :, 0, :] - current[:, :, 1, :]  # (batch, seq_len//2, n_features)
            lo = lo * 0.5
            hi = hi * 0.5
            coeffs.append(hi)
            current = lo
        coeffs.append(current)  # final approximation
        return coeffs  # list of [cD1, cD2, ..., cDn, cAn]
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Pad if necessary for wavelet decomposition
        if self.pad_len > 0:
            x = F.pad(x, (0, 0, 0, self.pad_len), mode='replicate')  # (batch, input_len+pad_len, n_features)
        
        # Multi-scale wavelet decomposition
        wavelet_coeffs = self.haar_decompose(x, self.n_scales)  # list of (batch, var_len, n_features)
        
        # Process each scale
        scale_outputs = []
        for i, (coeff, proj, conv, g_layers) in enumerate(zip(wavelet_coeffs, self.wavelet_projs, self.temporal_convs, self.graph_layers)):
            # Project to wavelet_dim
            h = proj(coeff)  # (batch, scale_len, wavelet_dim)
            
            # Temporal convolution
            h_conv = conv(h.permute(0, 2, 1))  # (batch, wavelet_dim, scale_len)
            h = h_conv.permute(0, 2, 1)  # (batch, scale_len, wavelet_dim)
            h = F.relu(h)
            
            # Graph layers for cross-feature interaction
            for g_layer in g_layers:
                h = g_layer(h)  # (batch, scale_len, wavelet_dim)
            
            # Global average pooling over time for this scale
            h_pooled = h.mean(dim=1)  # (batch, wavelet_dim)
            scale_outputs.append(h_pooled)
        
        # Fuse scales
        fused = torch.cat(scale_outputs, dim=-1)  # (batch, (n_scales+1) * wavelet_dim)
        fused = self.scale_fusion(fused)  # (batch, wavelet_dim)
        fused = F.relu(fused)
        
        # Expand to output sequence length
        fused_expanded = fused.unsqueeze(1).expand(-1, self.output_len, -1)  # (batch, output_len, wavelet_dim)
        
        # Project to features
        out = self.feature_proj(fused_expanded)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out


class AdaptiveSparseGraphLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, sparsity, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity
        self.dropout = nn.Dropout(dropout)
        
        # Node transformation
        self.node_transform = nn.Linear(in_dim, hidden_dim)
        
        # Adaptive adjacency: learnable node embeddings for graph structure
        self.node_emb = nn.Parameter(torch.randn(1, in_dim, hidden_dim // 2))
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, in_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(in_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        batch, seq_len, feat = x.shape
        
        # Node transformation
        h = self.node_transform(x)  # (batch, seq_len, hidden_dim)
        h = F.relu(h)
        
        # Compute adaptive adjacency from node embeddings
        # node_emb: (1, in_dim, hidden_dim//2) - treat features as nodes
        # We need to compute similarity across feature dimension
        # Transform x to feature-node space: (batch, seq_len, in_dim) -> features are nodes
        
        # Compute adjacency: use current features as node representations
        # x: (batch, seq_len, in_dim), treat in_dim as number of nodes
        x_feat = x.permute(0, 2, 1)  # (batch, in_dim, seq_len)
        
        # Simple adaptive adjacency based on feature correlations
        adj = torch.bmm(x_feat, x_feat.permute(0, 2, 1))  # (batch, in_dim, in_dim)
        adj = adj / (seq_len ** 0.5 + 1e-8)
        
        # Apply sparsity: keep top-k connections
        if self.sparsity < 1.0:
            k = max(1, int(self.sparsity * self.in_dim))
            mask = torch.zeros_like(adj)
            topk_vals, topk_idx = torch.topk(adj.abs(), k, dim=-1)
            mask.scatter_(-1, topk_idx, 1.0)
            adj = adj * mask
        
        # Normalize adjacency
        degree = adj.abs().sum(dim=-1, keepdim=True) + 1e-8
        adj_norm = adj / degree
        
        # Graph convolution: aggregate neighbor features
        # x: (batch, seq_len, in_dim), adj_norm: (batch, in_dim, in_dim)
        # We want to aggregate across feature dimension
        x_t = x.permute(0, 2, 1)  # (batch, in_dim, seq_len)
        out = torch.bmm(adj_norm, x_t)  # (batch, in_dim, seq_len)
        out = out.permute(0, 2, 1)  # (batch, seq_len, in_dim)
        
        # Transform and residual
        out = self.out_proj(h) + x  # skip connection with transformed features
        out = self.norm(out)
        out = self.dropout(out)
        
        return out
