import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class WaveletGraphFusionNet(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_scales': [2, 3, 4],
            'graph_dim': [32, 128, 256],
            'dropout': [0.0, 0.2],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_scales = self.config.get('n_scales', 3)
        self.graph_dim = self.config.get('graph_dim', 128)
        self.dropout = self.config.get('dropout', 0.2)
        
        # Instance normalization (not RevIN, to handle wavelet domain)
        self.instance_norm = nn.InstanceNorm1d(n_features, affine=True)
        
        # Wavelet filters (Haar-like, fixed for stability)
        self.register_buffer('low_pass', torch.tensor([0.5, 0.5]))
        self.register_buffer('high_pass', torch.tensor([0.5, -0.5]))
        
        # Per-scale graph convolutions
        self.scale_projs = nn.ModuleList([
            nn.Linear(input_len, self.hidden_dim) for _ in range(self.n_scales)
        ])
        
        # Adaptive adjacency per scale
        self.scale_embs = nn.ParameterList([
            nn.Parameter(torch.randn(n_features, self.graph_dim)) for _ in range(self.n_scales)
        ])
        for emb in self.scale_embs:
            nn.init.xavier_uniform_(emb)
        
        # Graph MLPs
        self.graph_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.graph_dim),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.graph_dim, self.hidden_dim)
            ) for _ in range(self.n_scales)
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * self.n_scales, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(self.dropout)
        )
        
        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ELU(),
            nn.Linear(self.hidden_dim // 2, output_len)
        )
        
    def wavelet_decompose(self, x, n_levels):
        # x: (batch, n_features, input_len)
        coeffs = []
        current = x
        for level in range(n_levels):
            # Ensure we can filter
            if current.shape[-1] < 2:
                break
            
            # Pad with zeros (safe for short sequences)
            pad_size = 1  # kernel size is 2
            current_padded = F.pad(current, (pad_size, 0), mode='constant', value=0)
            
            # Low pass (downsample)
            low = F.conv1d(
                current_padded, 
                self.low_pass.view(1, 1, 2).expand(current.size(1), 1, 2),
                groups=current.size(1),
                stride=2
            )  # (batch, n_features, len//2)
            
            # High pass (keep as detail)
            high = F.conv1d(
                current_padded,
                self.high_pass.view(1, 1, 2).expand(current.size(1), 1, 2),
                groups=current.size(1),
                stride=2
            )  # (batch, n_features, len//2)
            
            coeffs.append(high)
            current = low
        
        coeffs.append(current)  # final low-pass
        return coeffs
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        # Instance norm on feature dimension
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.instance_norm(x)  # (batch, n_features, input_len)
        
        # Wavelet decomposition
        scales = self.wavelet_decompose(x, self.n_scales - 1)
        # Pad or truncate to same length
        target_len = min(s.shape[-1] for s in scales)
        scales = [s[..., :target_len] for s in scales]
        
        # Pad to original length for projection
        if target_len < self.input_len:
            scales = [F.pad(s, (0, self.input_len - target_len), mode='constant', value=0) for s in scales]
        
        # Process each scale with graph convolution
        scale_outputs = []
        for i, scale in enumerate(scales[:self.n_scales]):
            # Project
            h = self.scale_projs[i](scale)  # (batch, n_features, hidden_dim)
            h = h.permute(0, 2, 1)  # (batch, hidden_dim, n_features)
            
            # Adaptive adjacency
            emb = self.scale_embs[i]  # (n_features, graph_dim)
            adj = torch.mm(emb, emb.t())  # (n_features, n_features)
            adj = F.softmax(adj / self.graph_dim**0.5, dim=-1)
            
            # Graph conv: (batch, hidden_dim, n_features) @ (n_features, n_features)
            h_graph = torch.einsum('bij,jk->bik', h, adj.t())  # (batch, hidden_dim, n_features)
            h_graph = h_graph.permute(0, 2, 1)  # (batch, n_features, hidden_dim)
            
            # MLP
            h_out = self.graph_mlps[i](h_graph)  # (batch, n_features, hidden_dim)
            h_out = h_out + h_graph  # residual
            
            # Global average pooling over features
            h_pooled = h_out.mean(dim=1)  # (batch, hidden_dim)
            scale_outputs.append(h_pooled)
        
        # Fuse scales
        h_fused = torch.cat(scale_outputs, dim=-1)  # (batch, hidden_dim * n_scales)
        h_fused = self.scale_fusion(h_fused)  # (batch, hidden_dim)
        
        # Expand to all features for output
        h_fused = h_fused.unsqueeze(1).expand(-1, self.n_features, -1)  # (batch, n_features, hidden_dim)
        
        # Output projection
        out = self.output_proj(h_fused)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        return out

