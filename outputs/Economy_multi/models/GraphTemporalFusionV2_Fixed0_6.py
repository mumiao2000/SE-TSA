import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class GraphTemporalFusionV2_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'graph_dim': [64, 256, 1024],
            'n_graph_layers': [1, 3, 6],
            'temporal_kernel': [3, 7, 15],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.15, 0.3],
            'graph_sparsity': [0.05, 0.3, 0.7],
            'expert_dim': [32, 128, 512],
            'mixture_temperature': [0.1, 0.5, 1.0],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.graph_dim = self.config.get('graph_dim', 256)
        self.n_graph_layers = self.config.get('n_graph_layers', 3)
        self.temporal_kernel = self.config.get('temporal_kernel', 7)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout = self.config.get('dropout', 0.15)
        self.graph_sparsity = self.config.get('graph_sparsity', 0.3)
        self.expert_dim = self.config.get('expert_dim', 128)
        self.mixture_temperature = self.config.get('mixture_temperature', 0.5)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.graph_dim)
        
        # Multi-scale temporal encoders for channel-independent expert
        self.temporal_scales = [3, 7, 15]
        n_scales = len(self.temporal_scales)
        # Properly distribute expert_dim across scales to ensure exact sum = expert_dim
        base_channels = self.expert_dim // n_scales
        remainder = self.expert_dim % n_scales
        self.scale_channels = []
        for i in range(n_scales):
            ch = base_channels + (1 if i < remainder else 0)
            self.scale_channels.append(ch)
        
        self.scale_encoders = nn.ModuleList()
        for i, k in enumerate(self.temporal_scales):
            pad = (k - 1) // 2
            self.scale_encoders.append(
                nn.Sequential(
                    nn.Conv1d(self.graph_dim, self.scale_channels[i], kernel_size=k, padding=pad),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                )
            )
        
        # Projection to ensure exact expert_dim after concatenation (safety)
        self.ci_proj = nn.Conv1d(self.expert_dim, self.expert_dim, kernel_size=1)
        
        # Adaptive graph structure for channel-dependent expert
        self.graph_adj = nn.Parameter(torch.randn(self.n_features, self.n_features) * 0.01)
        self.graph_threshold = nn.Parameter(torch.tensor(0.0))
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList()
        for _ in range(self.n_graph_layers):
            self.graph_convs.append(
                nn.Sequential(
                    nn.Linear(self.graph_dim, self.graph_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.graph_dim, self.graph_dim),
                )
            )
        
        # Temporal convolution for graph path
        self.graph_temporal = nn.Sequential(
            nn.Conv1d(self.graph_dim, self.expert_dim, kernel_size=self.temporal_kernel, padding=(self.temporal_kernel - 1) // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        
        # Mixture gating network - predicts blend weights per feature
        self.gate_encoder = nn.Sequential(
            nn.Linear(self.input_len * self.n_features, self.expert_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.expert_dim, self.n_features),
        )
        
        # Cross-feature attention for refinement
        self.cross_attn = nn.MultiheadAttention(self.expert_dim, self.n_heads, batch_first=True, dropout=self.dropout)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.expert_dim, self.expert_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.expert_dim // 2, self.n_features),
        )
        
        # Temporal decoder: project from input_len to output_len
        self.temporal_decoder = nn.Linear(self.input_len, self.output_len)
        
    def _get_sparse_graph(self):
        # Apply sparsemax-like sparsity with temperature
        adj = self.graph_adj / max(self.mixture_temperature, 0.01)
        # Soft thresholding for sparsity
        sparse_adj = F.softmax(adj, dim=-1)
        # Apply learned threshold
        mask = (sparse_adj > self.graph_sparsity).float()
        sparse_adj = sparse_adj * mask
        # Renormalize
        row_sums = sparse_adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sparse_adj = sparse_adj / row_sums
        return sparse_adj
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]  # (batch, input_len, n_features)
        
        # RevIN normalize
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store residual for skip connection
        x_residual = x  # (batch, input_len, n_features)
        
        # === Channel-Independent Expert: Multi-scale temporal convolutions ===
        # Project to graph_dim
        x_proj = self.input_proj(x)  # (batch, input_len, graph_dim)
        x_proj = x_proj.permute(0, 2, 1)  # (batch, graph_dim, input_len)
        
        # Multi-scale feature extraction
        scale_features = []
        for encoder in self.scale_encoders:
            sf = encoder(x_proj)  # (batch, scale_channels[i], input_len)
            scale_features.append(sf)
        
        # Concatenate multi-scale features
        ci_features = torch.cat(scale_features, dim=1)  # (batch, expert_dim, input_len)
        ci_features = self.ci_proj(ci_features)  # (batch, expert_dim, input_len)
        ci_features = ci_features.permute(0, 2, 1)  # (batch, input_len, expert_dim)
        
        # === Channel-Dependent Expert: Graph convolution with adaptive structure ===
        # Get sparse adjacency
        adj = self._get_sparse_graph()  # (n_features, n_features)
        
        # Apply graph convolutions: x_graph shape (batch, input_len, n_features)
        x_graph = x  # (batch, input_len, n_features)
        
        # Project to graph_dim space for graph processing
        x_gproj = self.input_proj(x)  # (batch, input_len, graph_dim)
        
        # Apply graph convolutions per timestep
        for gconv in self.graph_convs:
            residual_g = x_gproj  # (batch, input_len, graph_dim)
            x_gproj = gconv(x_gproj) + residual_g  # (batch, input_len, graph_dim)
        
        # Temporal convolution on graph features
        x_gproj = x_gproj.permute(0, 2, 1)  # (batch, graph_dim, input_len)
        cd_features = self.graph_temporal(x_gproj)  # (batch, expert_dim, input_len)
        cd_features = cd_features.permute(0, 2, 1)  # (batch, input_len, expert_dim)
        
        # === Mixture Gating ===
        # Global feature encoding for gating
        gate_input = x.reshape(batch_size, -1)  # (batch, input_len * n_features)
        gate_logits = self.gate_encoder(gate_input)  # (batch, n_features)
        gate_weights = torch.sigmoid(gate_logits / max(self.mixture_temperature, 0.01))  # (batch, n_features)
        
        # Expand gate weights for blending
        gate_weights_t = gate_weights.unsqueeze(1)  # (batch, 1, n_features)
        
        # === Blend experts and refine ===
        # Both ci_features and cd_features are (batch, input_len, expert_dim)
        # Simple addition blending
        blended = ci_features + cd_features  # (batch, input_len, expert_dim)
        
        # Self-attention for temporal dependencies
        attn_out, _ = self.cross_attn(blended, blended, blended)  # (batch, input_len, expert_dim)
        attn_out = attn_out + blended  # Residual  # (batch, input_len, expert_dim)
        
        # Output projection to n_features
        out_features = self.output_proj(attn_out)  # (batch, input_len, n_features)
        
        # Apply gating as residual modulation
        out_features = gate_weights_t * out_features + (1 - gate_weights_t) * x_residual  # (batch, input_len, n_features)
        
        # Temporal decoder: input_len -> output_len
        out_features = out_features.permute(0, 2, 1)  # (batch, n_features, input_len)
        out = self.temporal_decoder(out_features)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)

