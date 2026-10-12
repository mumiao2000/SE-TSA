import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SparseGraphWaveletForecaster_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'wavelet_dim': [64, 256, 1024],
            'n_graph_layers': [1, 2, 4],
            'graph_sparsity': [0.1, 0.5, 0.9],
            'temporal_kernel': [3, 7, 15],
            'dropout': [0.0, 0.3],
            'mix_temperature': [0.1, 0.5, 1.0],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.wavelet_dim = cfg.get('wavelet_dim', 256)
        self.n_graph_layers = cfg.get('n_graph_layers', 2)
        self.graph_sparsity = cfg.get('graph_sparsity', 0.5)
        self.temporal_kernel = cfg.get('temporal_kernel', 7)
        self.dropout_rate = cfg.get('dropout', 0.3)
        self.mix_temperature = cfg.get('mix_temperature', 0.5)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Haar wavelet decomposition: split into low-freq (avg) and high-freq (diff)
        self.scale_len = input_len // 2  # (batch, scale_len, n_features)
        
        # Feature projection for wavelet scales
        self.feature_proj = nn.Linear(self.n_features, self.wavelet_dim)
        
        # Adaptive graph adjacency (learnable with sparsity)
        self.graph_adj = nn.Parameter(torch.randn(self.n_features, self.n_features) * 0.01)
        self.register_buffer('adj_mask', torch.ones(self.n_features, self.n_features))
        
        # Temporal convolution for local patterns at each scale
        self.temporal_conv = nn.Conv1d(
            self.wavelet_dim, self.wavelet_dim,
            kernel_size=self.temporal_kernel,
            padding=self.temporal_kernel // 2,
            groups=1
        )
        
        # Sparse graph convolution layers
        self.graph_convs = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        for _ in range(self.n_graph_layers):
            self.graph_convs.append(nn.Linear(self.wavelet_dim, self.wavelet_dim))
            self.graph_norms.append(nn.LayerNorm(self.wavelet_dim))
        
        # Channel-independent path: separate predictor per feature
        self.ci_predictor = nn.Sequential(
            nn.Linear(self.scale_len * 2, self.wavelet_dim),  # 2 scales
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.wavelet_dim, self.output_len)
        )
        
        # Channel-dependent path: graph-enhanced predictor
        # Input: pooled over n_features, flattened (scale_len*2, wavelet_dim) -> scale_len*2 * wavelet_dim
        self.cd_predictor = nn.Sequential(
            nn.Linear(self.scale_len * 2 * self.wavelet_dim, self.wavelet_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.wavelet_dim, self.output_len * self.n_features)
        )
        
        # Channel mixer: learnable blend of CI and CD predictions
        # Input is concat of ci_stats and cd_stats, each is n_features*2, so total n_features*4
        self.mixer_gate = nn.Sequential(
            nn.Linear(self.n_features * 4, self.n_features),  # FIXED: was n_features*2
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.output_proj = nn.Linear(self.n_features, self.n_features)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _haar_decompose(self, x):
        # x: (batch, input_len, n_features)
        if x.size(1) % 2 == 1:
            x = x[:, :-1, :]  # (batch, input_len-1, n_features)
        x_reshaped = x.reshape(x.size(0), x.size(1) // 2, 2, x.size(2))  # (batch, scale_len, 2, n_features)
        low_freq = x_reshaped.mean(dim=2)  # (batch, scale_len, n_features)
        high_freq = x_reshaped[:, :, 1, :] - x_reshaped[:, :, 0, :]  # (batch, scale_len, n_features)
        return low_freq, high_freq  # each (batch, scale_len, n_features)
    
    def _apply_sparse_graph(self, x, adj):
        # x: (batch, seq_len, n_features, wavelet_dim)
        # adj: (n_features, n_features)
        batch, seq_len, n_feat, dim = x.shape
        
        # Apply graph convolution: X' = A * X * W
        x_flat = x.reshape(batch * seq_len, n_feat, dim)  # (batch*seq_len, n_features, wavelet_dim)
        
        # Message passing with sparse adjacency
        out = torch.bmm(adj.unsqueeze(0).expand(batch * seq_len, -1, -1), x_flat)  # (batch*seq_len, n_features, wavelet_dim)
        
        return out.reshape(batch, seq_len, n_feat, dim)  # (batch, seq_len, n_features, wavelet_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.size(0)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Haar wavelet decomposition
        low_freq, high_freq = self._haar_decompose(x)  # each (batch, scale_len, n_features)
        
        # Concatenate scales for processing: (batch, scale_len*2, n_features)
        wavelet_coeffs = torch.cat([low_freq, high_freq], dim=1)  # (batch, scale_len*2, n_features)
        
        # === Channel-Independent Path ===
        # Process each feature independently
        ci_input = wavelet_coeffs.permute(0, 2, 1)  # (batch, n_features, scale_len*2)
        ci_pred = self.ci_predictor(ci_input)  # (batch, n_features, output_len)
        ci_pred = ci_pred.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # === Channel-Dependent Path with Sparse Graph ===
        # Project features to wavelet_dim
        cd_input = self.feature_proj(wavelet_coeffs)  # (batch, scale_len*2, wavelet_dim)
        
        # Temporal convolution
        cd_temp = cd_input.permute(0, 2, 1)  # (batch, wavelet_dim, scale_len*2)
        cd_temp = self.temporal_conv(cd_temp)  # (batch, wavelet_dim, scale_len*2)
        cd_temp = cd_temp.permute(0, 2, 1)  # (batch, scale_len*2, wavelet_dim)
        cd_temp = F.relu(cd_temp)
        cd_temp = self.dropout(cd_temp)  # (batch, scale_len*2, wavelet_dim)
        
        # Expand for per-feature graph processing
        cd_graph = cd_temp.unsqueeze(2).expand(-1, -1, self.n_features, -1)  # (batch, scale_len*2, n_features, wavelet_dim)
        
        # Compute sparse adjacency with temperature-controlled sparsity
        adj_raw = self.graph_adj  # (n_features, n_features)
        # Apply sparsity mask via hard threshold
        adj_sorted = torch.sort(torch.abs(adj_raw.flatten()), descending=True)[0]
        threshold_idx = int(self.graph_sparsity * self.n_features * self.n_features)
        threshold = adj_sorted[min(threshold_idx, len(adj_sorted)-1)]
        adj_mask = (torch.abs(adj_raw) >= threshold).float()
        # Straight-through estimator for sparsity
        adj_sparse = adj_raw * adj_mask + (adj_mask - adj_mask.detach()) * adj_raw.detach()
        adj_sparse = F.softmax(adj_sparse / max(self.mix_temperature, 0.01), dim=-1)  # (n_features, n_features)
        
        # Graph convolution layers
        for i, (gconv, gnorm) in enumerate(zip(self.graph_convs, self.graph_norms)):
            residual = cd_graph  # (batch, scale_len*2, n_features, wavelet_dim)
            cd_graph = self._apply_sparse_graph(cd_graph, adj_sparse)  # (batch, scale_len*2, n_features, wavelet_dim)
            cd_graph = gconv(cd_graph.reshape(-1, self.wavelet_dim)).reshape(batch_size, -1, self.n_features, self.wavelet_dim)  # (batch, scale_len*2, n_features, wavelet_dim)
            cd_graph = gnorm(cd_graph.reshape(-1, self.wavelet_dim)).reshape(batch_size, -1, self.n_features, self.wavelet_dim)  # (batch, scale_len*2, n_features, wavelet_dim)
            cd_graph = F.relu(cd_graph)
            cd_graph = self.dropout(cd_graph)
            if i > 0:  # Residual after first layer
                cd_graph = cd_graph + residual
        
        # Pool over n_features dimension and flatten for predictor
        cd_pooled = cd_graph.mean(dim=2)  # (batch, scale_len*2, wavelet_dim)
        cd_seq_flat = cd_pooled.reshape(batch_size, -1)  # (batch, scale_len*2 * wavelet_dim)
        cd_pred = self.cd_predictor(cd_seq_flat)  # (batch, output_len * n_features)
        cd_pred = cd_pred.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # === Channel-Adaptive Mixing ===
        # Compute feature statistics for mixing
        ci_stats = torch.cat([ci_pred.mean(dim=1), ci_pred.std(dim=1)], dim=-1)  # (batch, n_features*2)
        cd_stats = torch.cat([cd_pred.mean(dim=1), cd_pred.std(dim=1)], dim=-1)  # (batch, n_features*2)
        
        # Learnable mixer gate
        mix_logits = self.mixer_gate(torch.cat([ci_stats, cd_stats], dim=-1))  # (batch, n_features)
        mix_weight = mix_logits.unsqueeze(1)  # (batch, 1, n_features)
        mix_weight = torch.sigmoid(mix_weight / max(self.mix_temperature, 0.01))  # (batch, 1, n_features)
        
        # Blend predictions
        output = mix_weight * ci_pred + (1 - mix_weight) * cd_pred  # (batch, output_len, n_features)
        
        # Final projection
        output = self.output_proj(output)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        output = self.revin(output, mode='denorm')  # (batch, output_len, n_features)
        
        return output  # (batch, output_len, n_features)

