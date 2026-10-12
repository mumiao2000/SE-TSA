import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierTemporalGraph(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_frequencies': [8, 32, 128],
            'graph_temp': [0.1, 0.3, 0.5],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_frequencies = self.config.get('n_frequencies', 32)
        self.graph_temp = self.config.get('graph_temp', 0.3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Learnable Fourier frequencies
        self.frequencies = nn.Parameter(torch.linspace(0, self.input_len // 2, self.n_frequencies))
        
        # Input encoding: project to hidden and compute Fourier features
        self.time_encoder = nn.Linear(self.input_len, self.hidden_dim)
        
        # Fourier feature projection
        self.fourier_proj = nn.Linear(self.n_frequencies * 2, self.hidden_dim)
        
        # Sparse graph components
        self.node_emb = nn.Parameter(torch.randn(self.n_features, self.hidden_dim) * 0.02)
        self.temp_param = nn.Parameter(torch.tensor(self.graph_temp))
        
        # Graph message passing (shallow)
        self.graph_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Frequency-domain prediction MLP
        self.freq_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.output_len)
        )
        
    def compute_fourier_features(self, x):
        # x: (batch, n_features, input_len)
        batch_size, n_features, seq_len = x.shape
        
        # Create time grid
        t = torch.arange(seq_len, device=x.device).float()  # (input_len,)
        
        # Compute Fourier basis: cos(2*pi*f*t/T) and sin(2*pi*f*t/T)
        # frequencies: (n_frequencies,)
        phase = 2 * 3.14159 * self.frequencies.unsqueeze(1) * t.unsqueeze(0) / seq_len  # (n_frequencies, input_len)
        
        cos_basis = torch.cos(phase)  # (n_frequencies, input_len)
        sin_basis = torch.sin(phase)  # (n_frequencies, input_len)
        
        # Apply to each feature
        x_expanded = x.unsqueeze(2)  # (batch, n_features, 1, input_len)
        cos_feat = torch.matmul(x_expanded, cos_basis.t().unsqueeze(0).unsqueeze(0))  # (batch, n_features, 1, n_frequencies)
        sin_feat = torch.matmul(x_expanded, sin_basis.t().unsqueeze(0).unsqueeze(0))  # (batch, n_features, 1, n_frequencies)
        
        cos_feat = cos_feat.squeeze(2)  # (batch, n_features, n_frequencies)
        sin_feat = sin_feat.squeeze(2)  # (batch, n_features, n_frequencies)
        
        fourier = torch.cat([cos_feat, sin_feat], dim=-1)  # (batch, n_features, n_frequencies*2)
        return fourier  # (batch, n_features, n_frequencies*2)
        
    def get_sparse_adjacency(self):
        adj = torch.matmul(self.node_emb, self.node_emb.t())  # (n_features, n_features)
        adj = adj / self.hidden_dim ** 0.5
        adj = F.softmax(adj / F.softplus(self.temp_param), dim=-1)
        # Sparsify
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
        
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Temporal encoding
        time_encoded = self.time_encoder(x_perm)  # (batch, n_features, hidden_dim)
        
        # Fourier features
        fourier = self.compute_fourier_features(x_perm)  # (batch, n_features, n_frequencies*2)
        fourier_encoded = self.fourier_proj(fourier)  # (batch, n_features, hidden_dim)
        
        # Combine temporal and frequency info
        combined = time_encoded + fourier_encoded  # (batch, n_features, hidden_dim)
        
        # Sparse graph convolution
        adj = self.get_sparse_adjacency()  # (n_features, n_features)
        
        # Message passing: aggregate neighbor features
        neighbor_agg = torch.matmul(adj.unsqueeze(0), combined)  # (batch, n_features, hidden_dim)
        
        # Combine self and neighbor
        graph_input = torch.cat([combined, neighbor_agg], dim=-1)  # (batch, n_features, hidden_dim*2)
        graph_out = self.graph_mlp(graph_input)  # (batch, n_features, hidden_dim)
        
        # Residual
        graph_out = combined + graph_out  # (batch, n_features, hidden_dim)
        
        # Predict
        out = self.freq_predictor(graph_out)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)
