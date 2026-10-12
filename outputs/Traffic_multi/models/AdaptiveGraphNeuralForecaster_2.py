import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveGraphNeuralForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'graph_dim': [16, 64, 256],
            'n_tcn_layers': [2, 4, 8],
            'kernel_size': [3, 7, 15],
            'dropout': [0.0, 0.3],
            'alpha': [0.1, 0.5, 0.9]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.graph_dim = self.config.get('graph_dim', 64)
        self.n_tcn_layers = self.config.get('n_tcn_layers', 4)
        self.kernel_size = self.config.get('kernel_size', 7)
        self.dropout = self.config.get('dropout', 0.1)
        self.alpha = self.config.get('alpha', 0.5)  # residual weight for graph
        
        # Instance normalization for input
        self.in_norm = nn.InstanceNorm1d(n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(n_features, self.hidden_dim)
        
        # Adaptive graph: learn node embeddings and compute similarity
        self.node_emb = nn.Parameter(torch.randn(n_features, self.graph_dim))
        
        # Graph convolution layer
        self.graph_conv = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # TCN blocks for temporal modeling
        self.tcn_blocks = nn.ModuleList()
        for i in range(self.n_tcn_layers):
            dilation = 2 ** i
            self.tcn_blocks.append(
                TCNBlock(self.hidden_dim, self.kernel_size, dilation, self.dropout)
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, n_features)
        )
        
        # Direct skip connection from input to output for residual learning
        self.skip_proj = nn.Linear(input_len, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        B, T, N = x.shape
        
        # Instance norm: transpose for norm, then back
        x_norm = self.in_norm(x.transpose(1, 2)).transpose(1, 2)  # (batch, input_len, n_features)
        residual_input = x_norm  # for skip connection
        
        # Input projection: (batch, input_len, hidden_dim)
        h = self.input_proj(x_norm)  # (batch, input_len, hidden_dim)
        
        # Compute adaptive adjacency matrix
        # node_emb: (n_features, graph_dim)
        adj = F.softmax(F.relu(torch.mm(self.node_emb, self.node_emb.t())), dim=1)  # (n_features, n_features)
        
        # Graph convolution: mix features at each time step
        # h: (batch, input_len, hidden_dim) -> treat hidden_dim as features for graph conv
        # Actually we need to apply graph conv on feature dimension
        # Reshape: (batch * input_len, n_features, hidden_dim/n_features)? No.
        
        # Alternative: apply graph convolution in feature space
        # h: (batch, input_len, hidden_dim). We want to use adj on n_features dimension
        # But hidden_dim != n_features. Let's use a different approach.
        
        # Project to feature space, apply graph conv, project back
        h_graph = h.transpose(1, 2)  # (batch, hidden_dim, input_len)
        
        # Actually, let's apply graph convolution differently
        # Use the adjacency to mix the original features, then project
        x_graph = torch.einsum('btn,nm->btm', x_norm, adj)  # (batch, input_len, n_features)
        x_graph = self.alpha * x_graph + (1 - self.alpha) * x_norm  # residual graph convolution
        
        # Re-project after graph conv
        h = self.input_proj(x_graph)  # (batch, input_len, hidden_dim)
        
        # TCN temporal modeling
        h = h.transpose(1, 2)  # (batch, hidden_dim, input_len) for conv1d
        for block in self.tcn_blocks:
            h = block(h)  # (batch, hidden_dim, input_len)
        h = h.transpose(1, 2)  # (batch, input_len, hidden_dim)
        
        # Use only last time step or project all to output
        # Global average pooling + MLP for prediction
        h_pool = h.mean(dim=1)  # (batch, hidden_dim)
        
        # Repeat for output length or use more sophisticated approach
        h_out = h_pool.unsqueeze(1).repeat(1, self.output_len, 1)  # (batch, output_len, hidden_dim)
        
        # Add temporal bias
        time_emb = torch.arange(self.output_len, device=h.device).float().unsqueeze(0).unsqueeze(-1)  # (1, output_len, 1)
        time_emb = time_emb / self.output_len
        h_out = h_out + time_emb * 0.01  # weak temporal encoding
        
        # Output projection
        out = self.output_proj(h_out)  # (batch, output_len, n_features)
        
        # Skip connection from normalized input
        skip = residual_input.transpose(1, 2)  # (batch, n_features, input_len)
        skip = self.skip_proj(skip)  # (batch, n_features, output_len)
        skip = skip.transpose(1, 2)  # (batch, output_len, n_features)
        
        out = out + skip
        
        # Denormalize: we used instance norm, so we need to denormalize
        # Instance norm doesn't store statistics, so we approximate with learned scale/shift
        # Actually, let's use batch statistics from input
        mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        
        # The model learns residual on normalized data, so we need to apply to original
        # Actually we already added skip from normalized input, need to adjust
        # Let's keep it simple: the output should be in original scale
        # Since we used instance norm and skip from normalized, we need to denormalize
        
        # Recompute: apply denorm to output
        out = out * std + mean  # approximate denormalization
        
        return out


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size-1)*dilation//2, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size-1)*dilation//2, dilation=dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        
        # Ensure output length matches input length
        self.padding = (kernel_size - 1) * dilation
        
    def forward(self, x):
        # x: (batch, channels, seq_len)
        residual = x
        
        # Causal padding
        x = F.pad(x, (self.padding, 0))
        x = self.conv1(x)
        x = x[:, :, :residual.shape[2]]  # trim to match
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = F.gelu(x)
        x = self.dropout(x)
        
        x = F.pad(x, (self.padding, 0))
        x = self.conv2(x)
        x = x[:, :, :residual.shape[2]]
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = F.gelu(x)
        x = self.dropout(x)
        
        return x + residual
