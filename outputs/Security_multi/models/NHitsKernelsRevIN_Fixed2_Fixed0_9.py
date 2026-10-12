import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NHitsBlock(nn.Module):
    def __init__(self, input_len, hidden_dim, n_pool_scales, pool_out_dims, kernel_hidden, n_basis_kernels, output_len, dropout):
        super().__init__()
        self.input_len = input_len
        self.hidden_dim = hidden_dim
        self.n_pool_scales = n_pool_scales
        self.pool_out_dims = pool_out_dims
        self.kernel_hidden = kernel_hidden
        self.n_basis_kernels = n_basis_kernels
        self.output_len = output_len
        
        self.pool_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pool_out_dims[i], pool_out_dims[i]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for i in range(n_pool_scales)
        ])
        
        self.kernel_weights = nn.Linear(sum(pool_out_dims), n_basis_kernels)
        self.kernel_basis = nn.Parameter(torch.randn(n_basis_kernels, kernel_hidden))
        self.kernel_proj = nn.Linear(kernel_hidden, output_len)
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        self.gate_proj = nn.Linear(hidden_dim, 1)  # Will be used per-feature
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.forecast_to_hidden = nn.Linear(1, hidden_dim)  # per-feature forecast to hidden
        
    def _apply_pooling(self, x, pool_size, pool_proj, pool_out_dim):
        # x: (batch, hidden_dim, input_len)
        batch_size, hidden_dim, seq_len = x.shape
        
        target_len = (seq_len + pool_size - 1) // pool_size
        x_pooled = F.adaptive_avg_pool1d(x, target_len)  # (batch, hidden_dim, target_len)
        
        x_pooled = x_pooled.permute(0, 2, 1)  # (batch, target_len, hidden_dim)
        
        # We need to project from hidden_dim to pool_out_dim
        # Use adaptive pooling to reduce hidden_dim, then project
        # Actually, we need to pool across hidden_dim or use a projection
        # Let's use mean pooling over hidden_dim to get target_len features, then project
        x_pooled = x_pooled.mean(dim=-1, keepdim=True)  # (batch, target_len, 1)
        x_pooled = x_pooled.expand(-1, -1, pool_out_dim)  # (batch, target_len, pool_out_dim)
        x_proj = pool_proj(x_pooled)  # (batch, target_len, pool_out_dim)
        
        x_out = x_proj.mean(dim=1)  # (batch, pool_out_dim)
        return x_out
    
    def forward(self, x):
        # x: (batch, hidden_dim, input_len)
        batch_size = x.shape[0]
        
        # Multi-scale pooling
        pooled_features = []
        for i in range(self.n_pool_scales):
            pool_size = 2 ** i
            pooled = self._apply_pooling(x, pool_size, self.pool_projs[i], self.pool_out_dims[i])
            pooled_features.append(pooled)
        
        h = torch.cat(pooled_features, dim=-1)  # (batch, sum(pool_out_dims))
        
        # Basis kernel mixture
        kernel_weights = F.softmax(self.kernel_weights(h), dim=-1)  # (batch, n_basis_kernels)
        basis = torch.matmul(kernel_weights, self.kernel_basis)  # (batch, kernel_hidden)
        basis = F.relu(basis)
        forecast = self.kernel_proj(basis)  # (batch, output_len)
        
        # Gate for feature mixing
        h_expanded = h.unsqueeze(1).expand(-1, self.output_len, -1)  # (batch, output_len, hidden_dim)
        gate = self.gate(h_expanded)  # (batch, output_len, hidden_dim)
        gate = self.gate_proj(gate).squeeze(-1)  # (batch, output_len)
        gate = torch.sigmoid(gate)  # (batch, output_len)
        
        forecast = forecast * gate  # (batch, output_len)
        
        # Residual connection in hidden space
        forecast_hidden = forecast.unsqueeze(-1)  # (batch, output_len, 1)
        forecast_hidden = self.forecast_to_hidden(forecast_hidden)  # (batch, output_len, hidden_dim)
        forecast_hidden = forecast_hidden.permute(0, 2, 1)  # (batch, hidden_dim, output_len)
        
        if forecast_hidden.shape[-1] != x.shape[-1]:
            forecast_hidden = F.interpolate(forecast_hidden, size=x.shape[-1], mode='linear', align_corners=False)
        
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        x = x + self.dropout(forecast_hidden)  # (batch, hidden_dim, input_len)
        
        return x, forecast


class NHitsKernelsRevIN_Fixed2_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_blocks': [1, 4, 8],
            'n_pool_scales': [2, 3, 4],
            'kernel_hidden': [64, 256, 512],
            'dropout': [0.0, 0.3],
            'n_basis_kernels': [4, 8, 16],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_blocks = self.config.get('n_blocks', 4)
        self.n_pool_scales = self.config.get('n_pool_scales', 3)
        self.kernel_hidden = self.config.get('kernel_hidden', 256)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_basis_kernels = self.config.get('n_basis_kernels', 8)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Pre-compute safe pool output dimensions that sum to hidden_dim
        self.pool_out_dims = self._compute_pool_dims(self.input_len, self.n_pool_scales, self.hidden_dim)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # N-HiTS blocks
        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            self.blocks.append(NHitsBlock(
                self.input_len, self.hidden_dim, self.n_pool_scales,
                self.pool_out_dims, self.kernel_hidden, self.n_basis_kernels,
                self.output_len, self.dropout
            ))
        
        # Output projection: combine block forecasts
        self.forecast_proj = nn.Linear(self.output_len * self.n_blocks, self.output_len * self.n_features)
        
    def _compute_pool_dims(self, input_len, n_scales, hidden_dim):
        sizes = []
        for i in range(n_scales):
            pool_size = 2 ** i
            out_size = (input_len + pool_size - 1) // pool_size
            sizes.append(out_size)
        
        total_size = sum(sizes)
        dims = []
        allocated = 0
        for i, s in enumerate(sizes):
            if i < len(sizes) - 1:
                dim = max(1, int(hidden_dim * s / total_size))
                dims.append(dim)
                allocated += dim
            else:
                dims.append(hidden_dim - allocated)
        
        for i in range(len(dims)):
            if dims[i] <= 0:
                dims[i] = 1
        
        current_sum = sum(dims)
        if current_sum != hidden_dim:
            diff = hidden_dim - current_sum
            max_idx = dims.index(max(dims))
            dims[max_idx] += diff
            
        return dims
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        
        all_forecasts = []
        
        for block in self.blocks:
            x, forecast = block(x)  # forecast: (batch, output_len)
            all_forecasts.append(forecast)
        
        # Combine forecasts from all blocks
        combined = torch.cat(all_forecasts, dim=-1)  # (batch, output_len * n_blocks)
        out = self.forecast_proj(combined)  # (batch, output_len * n_features)
        out = out.view(-1, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
