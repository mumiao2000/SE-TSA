import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttentionMLP_Improved_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3],
            'n_scales': [2, 3, 4],
            'gate_dim': [16, 64, 256],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.n_scales = self.config.get('n_scales', 3)
        self.gate_dim = self.config.get('gate_dim', 64)
        
        assert self.hidden_dim % self.n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.head_dim = self.hidden_dim // self.n_heads
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Multi-scale temporal decomposition: different kernel sizes for different periodicities
        # Dominant period is ~12, so we use scales that capture sub-period, period, and super-period
        self.scale_kernels = [3, 7, 15][:self.n_scales]
        self.decompose_convs = nn.ModuleList()
        for k in self.scale_kernels:
            # Causal padding to maintain sequence length
            pad = (k - 1) // 2
            self.decompose_convs.append(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=k, padding=pad, groups=self.hidden_dim)
            )
        
        # Learnable mixing weights for decomposition components
        self.mix_weights = nn.Parameter(torch.ones(self.n_scales))
        
        # Residual trend-seasonality separation
        self.trend_filter = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5, padding=2, groups=self.hidden_dim)
        self.seasonal_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Cross-channel attention with feature-wise gating
        self.channel_query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.channel_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.channel_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.channel_out = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Feature-wise gating: adapts to heterogeneous feature scales
        self.gate_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.gate_dim),
            nn.GELU(),
            nn.Linear(self.gate_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        
        # Channel-adaptive MLP with feature-specific transformations
        self.channel_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Dropout(self.dropout_rate),
        )
        
        # Layer norms for stability
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)
        
        # Output projection: map back to features
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        
        # Direct skip from input for gradient flow
        self.skip_proj = nn.Linear(self.input_len, self.output_len)
        
        # Temporal refinement conv (replaces broken F.conv1d)
        self.temporal_refine = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1, groups=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Store raw input for skip connection
        x_raw = x  # (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection
        h = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Multi-scale temporal decomposition
        h_perm = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        
        decomposed = []
        for conv in self.decompose_convs:
            comp = conv(h_perm)  # (batch, hidden_dim, input_len)
            decomposed.append(comp)
        
        # Weighted combination of scales
        weights = F.softmax(self.mix_weights, dim=0)  # (n_scales,)
        h_decomp = sum(w * comp for w, comp in zip(weights, decomposed))  # (batch, hidden_dim, input_len)
        h_decomp = h_decomp.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
        
        # Residual trend-seasonality separation
        trend = self.trend_filter(h_perm)  # (batch, hidden_dim, input_len)
        trend = trend.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
        seasonal = h - trend  # (batch, input_len, hidden_dim)
        seasonal = self.seasonal_proj(seasonal)  # (batch, input_len, hidden_dim)
        
        h = self.norm1(trend + seasonal + h_decomp)  # (batch, input_len, hidden_dim)
        
        # Cross-channel attention: attend across feature channels at each time step
        # Reshape for multi-head attention: (batch, input_len, n_heads, head_dim)
        Q = self.channel_query(h).view(batch_size, self.input_len, self.n_heads, self.head_dim)  # (batch, input_len, n_heads, head_dim)
        K = self.channel_key(h).view(batch_size, self.input_len, self.n_heads, self.head_dim)  # (batch, input_len, n_heads, head_dim)
        V = self.channel_value(h).view(batch_size, self.input_len, self.n_heads, self.head_dim)  # (batch, input_len, n_heads, head_dim)
        
        # Transpose for attention: (batch, n_heads, input_len, head_dim)
        Q = Q.permute(0, 2, 1, 3)  # (batch, n_heads, input_len, head_dim)
        K = K.permute(0, 2, 1, 3)  # (batch, n_heads, input_len, head_dim)
        V = V.permute(0, 2, 1, 3)  # (batch, n_heads, input_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, n_heads, input_len, input_len)
        attn = F.softmax(scores, dim=-1)  # (batch, n_heads, input_len, input_len)
        
        out = torch.matmul(attn, V)  # (batch, n_heads, input_len, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()  # (batch, input_len, n_heads, head_dim)
        out = out.view(batch_size, self.input_len, self.hidden_dim)  # (batch, input_len, hidden_dim)
        
        out = self.channel_out(out)  # (batch, input_len, hidden_dim)
        h = self.norm2(h + out)  # (batch, input_len, hidden_dim)
        
        # Feature-wise gating
        gate = self.gate_proj(h)  # (batch, input_len, hidden_dim)
        h = h * gate  # (batch, input_len, hidden_dim)
        
        # Channel-adaptive MLP
        mlp_out = self.channel_mlp(h)  # (batch, input_len, hidden_dim)
        h = self.norm3(h + mlp_out)  # (batch, input_len, hidden_dim)
        
        # Temporal reduction: project from input_len to output_len
        # Use learned interpolation with MLP
        h = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        
        # Multi-scale temporal pooling for output length
        # Adaptive pooling to output_len, then refine with learned weights
        h_pooled = F.adaptive_avg_pool1d(h, self.output_len)  # (batch, hidden_dim, output_len)
        
        # Also use max pooling for peak capture
        h_max = F.adaptive_max_pool1d(h, self.output_len)  # (batch, hidden_dim, output_len)
        
        # Learned combination
        h_temp = 0.7 * h_pooled + 0.3 * h_max  # (batch, hidden_dim, output_len)
        
        # Final refinement with 1x1 conv (properly defined as nn.Conv1d)
        h_temp = self.temporal_refine(h_temp)  # (batch, hidden_dim, output_len)
        
        h_temp = h_temp.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Output projection
        out = self.output_proj(h_temp)  # (batch, output_len, n_features)
        
        # Direct skip connection from raw input: simple linear projection across time
        x_skip = x_raw.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_skip = self.skip_proj(x_skip)  # (batch, n_features, output_len)
        x_skip = x_skip.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Residual combination with learned weight
        out = out + 0.1 * x_skip  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
