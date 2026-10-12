import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FEDformerStyle_ChannelAware_Improved_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'modes': [8, 32, 128],
            'decomp_kernel': [3, 15, 63],
            'ffn_dim': [256, 1024, 4096],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 512)
        self.n_heads = cfg.get('n_heads', 8)
        self.n_layers = cfg.get('n_layers', 4)
        self.dropout = cfg.get('dropout', 0.1)
        self.modes = cfg.get('modes', 32)
        self.decomp_kernel = cfg.get('decomp_kernel', 15)
        self.ffn_dim = cfg.get('ffn_dim', 1024)
        
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.d_model)  # (batch, seq, n_features) -> (batch, seq, d_model)
        
        # Multi-scale decomposition with learnable moving average kernels
        self.decomp_kernels = [self.decomp_kernel, self.decomp_kernel * 2 + 1, self.decomp_kernel * 4 + 3]
        self.decomp_layers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=k, stride=1, padding=k//2) for k in self.decomp_kernels
        ])
        self.decomp_proj = nn.Linear(len(self.decomp_kernels) * self.d_model, self.d_model)
        
        # Seasonal-trend separation weights
        self.seasonal_weights = nn.Parameter(torch.ones(len(self.decomp_kernels)))
        self.trend_weights = nn.Parameter(torch.ones(len(self.decomp_kernels)))
        
        # Encoder: stack of frequency-enhanced blocks with channel-aware attention
        self.encoder_layers = nn.ModuleList([
            FreqEnhancedChannelBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                modes=self.modes,
                dropout=self.dropout,
                ffn_dim=self.ffn_dim,
                n_features=self.n_features
            ) for _ in range(self.n_layers)
        ])
        
        # Gated cross-channel fusion
        self.channel_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.GELU(),
            nn.Linear(self.d_model // 4, self.n_features),
            nn.Sigmoid()
        )
        
        # Decoder: project to output
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.n_features)
        )
        
        # Direct path from input to output for residual learning
        self.direct_proj = nn.Linear(self.input_len, self.output_len)
        
    def multi_scale_decomp(self, x):
        # x: (batch, seq_len, d_model)
        # Decompose at multiple scales and combine
        seasonal_parts = []
        trend_parts = []
        
        x_perm = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        
        for i, pool in enumerate(self.decomp_layers):
            # Moving average for trend
            trend = pool(x_perm)  # (batch, d_model, seq_len)
            # Seasonal = original - trend
            seasonal = x_perm - trend
            
            seasonal_parts.append(seasonal * F.softmax(self.seasonal_weights, dim=0)[i])
            trend_parts.append(trend * F.softmax(self.trend_weights, dim=0)[i])
        
        # Combine seasonal and trend
        seasonal_combined = sum(seasonal_parts)  # (batch, d_model, seq_len)
        trend_combined = sum(trend_parts)  # (batch, d_model, seq_len)
        
        # Concatenate all decompositions
        multi_scale = torch.cat([
            seasonal_combined,
            trend_combined,
            x_perm  # original residual
        ], dim=1)  # (batch, 3 * d_model, seq_len)
        
        multi_scale = multi_scale.permute(0, 2, 1)  # (batch, seq_len, 3 * d_model)
        out = self.decomp_proj(multi_scale)  # (batch, seq_len, d_model)
        return out
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # Store for direct residual
        x_input = x  # (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection
        x = self.input_proj(x)  # (batch, input_len, d_model)
        
        # Multi-scale decomposition
        x = self.multi_scale_decomp(x)  # (batch, input_len, d_model)
        
        # Encoder with frequency-enhanced blocks
        for layer in self.encoder_layers:
            x = layer(x)  # (batch, input_len, d_model)
        
        # Global average pooling for context
        context = x.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        
        # Expand to output length
        x = context.expand(-1, self.output_len, -1)  # (batch, output_len, d_model)
        
        # Gated cross-channel fusion
        gate = self.channel_gate(x)  # (batch, output_len, n_features)
        # gate will be applied after decoding
        
        # Decode to features
        out = self.decoder(x)  # (batch, output_len, n_features)
        
        # Apply channel gate (broadcast across features)
        out = out * gate  # (batch, output_len, n_features)
        
        # Direct residual from input
        direct = x_input.permute(0, 2, 1)  # (batch, n_features, input_len)
        direct = self.direct_proj(direct)  # (batch, n_features, output_len)
        direct = direct.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Residual connection
        out = out + direct * 0.1  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)


class FreqEnhancedChannelBlock(nn.Module):
    def __init__(self, d_model, n_heads, modes, dropout, ffn_dim, n_features):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.modes = modes
        self.head_dim = d_model // n_heads
        self.n_features = n_features
        
        # Channel-aware frequency attention
        self.freq_attn = ChannelAwareFreqAttention(d_model, n_heads, modes, n_features)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        residual = x
        x = self.norm1(x)
        x = self.freq_attn(x)  # (batch, seq_len, d_model)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)  # (batch, seq_len, d_model)
        x = x + residual
        
        return x


class ChannelAwareFreqAttention(nn.Module):
    def __init__(self, d_model, n_heads, modes, n_features):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.modes = modes
        self.head_dim = d_model // n_heads
        self.n_features = n_features
        
        # Per-channel mode allocation (distribute modes across features)
        self.modes_per_feature = max(1, modes // n_features)
        self.total_modes = min(modes, 64)  # cap total modes for stability
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Learnable frequency filters per head (truncated dynamically)
        self.freq_filter_real = nn.Parameter(torch.randn(n_heads, self.total_modes, 1) * 0.02)
        self.freq_filter_imag = nn.Parameter(torch.randn(n_heads, self.total_modes, 1) * 0.02)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)  # (batch, seq_len, d_model)
        v = self.v_proj(x)  # (batch, seq_len, d_model)
        
        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # rFFT on last two dims: (batch, n_heads, seq_len, head_dim) -> freq on seq_len
        q_fft = torch.fft.rfft(q, dim=2)  # (batch, n_heads, seq_len//2+1, head_dim)
        k_fft = torch.fft.rfft(k, dim=2)  # (batch, n_heads, seq_len//2+1, head_dim)
        v_fft = torch.fft.rfft(v, dim=2)  # (batch, n_heads, seq_len//2+1, head_dim)
        
        # Dynamic mode truncation
        max_modes = q_fft.shape[2]
        active_modes = min(self.total_modes, max_modes)
        
        # Truncate frequency modes
        q_fft = q_fft[:, :, :active_modes, :]  # (batch, n_heads, active_modes, head_dim)
        k_fft = k_fft[:, :, :active_modes, :]  # (batch, n_heads, active_modes, head_dim)
        v_fft = v_fft[:, :, :active_modes, :]  # (batch, n_heads, active_modes, head_dim)
        
        # Apply learnable frequency filters (truncated to active_modes)
        filter_real = self.freq_filter_real[:, :active_modes, :]  # (n_heads, active_modes, 1)
        filter_imag = self.freq_filter_imag[:, :active_modes, :]  # (n_heads, active_modes, 1)
        
        # Complex multiplication in frequency domain for Q
        q_filtered_real = q_fft.real * filter_real - q_fft.imag * filter_imag
        q_filtered_imag = q_fft.real * filter_imag + q_fft.imag * filter_real
        q_filtered = torch.complex(q_filtered_real, q_filtered_imag)
        
        # Complex multiplication in frequency domain for K
        k_filtered_real = k_fft.real * filter_real - k_fft.imag * filter_imag
        k_filtered_imag = k_fft.real * filter_imag + k_fft.imag * filter_real
        k_filtered = torch.complex(k_filtered_real, k_filtered_imag)
        
        # Attention in frequency domain: Q_fft * K_fft^* then multiply with V_fft
        scores = torch.matmul(q_filtered, k_filtered.conj().transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores.real.abs(), dim=-1)
        
        # Apply attention to V
        out_fft = torch.matmul(attn_weights.to(v_fft.dtype), v_fft)
        
        # Pad back to original frequency dimension
        if active_modes < max_modes:
            pad_size = max_modes - active_modes
            out_fft = F.pad(out_fft, (0, 0, 0, pad_size, 0, 0, 0, 0))
        
        # Inverse FFT
        out = torch.fft.irfft(out_fft, n=seq_len, dim=2)  # (batch, n_heads, seq_len, head_dim)
        
        # Reshape back
        out = out.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, n_heads, head_dim)
        out = out.view(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)
        
        out = self.out_proj(out)  # (batch, seq_len, d_model)
        return out

