import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FEDformerStyle_ChannelAware_Fixed1_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'modes': [8, 32, 128],
            'd_ff': [256, 1024, 4096],
            'kernel_sizes': [[3], [3, 7], [3, 7, 15]],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 512)
        self.n_heads = cfg.get('n_heads', 8)
        self.n_layers = cfg.get('n_layers', 4)
        self.dropout = cfg.get('dropout', 0.1)
        self.modes = cfg.get('modes', 32)
        self.d_ff = cfg.get('d_ff', 1024)
        
        # Fix: ensure kernel_sizes is always a list
        kernel_sizes_val = cfg.get('kernel_sizes', [3, 7, 15])
        if isinstance(kernel_sizes_val, int):
            self.kernel_sizes = [kernel_sizes_val]
        else:
            self.kernel_sizes = list(kernel_sizes_val)
        
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: (batch, input_len, n_features) -> (batch, input_len, d_model)
        self.input_projection = nn.Linear(self.n_features, self.d_model)
        
        # Multi-scale temporal decomposition
        self.decomp_kernels = nn.ModuleList([
            nn.AvgPool1d(kernel_size=k, stride=1, padding=k//2)
            for k in self.kernel_sizes
        ])
        self.decomp_proj = nn.Linear(len(self.kernel_sizes) * self.d_model, self.d_model)
        
        # Encoder: stack of frequency-enhanced blocks with multi-scale decomposition
        self.encoder_layers = nn.ModuleList([
            FreqEnhancedBlockFixed(
                d_model=self.d_model,
                n_heads=self.n_heads,
                modes=self.modes,
                dropout=self.dropout,
                d_ff=self.d_ff,
                kernel_sizes=self.kernel_sizes
            )
            for _ in range(self.n_layers)
        ])
        
        # Cross-channel gated fusion
        self.channel_gate = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Sigmoid()
        )
        
        # Decoder: project to output length
        self.decoder = nn.Sequential(
            nn.Linear(self.input_len, self.output_len),
            nn.Dropout(self.dropout),
        )
        
        # Output projection: (batch, output_len, d_model) -> (batch, output_len, n_features)
        self.output_projection = nn.Linear(self.d_model, self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection
        x = self.input_projection(x)  # (batch, input_len, d_model)
        
        # Multi-scale temporal decomposition
        decomposed = []
        x_perm = x.permute(0, 2, 1)  # (batch, d_model, input_len)
        for pool in self.decomp_kernels:
            trend = pool(x_perm)  # (batch, d_model, input_len)
            decomposed.append(trend.permute(0, 2, 1))  # (batch, input_len, d_model)
        
        # Concatenate multi-scale decompositions and project back
        multi_scale = torch.cat(decomposed, dim=-1)  # (batch, input_len, len(kernels)*d_model)
        x_decomp = self.decomp_proj(multi_scale)  # (batch, input_len, d_model)
        
        # Residual connection with original input
        x = x + x_decomp  # (batch, input_len, d_model)
        
        # Encoder with frequency attention
        for layer in self.encoder_layers:
            x = layer(x)  # (batch, input_len, d_model)
        
        # Cross-channel gated fusion: combine time and frequency info
        # Compute channel-wise statistics
        channel_mean = x.mean(dim=1, keepdim=True).expand(-1, self.input_len, -1)  # (batch, input_len, d_model)
        channel_gate_input = torch.cat([x, channel_mean], dim=-1)  # (batch, input_len, d_model*2)
        gate = self.channel_gate(channel_gate_input)  # (batch, input_len, d_model)
        x = gate * x + (1 - gate) * channel_mean  # (batch, input_len, d_model)
        
        # Decode to output length
        x = x.permute(0, 2, 1)  # (batch, d_model, input_len)
        x = self.decoder(x)  # (batch, d_model, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, d_model)
        
        # Output projection
        x = self.output_projection(x)  # (batch, output_len, n_features)
        
        # Denormalize
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x


class FreqEnhancedBlockFixed(nn.Module):
    def __init__(self, d_model, n_heads, modes, dropout, d_ff, kernel_sizes):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.modes = modes
        self.head_dim = d_model // n_heads
        
        # Fix: ensure kernel_sizes is always a list
        if isinstance(kernel_sizes, int):
            self.kernel_sizes = [kernel_sizes]
        else:
            self.kernel_sizes = list(kernel_sizes)
        
        # Frequency attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Learnable frequency filter (real-valued, applied to magnitude)
        self.freq_filter = nn.Parameter(torch.randn(n_heads, modes, 1) * 0.02)
        
        # Time-domain attention for local patterns
        self.time_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Gating for frequency vs time attention
        self.gate = nn.Linear(d_model * 2, 2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Frequency attention path
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)  # (batch, seq_len, d_model)
        v = self.v_proj(x)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head: (batch, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply rFFT along sequence dimension
        q_fft = torch.fft.rfft(q, dim=2)  # (batch, n_heads, seq_len//2+1, head_dim) complex
        k_fft = torch.fft.rfft(k, dim=2)  # (batch, n_heads, seq_len//2+1, head_dim) complex
        v_fft = torch.fft.rfft(v, dim=2)  # (batch, n_heads, seq_len//2+1, head_dim) complex
        
        # Dynamic mode truncation
        max_modes = q_fft.shape[2]
        actual_modes = min(self.modes, max_modes)
        
        # Truncate to actual modes
        q_fft = q_fft[:, :, :actual_modes, :]  # (batch, n_heads, actual_modes, head_dim) complex
        k_fft = k_fft[:, :, :actual_modes, :]  # (batch, n_heads, actual_modes, head_dim) complex
        v_fft = v_fft[:, :, :actual_modes, :]  # (batch, n_heads, actual_modes, head_dim) complex
        
        # Resize filter if needed
        if actual_modes != self.freq_filter.shape[1]:
            freq_filter = self.freq_filter[:, :actual_modes, :]
        else:
            freq_filter = self.freq_filter
        
        # Apply learnable frequency filter to magnitude (real-valued operation)
        q_magnitude = torch.abs(q_fft)  # (batch, n_heads, actual_modes, head_dim) real
        k_magnitude = torch.abs(k_fft)  # (batch, n_heads, actual_modes, head_dim) real
        
        q_magnitude = q_magnitude * freq_filter  # (batch, n_heads, actual_modes, head_dim) real
        k_magnitude = k_magnitude * freq_filter  # (batch, n_heads, actual_modes, head_dim) real
        
        # Frequency domain attention using magnitudes (real-valued, softmax-safe)
        scores = torch.matmul(q_magnitude, k_magnitude.transpose(-2, -1))  # (batch, n_heads, actual_modes, actual_modes) real
        scores = scores / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, actual_modes, actual_modes) real
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to complex v_fft by treating real and imag separately
        v_real = v_fft.real  # (batch, n_heads, actual_modes, head_dim) real
        v_imag = v_fft.imag  # (batch, n_heads, actual_modes, head_dim) real
        
        freq_out_real = torch.matmul(attn_weights, v_real)  # (batch, n_heads, actual_modes, head_dim) real
        freq_out_imag = torch.matmul(attn_weights, v_imag)  # (batch, n_heads, actual_modes, head_dim) real
        
        # Reconstruct complex tensor
        freq_out = torch.complex(freq_out_real, freq_out_imag)  # (batch, n_heads, actual_modes, head_dim) complex
        
        # Pad back to original frequency size and inverse FFT
        freq_full = torch.zeros(batch_size, self.n_heads, max_modes, self.head_dim, 
                                dtype=freq_out.dtype, device=freq_out.device)
        freq_full[:, :, :actual_modes, :] = freq_out  # (batch, n_heads, max_modes, head_dim) complex
        
        freq_out = torch.fft.irfft(freq_full, n=seq_len, dim=2)  # (batch, n_heads, seq_len, head_dim) real
        freq_out = freq_out.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, n_heads, head_dim) real
        freq_out = freq_out.view(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model) real
        
        # Time-domain attention path
        time_out, _ = self.time_attn(x, x, x)  # (batch, seq_len, d_model)
        
        # Gated fusion of frequency and time paths
        gate_input = torch.cat([freq_out, time_out], dim=-1)  # (batch, seq_len, d_model*2)
        gates = F.softmax(self.gate(gate_input), dim=-1)  # (batch, seq_len, 2)
        
        fused = gates[:, :, 0:1] * freq_out + gates[:, :, 1:2] * time_out  # (batch, seq_len, d_model)
        
        # First residual
        x = self.norm1(x + self.dropout(fused))  # (batch, seq_len, d_model)
        
        # Feedforward
        ff_out = self.ffn(x)  # (batch, seq_len, d_model)
        x = self.norm2(x + ff_out)  # (batch, seq_len, d_model)
        
        return x
