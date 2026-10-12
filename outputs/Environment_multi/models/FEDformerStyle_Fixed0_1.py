import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FEDformerStyle_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.3],
            'modes': [8, 32, 128],
            'decomp_kernel': [13, 25, 51]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.modes = self.config.get('modes', 32)
        self.decomp_kernel = self.config.get('decomp_kernel', 25)
        
        # Ensure odd kernel size
        if self.decomp_kernel % 2 == 0:
            self.decomp_kernel += 1
        
        # Decomposition
        self.decomp = nn.AvgPool1d(kernel_size=self.decomp_kernel, stride=1, padding=self.decomp_kernel//2)
        
        # Input embedding
        self.value_embedding = nn.Linear(self.n_features, self.d_model)
        self.seasonal_embedding = nn.Linear(self.n_features, self.d_model)
        
        # Frequency attention blocks
        self.seasonal_blocks = nn.ModuleList()
        self.trend_blocks = nn.ModuleList()
        
        for _ in range(self.n_layers):
            self.seasonal_blocks.append(
                FrequencyAttention(self.d_model, self.n_heads, self.modes, self.dropout)
            )
            self.trend_blocks.append(
                nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * 4),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.d_model * 4, self.d_model)
                )
            )
        
        # Output projection
        self.seasonal_proj = nn.Linear(self.d_model, self.n_features)
        self.trend_proj = nn.Linear(self.d_model, self.n_features)
        
        # Direct predictors
        self.seasonal_predictor = nn.Linear(self.input_len, self.output_len)
        self.trend_predictor = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # Instance normalization
        mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        x = (x - mean) / std  # (batch, input_len, n_features)
        
        # Decomposition
        x_permute = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        moving_avg = self.decomp(x_permute)  # (batch, n_features, input_len)
        moving_avg = moving_avg.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        residual = x - moving_avg  # (batch, input_len, n_features)
        
        # Trend and seasonal embeddings
        trend = self.value_embedding(moving_avg)  # (batch, input_len, d_model)
        seasonal = self.seasonal_embedding(residual)  # (batch, input_len, d_model)
        
        # Process through blocks
        for s_block, t_block in zip(self.seasonal_blocks, self.trend_blocks):
            # Seasonal with frequency attention
            seasonal = s_block(seasonal)  # (batch, input_len, d_model)
            # Trend with MLP
            trend = trend + t_block(trend)  # (batch, input_len, d_model)
        
        # Predict
        trend = trend.permute(0, 2, 1)  # (batch, d_model, input_len)
        seasonal = seasonal.permute(0, 2, 1)  # (batch, d_model, input_len)
        
        trend = self.trend_predictor(trend)  # (batch, d_model, output_len)
        seasonal = self.seasonal_predictor(seasonal)  # (batch, d_model, output_len)
        
        trend = trend.permute(0, 2, 1)  # (batch, output_len, d_model)
        seasonal = seasonal.permute(0, 2, 1)  # (batch, output_len, d_model)
        
        trend = self.trend_proj(trend)  # (batch, output_len, n_features)
        seasonal = self.seasonal_proj(seasonal)  # (batch, output_len, n_features)
        
        x = trend + seasonal  # (batch, output_len, n_features)
        
        # Denormalize
        x = x * std + mean  # (batch, output_len, n_features)
        
        return x


class FrequencyAttention(nn.Module):
    def __init__(self, d_model, n_heads, modes=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.modes = modes
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable frequency modes - use torch.sqrt instead of math.sqrt
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)  # (batch, seq_len, d_model)
        v = self.v_proj(x)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        
        # Apply FFT for frequency domain attention
        q_fft = torch.fft.rfft(q, dim=2)  # (batch, n_heads, freq_len, head_dim)
        k_fft = torch.fft.rfft(k, dim=2)  # (batch, n_heads, freq_len, head_dim)
        
        # Keep only top modes
        freq_len = q_fft.shape[2]
        modes = min(self.modes, freq_len)
        
        q_fft = q_fft[:, :, :modes, :]  # (batch, n_heads, modes, head_dim)
        k_fft = k_fft[:, :, :modes, :]  # (batch, n_heads, modes, head_dim)
        
        # Frequency domain attention
        scores = torch.matmul(q_fft, k_fft.transpose(-2, -1).conj()) * self.scale  # (batch, n_heads, modes, modes)
        attn = F.softmax(scores.real, dim=-1)  # (batch, n_heads, modes, modes)
        attn = attn + 1j * torch.zeros_like(attn)  # Convert to complex
        
        # Apply to v in frequency domain
        v_fft = torch.fft.rfft(v, dim=2)  # (batch, n_heads, freq_len, head_dim)
        v_fft = v_fft[:, :, :modes, :]  # (batch, n_heads, modes, head_dim)
        
        out_fft = torch.matmul(attn, v_fft)  # (batch, n_heads, modes, head_dim)
        
        # Pad back to full frequency length
        full_out = torch.zeros(batch_size, self.n_heads, freq_len, self.head_dim, dtype=out_fft.dtype, device=out_fft.device)
        full_out[:, :, :modes, :] = out_fft
        
        # Inverse FFT
        out = torch.fft.irfft(full_out, n=seq_len, dim=2)  # (batch, n_heads, seq_len, head_dim)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)
        out = self.out_proj(out)  # (batch, seq_len, d_model)
        out = self.dropout(out)
        
        return out

