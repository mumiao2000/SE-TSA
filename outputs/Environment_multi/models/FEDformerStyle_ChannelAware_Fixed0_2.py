import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FEDformerStyle_ChannelAware_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'modes': [8, 64, 256],
            'decomp_kernel': [13, 25, 51],
            'channel_dim': [16, 64, 256],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.modes = self.config.get('modes', 64)
        self.decomp_kernel = self.config.get('decomp_kernel', 25)
        self.channel_dim = self.config.get('channel_dim', 64)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale trend extractor with dilated convolutions
        self.trend_extractors = nn.ModuleList([
            nn.Conv1d(self.n_features, self.n_features, kernel_size=k, padding=(k-1)//2, dilation=1, groups=self.n_features, bias=False)
            for k in [self.decomp_kernel, self.decomp_kernel * 2 + 1]
        ])
        self.trend_fusion = nn.Conv1d(self.n_features * 2, self.n_features, kernel_size=1)
        
        # Channel-wise adaptive gating
        self.channel_gate = nn.Sequential(
            nn.Linear(self.n_features, self.channel_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.channel_dim, self.n_features),
            nn.Sigmoid()
        )
        
        # Input projection
        self.input_projection = nn.Linear(self.n_features, self.d_model)
        
        # Frequency-enhanced attention blocks
        self.encoder_layers = nn.ModuleList([
            FreqEnhancedBlock(self.d_model, self.n_heads, self.modes, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Layer norms
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.d_model)
            for _ in range(self.n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.n_features)
        
        # Direct trend prediction for long-term forecasting
        self.trend_predictor = nn.Sequential(
            nn.Linear(self.input_len, self.output_len),
            nn.Dropout(self.dropout)
        )
        
        # Seasonal prediction
        self.seasonal_predictor = nn.Sequential(
            nn.Linear(self.input_len, self.output_len),
            nn.Dropout(self.dropout)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x_norm = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-wise adaptive gating
        channel_importance = self.channel_gate(x_norm.mean(dim=1))  # (batch, n_features)
        x_gated = x_norm * channel_importance.unsqueeze(1)  # (batch, input_len, n_features)
        
        # Multi-scale trend extraction
        x_perm = x_gated.permute(0, 2, 1)  # (batch, n_features, input_len)
        trend_components = []
        for extractor in self.trend_extractors:
            t = extractor(x_perm)  # (batch, n_features, input_len)
            trend_components.append(t)
        trend_concat = torch.cat(trend_components, dim=1)  # (batch, n_features*2, input_len)
        trend = self.trend_fusion(trend_concat)  # (batch, n_features, input_len)
        trend = trend.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Seasonal component
        seasonal = x_gated - trend  # (batch, input_len, n_features)
        
        # Predict trend continuation
        trend_pred = self.trend_predictor(trend.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Process seasonal through frequency-enhanced encoder
        seasonal_proj = self.input_projection(seasonal)  # (batch, input_len, d_model)
        
        # Encoder layers with residual connections
        h = seasonal_proj
        for i, (block, norm) in enumerate(zip(self.encoder_layers, self.norm_layers)):
            h_new = block(h)  # (batch, input_len, d_model)
            h = norm(h + h_new)  # (batch, input_len, d_model)
        
        # Project to output features
        seasonal_encoded = self.output_projection(h)  # (batch, input_len, n_features)
        
        # Predict seasonal continuation
        seasonal_pred = self.seasonal_predictor(seasonal_encoded.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Combine predictions with learnable residual weight
        out = trend_pred + self.residual_weight * seasonal_pred  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out


class FreqEnhancedBlock(nn.Module):
    def __init__(self, d_model, n_heads, modes, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.modes = modes
        self.head_dim = d_model // n_heads
        
        # Frequency-domain attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Learnable frequency filters (max size, will be truncated dynamically)
        self.freq_real = nn.Parameter(torch.randn(n_heads, modes) * 0.02)
        self.freq_imag = nn.Parameter(torch.randn(n_heads, modes) * 0.02)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)  # (batch, seq_len, d_model)
        v = self.v_proj(x)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        
        # FFT-based attention in frequency domain
        q_fft = torch.fft.rfft(q, dim=2)  # (batch, n_heads, seq_len//2+1, head_dim)
        k_fft = torch.fft.rfft(k, dim=2)  # (batch, n_heads, seq_len//2+1, head_dim)
        
        # Determine actual number of frequency modes available
        max_modes = q_fft.shape[2]  # seq_len // 2 + 1
        actual_modes = min(self.modes, max_modes)  # clamp to available modes
        
        # Truncate to actual modes
        q_fft = q_fft[:, :, :actual_modes, :]  # (batch, n_heads, actual_modes, head_dim)
        k_fft = k_fft[:, :, :actual_modes, :]  # (batch, n_heads, actual_modes, head_dim)
        
        # Apply learnable frequency filters (truncate parameters to actual_modes)
        freq_filter = torch.complex(self.freq_real[:, :actual_modes], self.freq_imag[:, :actual_modes]).unsqueeze(-1)  # (n_heads, actual_modes, 1)
        q_fft = q_fft * freq_filter.unsqueeze(0)  # (batch, n_heads, actual_modes, head_dim)
        k_fft = k_fft * freq_filter.unsqueeze(0)  # (batch, n_heads, actual_modes, head_dim)
        
        # Attention in frequency domain
        scores = torch.einsum('bhmd,bhnd->bhmn', q_fft, k_fft.conj()) * self.scale  # (batch, n_heads, actual_modes, actual_modes)
        attn = F.softmax(scores.real, dim=-1)  # (batch, n_heads, actual_modes, actual_modes)
        attn = torch.complex(attn, torch.zeros_like(attn))
        
        v_fft = torch.fft.rfft(v, dim=2)[:, :, :actual_modes, :]  # (batch, n_heads, actual_modes, head_dim)
        out_fft = torch.einsum('bhmn,bhnd->bhmd', attn, v_fft)  # (batch, n_heads, actual_modes, head_dim)
        
        # Pad back and inverse FFT
        out_fft_full = torch.zeros(batch_size, self.n_heads, max_modes, self.head_dim, dtype=out_fft.dtype, device=out_fft.device)
        out_fft_full[:, :, :actual_modes, :] = out_fft
        
        out = torch.fft.irfft(out_fft_full, n=seq_len, dim=2)  # (batch, n_heads, seq_len, head_dim)
        
        # Reshape and project
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)
        out = self.out_proj(out)  # (batch, seq_len, d_model)
        out = self.dropout(out)
        
        return out
