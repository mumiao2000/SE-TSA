import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FEDformerLite_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 512],
            'n_heads': [2, 4, 8],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.1, 0.3],
            'modes': [8, 32, 64]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 4)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.modes = self.config.get('modes', 32)
        
        # Ensure divisibility
        while self.d_model % self.n_heads != 0 and self.n_heads > 1:
            self.n_heads -= 1
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Decomposition
        self.decomp = SeriesDecomp(kernel_size=13)
        
        # Embedding
        self.enc_embedding = nn.Linear(self.n_features, self.d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            FEDformerEncoderLayer(self.d_model, self.n_heads, self.modes, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.norm = nn.LayerNorm(self.d_model)
        
        # Decoder: project to output
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.n_features)
        )
        
        # Temporal projection
        self.temporal_proj = nn.Linear(self.input_len, self.output_len)
        
        # Direct
        self.direct = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Decomposition
        seasonal, trend = self.decomp(x)  # both (batch, input_len, n_features)
        
        # Direct for trend
        trend_perm = trend.permute(0, 2, 1)  # (batch, n_features, input_len)
        trend_out = self.direct(trend_perm)  # (batch, n_features, output_len)
        trend_out = trend_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Process seasonal with frequency attention
        seasonal_embed = self.enc_embedding(seasonal)  # (batch, input_len, d_model)
        seasonal_embed = seasonal_embed.permute(1, 0, 2)  # (input_len, batch, d_model)
        
        for layer in self.encoder_layers:
            seasonal_embed = layer(seasonal_embed)  # (input_len, batch, d_model)
        
        seasonal_embed = seasonal_embed.permute(1, 0, 2)  # (batch, input_len, d_model)
        seasonal_embed = self.norm(seasonal_embed)  # (batch, input_len, d_model)
        
        # Temporal projection
        seasonal_embed = seasonal_embed.permute(0, 2, 1)  # (batch, d_model, input_len)
        seasonal_embed = self.temporal_proj(seasonal_embed)  # (batch, d_model, output_len)
        seasonal_embed = seasonal_embed.permute(0, 2, 1)  # (batch, output_len, d_model)
        
        # Decode
        seasonal_out = self.decoder(seasonal_embed)  # (batch, output_len, n_features)
        
        # Combine
        out = seasonal_out + trend_out  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        trend = self.moving_avg(x_perm)  # (batch, n_features, seq_len)
        trend = trend.permute(0, 2, 1)  # (batch, seq_len, n_features)
        seasonal = x - trend  # (batch, seq_len, n_features)
        return seasonal, trend


class FEDformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, modes, dropout):
        super().__init__()
        self.d_model = d_model
        self.modes = modes
        
        # Frequency domain attention
        self.freq_attn = FrequencyAttention(d_model, n_heads, modes, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (seq_len, batch, d_model)
        attn_out = self.freq_attn(x)  # (seq_len, batch, d_model)
        x = self.norm1(x + attn_out)  # (seq_len, batch, d_model)
        
        ffn_out = self.ffn(x)  # (seq_len, batch, d_model)
        x = self.norm2(x + ffn_out)  # (seq_len, batch, d_model)
        
        return x


class FrequencyAttention(nn.Module):
    def __init__(self, d_model, n_heads, modes, dropout):
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
    
    def forward(self, x):
        # x: (seq_len, batch, d_model)
        seq_len, batch, d_model = x.shape
        
        # Project
        q = self.q_proj(x)  # (seq_len, batch, d_model)
        k = self.k_proj(x)  # (seq_len, batch, d_model)
        v = self.v_proj(x)  # (seq_len, batch, d_model)
        
        # FFT for frequency domain - use real FFT
        q_ft = torch.fft.rfft(q, dim=0)  # (freq, batch, d_model)
        k_ft = torch.fft.rfft(k, dim=0)  # (freq, batch, d_model)
        v_ft = torch.fft.rfft(v, dim=0)  # (freq, batch, d_model)
        
        # Select modes
        min_modes = min(self.modes, q_ft.shape[0])
        q_ft = q_ft[:min_modes]  # (min_modes, batch, d_model)
        k_ft = k_ft[:min_modes]  # (min_modes, batch, d_model)
        v_ft = v_ft[:min_modes]  # (min_modes, batch, d_model)
        
        # Convert complex to real by taking magnitude and phase separately
        # Use view_as_real to get real-valued tensor: [..., 2] for real/imag parts
        q_real = torch.view_as_real(q_ft)  # (min_modes, batch, d_model, 2)
        k_real = torch.view_as_real(k_ft)  # (min_modes, batch, d_model, 2)
        v_real = torch.view_as_real(v_ft)  # (min_modes, batch, d_model, 2)
        
        # Combine real and imag as separate features: (min_modes, batch, d_model * 2)
        q_real = q_real.reshape(min_modes, batch, -1)  # (min_modes, batch, d_model * 2)
        k_real = k_real.reshape(min_modes, batch, -1)  # (min_modes, batch, d_model * 2)
        v_real = v_real.reshape(min_modes, batch, -1)  # (min_modes, batch, d_model * 2)
        
        # Permute for attention: (batch, min_modes, d_model * 2)
        q_real = q_real.permute(1, 0, 2)  # (batch, min_modes, d_model * 2)
        k_real = k_real.permute(1, 0, 2)  # (batch, min_modes, d_model * 2)
        v_real = v_real.permute(1, 0, 2)  # (batch, min_modes, d_model * 2)
        
        # Multi-head: treat the doubled dimension as the new feature dim
        # We need to adjust head_dim to account for doubled features
        effective_dim = d_model * 2
        effective_head_dim = effective_dim // self.n_heads
        
        q_real = q_real.reshape(batch, min_modes, self.n_heads, effective_head_dim).permute(0, 2, 1, 3)  # (batch, n_heads, min_modes, effective_head_dim)
        k_real = k_real.reshape(batch, min_modes, self.n_heads, effective_head_dim).permute(0, 2, 1, 3)  # (batch, n_heads, min_modes, effective_head_dim)
        v_real = v_real.reshape(batch, min_modes, self.n_heads, effective_head_dim).permute(0, 2, 1, 3)  # (batch, n_heads, min_modes, effective_head_dim)
        
        # Scaled dot-product attention (all real-valued now)
        scores = torch.matmul(q_real, k_real.transpose(-2, -1)) / (effective_head_dim ** 0.5)  # (batch, n_heads, min_modes, min_modes)
        attn = F.softmax(scores, dim=-1)  # (batch, n_heads, min_modes, min_modes)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v_real)  # (batch, n_heads, min_modes, effective_head_dim)
        out = out.permute(0, 2, 1, 3).reshape(batch, min_modes, effective_dim)  # (batch, min_modes, d_model * 2)
        
        # Split back to real/imag and reconstruct complex
        out = out.reshape(batch, min_modes, d_model, 2)  # (batch, min_modes, d_model, 2)
        # Use torch.stack and complex constructor instead of view_as_complex
        out_complex = torch.complex(out[..., 0], out[..., 1])  # (batch, min_modes, d_model)
        
        # IFFT back to time domain
        out_complex = out_complex.permute(1, 0, 2)  # (min_modes, batch, d_model)
        
        # Pad to original frequency length
        full_freq = torch.zeros(seq_len // 2 + 1, batch, d_model, dtype=out_complex.dtype, device=out_complex.device)  # (seq_len//2+1, batch, d_model)
        full_freq[:min_modes] = out_complex  # (seq_len//2+1, batch, d_model)
        
        out = torch.fft.irfft(full_freq, n=seq_len, dim=0)  # (seq_len, batch, d_model)
        
        out = self.out_proj(out)  # (seq_len, batch, d_model)
        
        return out
