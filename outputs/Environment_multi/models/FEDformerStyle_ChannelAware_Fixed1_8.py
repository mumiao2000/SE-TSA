import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FEDformerStyle_ChannelAware_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'n_encoder_layers': [1, 4, 8],
            'n_decoder_layers': [1, 4, 8],
            'modes': [16, 64, 256],
            'dropout': [0.0, 0.3],
            'trend_dim': [32, 128, 512],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 512)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_encoder_layers = self.config.get('n_encoder_layers', 4)
        self.n_decoder_layers = self.config.get('n_decoder_layers', 4)
        self.modes = self.config.get('modes', 64)
        self.dropout = self.config.get('dropout', 0.1)
        self.trend_dim = self.config.get('trend_dim', 128)
        
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.d_model)  # (batch, seq, n_features) -> (batch, seq, d_model)
        
        # Trend extractor: parallel path for slow-varying patterns
        self.trend_pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.trend_proj = nn.Linear(self.n_features, self.trend_dim)  # (batch, seq, n_features) -> (batch, seq, trend_dim)
        self.trend_out_proj = nn.Linear(self.trend_dim, self.d_model)  # (batch, seq, trend_dim) -> (batch, seq, d_model)
        
        # Trend-cyclical fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.d_model + self.trend_dim, self.d_model),
            nn.Sigmoid()
        )  # (batch, seq, d_model + trend_dim) -> (batch, seq, d_model)
        
        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, max(input_len, output_len) * 2, self.d_model) * 0.02)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            FreqEnhancedBlockFixed(
                self.d_model, self.n_heads, self.head_dim, self.modes, self.dropout
            ) for _ in range(self.n_encoder_layers)
        ])
        self.encoder_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.n_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            FreqEnhancedBlockFixed(
                self.d_model, self.n_heads, self.head_dim, self.modes, self.dropout
            ) for _ in range(self.n_decoder_layers)
        ])
        self.decoder_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.n_decoder_layers)
        ])
        
        # Cross-channel gating after frequency processing
        self.channel_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, self.d_model),
            nn.Sigmoid()
        )  # (batch, seq, d_model) -> (batch, seq, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.n_features)  # (batch, seq, d_model) -> (batch, seq, n_features)
        
        # Decoder start token
        self.dec_start = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection
        x_enc = self.input_proj(x)  # (batch, input_len, d_model)
        
        # Trend extraction (parallel path)
        x_trend = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_trend = self.trend_pool(x_trend)  # (batch, n_features, input_len)
        x_trend = x_trend.permute(0, 2, 1)  # (batch, input_len, n_features)
        x_trend = self.trend_proj(x_trend)  # (batch, input_len, trend_dim)
        x_trend = self.trend_out_proj(x_trend)  # (batch, input_len, d_model)
        
        # Fusion gate: combine trend with projected input
        gate_input = torch.cat([x_enc, x_trend[:, :, :self.trend_dim]], dim=-1)  # (batch, input_len, d_model + trend_dim)
        gate = self.fusion_gate(gate_input)  # (batch, input_len, d_model)
        x_enc = gate * x_enc + (1 - gate) * x_trend  # (batch, input_len, d_model)
        
        # Add positional encoding
        x_enc = x_enc + self.pos_enc[:, :self.input_len, :]  # (batch, input_len, d_model)
        
        # Encoder
        for layer, norm in zip(self.encoder_layers, self.encoder_norms):
            residual = x_enc  # (batch, input_len, d_model)
            x_enc = layer(x_enc)  # (batch, input_len, d_model)
            x_enc = norm(x_enc + residual)  # (batch, input_len, d_model)
        
        # Decoder initialization
        dec_input = self.dec_start.expand(batch_size, self.output_len, -1)  # (batch, output_len, d_model)
        dec_input = dec_input + self.pos_enc[:, self.input_len:self.input_len + self.output_len, :]  # (batch, output_len, d_model)
        
        # Cross-attention: use encoder output as context
        # Simplified: decoder self-attention with encoder context via residual connections
        x_dec = dec_input  # (batch, output_len, d_model)
        for layer, norm in zip(self.decoder_layers, self.decoder_norms):
            residual = x_dec  # (batch, output_len, d_model)
            # Concatenate encoder context for cross-attention effect
            combined = torch.cat([x_enc, x_dec], dim=1)  # (batch, input_len + output_len, d_model)
            combined_out = layer(combined)  # (batch, input_len + output_len, d_model)
            x_dec_new = combined_out[:, self.input_len:, :]  # (batch, output_len, d_model)
            x_dec = norm(x_dec_new + residual)  # (batch, output_len, d_model)
        
        # Cross-channel gating
        gate = self.channel_gate(x_dec)  # (batch, output_len, d_model)
        x_dec = gate * x_dec  # (batch, output_len, d_model)
        
        # Output projection
        out = self.output_proj(x_dec)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out


class FreqEnhancedBlockFixed(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, modes, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.modes = modes
        self.dropout = dropout
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)  # (batch, seq_len, d_model)
        v = self.v_proj(x)  # (batch, seq_len, d_model)
        
        # Reshape to multi-head
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        
        # Frequency domain attention
        q_fft = torch.fft.rfft(q, dim=2)  # (batch, n_heads, seq_len//2+1, head_dim)
        k_fft = torch.fft.rfft(k, dim=2)  # (batch, n_heads, seq_len//2+1, head_dim)
        
        # Dynamic mode truncation: ensure we don't exceed available frequency modes
        max_modes = q_fft.shape[2]  # seq_len//2 + 1
        actual_modes = min(self.modes, max_modes)  # scalar
        
        # Truncate to actual modes
        q_fft = q_fft[:, :, :actual_modes, :]  # (batch, n_heads, actual_modes, head_dim)
        k_fft = k_fft[:, :, :actual_modes, :]  # (batch, n_heads, actual_modes, head_dim)
        
        # Compute attention in frequency domain
        scores = torch.matmul(q_fft, k_fft.transpose(-2, -1).conj()) * self.scale  # (batch, n_heads, actual_modes, actual_modes)
        attn = F.softmax(scores.real, dim=-1)  # (batch, n_heads, actual_modes, actual_modes)
        attn = attn + 1j * torch.zeros_like(attn)  # convert to complex
        
        # Apply attention to values in frequency domain
        v_fft = torch.fft.rfft(v, dim=2)  # (batch, n_heads, seq_len//2+1, head_dim)
        v_fft = v_fft[:, :, :actual_modes, :]  # (batch, n_heads, actual_modes, head_dim)
        
        out_fft = torch.matmul(attn, v_fft)  # (batch, n_heads, actual_modes, head_dim)
        
        # Pad back to original frequency dimension if needed
        if actual_modes < max_modes:
            pad_size = max_modes - actual_modes
            out_fft = F.pad(out_fft, (0, 0, 0, pad_size))  # (batch, n_heads, max_modes, head_dim)
        
        # Inverse FFT
        out = torch.fft.irfft(out_fft, n=seq_len, dim=2)  # (batch, n_heads, seq_len, head_dim)
        
        # Reshape and project
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)
        out = self.out_proj(out)  # (batch, seq_len, d_model)
        out = self.dropout_layer(out)  # (batch, seq_len, d_model)
        
        return out
