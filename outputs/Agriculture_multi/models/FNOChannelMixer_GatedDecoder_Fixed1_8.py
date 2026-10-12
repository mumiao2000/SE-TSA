import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FNOChannelMixer_GatedDecoder_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_modes': [8, 24, 64],
            'n_layers': [1, 4, 8],
            'decoder_dim': [128, 512, 2048],
            'dropout': [0.0, 0.3],
            'gate_ratio': [2, 4],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_modes = self.config.get('n_modes', 24)
        self.n_layers = self.config.get('n_layers', 4)
        self.decoder_dim = self.config.get('decoder_dim', 512)
        self.dropout = self.config.get('dropout', 0.1)
        self.gate_ratio = self.config.get('gate_ratio', 2)
        
        # Actual number of modes is limited by input length
        self.actual_n_modes = min(self.n_modes, self.input_len // 2 + 1)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to hidden dimension
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # FNO layers with multi-scale frequency gating
        self.fno_layers = nn.ParameterList()
        self.freq_gates = nn.ModuleList()
        for _ in range(self.n_layers):
            # Spectral convolution weights sized to actual_n_modes
            self.fno_layers.append(nn.Parameter(
                torch.randn(self.hidden_dim, self.actual_n_modes, 2) * 0.02
            ))
            # Multi-scale frequency gating: low vs high freq
            self.freq_gates.append(nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Sigmoid()
            ))
        
        # Temporal mixing after spectral operations
        self.temporal_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_len, self.input_len),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ) for _ in range(self.n_layers)
        ])
        
        # Layer norms for stability
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layers)
        ])
        
        # Encoder-to-decoder skip projection
        self.skip_proj = nn.Linear(self.hidden_dim, self.decoder_dim)
        
        # Gated decoder: Stage 1 - feature-wise expansion with SiLU gating
        self.decoder_gate_expand = nn.Linear(self.hidden_dim, self.decoder_dim * self.gate_ratio)
        self.decoder_gate_proj = nn.Linear(self.hidden_dim, self.decoder_dim * self.gate_ratio)
        self.decoder_gate_combine = nn.Linear(self.decoder_dim * self.gate_ratio, self.decoder_dim)
        
        # Stage 2 - cross-channel attention gating
        self.cross_channel_query = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.cross_channel_key = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.cross_channel_value = nn.Linear(self.decoder_dim, self.decoder_dim)
        self.cross_channel_gate = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.Sigmoid()
        )
        self.cross_channel_norm = nn.LayerNorm(self.decoder_dim)
        
        # Output projections
        self.output_temporal = nn.Linear(self.input_len, self.output_len)
        self.output_feature = nn.Linear(self.decoder_dim, self.n_features)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden dimension
        h = self.input_proj(x)  # (batch, input_len, hidden_dim)
        h = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        
        # Store for skip connection
        h_skip = h  # (batch, hidden_dim, input_len)
        
        # FNO layers with spectral filtering and multi-scale gating
        for i in range(self.n_layers):
            h_in = h  # (batch, hidden_dim, input_len)
            
            # FFT to frequency domain
            h_fft = torch.fft.rfft(h, dim=-1)  # (batch, hidden_dim, input_len//2+1)
            
            # Truncate to actual_n_modes
            h_fft_trunc = h_fft[:, :, :self.actual_n_modes]  # (batch, hidden_dim, actual_n_modes)
            
            # Separate real and imaginary parts
            h_fft_real = h_fft_trunc.real  # (batch, hidden_dim, actual_n_modes)
            h_fft_imag = h_fft_trunc.imag  # (batch, hidden_dim, actual_n_modes)
            h_fft_stack = torch.stack([h_fft_real, h_fft_imag], dim=-1)  # (batch, hidden_dim, actual_n_modes, 2)
            
            # Apply spectral weights with correct einsum
            weights = self.fno_layers[i]  # (hidden_dim, actual_n_modes, 2)
            h_fft_filtered = torch.einsum('bfnm,fnm->bfnm', h_fft_stack, weights)  # (batch, hidden_dim, actual_n_modes, 2)
            
            # Reconstruct complex tensor
            h_fft_out = torch.view_as_complex(h_fft_filtered.contiguous())  # (batch, hidden_dim, actual_n_modes)
            
            # Pad back to original frequency size
            h_fft_full = torch.zeros_like(h_fft)  # (batch, hidden_dim, input_len//2+1)
            h_fft_full[:, :, :self.actual_n_modes] = h_fft_out  # (batch, hidden_dim, input_len//2+1)
            
            # Inverse FFT back to time domain
            h_spec = torch.fft.irfft(h_fft_full, n=h.size(-1), dim=-1)  # (batch, hidden_dim, input_len)
            
            # Multi-scale frequency gating: separate low and high freq info
            # Use h_fft_trunc for consistent mode count
            # Low frequencies (first half of actual modes)
            h_fft_low = torch.zeros_like(h_fft_trunc)  # (batch, hidden_dim, actual_n_modes)
            low_modes = self.actual_n_modes // 2
            h_fft_low[:, :, :low_modes] = h_fft_trunc[:, :, :low_modes]
            # Pad to full FFT size for IFFT
            h_fft_low_full = torch.zeros_like(h_fft)
            h_fft_low_full[:, :, :self.actual_n_modes] = h_fft_low
            h_low = torch.fft.irfft(h_fft_low_full, n=h.size(-1), dim=-1)  # (batch, hidden_dim, input_len)
            
            # High frequencies (remaining modes)
            h_fft_high = torch.zeros_like(h_fft_trunc)  # (batch, hidden_dim, actual_n_modes)
            h_fft_high[:, :, low_modes:self.actual_n_modes] = h_fft_trunc[:, :, low_modes:self.actual_n_modes]
            # Pad to full FFT size for IFFT
            h_fft_high_full = torch.zeros_like(h_fft)
            h_fft_high_full[:, :, :self.actual_n_modes] = h_fft_high
            h_high = torch.fft.irfft(h_fft_high_full, n=h.size(-1), dim=-1)  # (batch, hidden_dim, input_len)
            
            # Concatenate low/high stats for gating
            h_low_mean = h_low.mean(dim=-1, keepdim=True)  # (batch, hidden_dim, 1)
            h_high_mean = h_high.mean(dim=-1, keepdim=True)  # (batch, hidden_dim, 1)
            h_freq_stats = torch.cat([h_low_mean, h_high_mean], dim=1)  # (batch, hidden_dim*2, 1)
            h_freq_stats = h_freq_stats.squeeze(-1)  # (batch, hidden_dim*2)
            
            # Compute frequency gate and apply
            freq_gate = self.freq_gates[i](h_freq_stats)  # (batch, hidden_dim)
            freq_gate = freq_gate.unsqueeze(-1)  # (batch, hidden_dim, 1)
            h_gated = freq_gate * h_low + (1 - freq_gate) * h_high  # (batch, hidden_dim, input_len)
            
            # Combine with direct spectral path
            h = h_spec + h_gated  # (batch, hidden_dim, input_len)
            
            # Temporal MLP mixing
            h = self.temporal_mlps[i](h)  # (batch, hidden_dim, input_len)
            
            # Residual and norm
            h = h + h_in  # (batch, hidden_dim, input_len)
            h = h.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
            h = self.norms[i](h)  # (batch, input_len, hidden_dim)
            h = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
            h = self.dropout_layer(h)  # (batch, hidden_dim, input_len)
        
        # Prepare for decoder: (batch, input_len, hidden_dim)
        h = h.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
        
        # Compute skip connection
        h_skip = h_skip.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
        skip = self.skip_proj(h_skip)  # (batch, input_len, decoder_dim)
        
        # Stage 1: SiLU-based feature-wise gating (SwiGLU-like)
        gate_expand = self.decoder_gate_expand(h)  # (batch, input_len, decoder_dim * gate_ratio)
        gate_proj = self.decoder_gate_proj(h)  # (batch, input_len, decoder_dim * gate_ratio)
        gate_activated = F.silu(gate_expand) * gate_proj  # (batch, input_len, decoder_dim * gate_ratio)
        h_dec = self.decoder_gate_combine(gate_activated)  # (batch, input_len, decoder_dim)
        
        # Add skip connection
        h_dec = h_dec + skip  # (batch, input_len, decoder_dim)
        
        # Stage 2: Cross-channel attention gating
        # Treat decoder_dim as channels, compute cross-channel attention
        q = self.cross_channel_query(h_dec)  # (batch, input_len, decoder_dim)
        k = self.cross_channel_key(h_dec)  # (batch, input_len, decoder_dim)
        v = self.cross_channel_value(h_dec)  # (batch, input_len, decoder_dim)
        
        # Scaled dot-product attention across channel dimension
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.decoder_dim ** 0.5)  # (batch, input_len, input_len)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, input_len, input_len)
        attn_out = torch.bmm(attn_weights, v)  # (batch, input_len, decoder_dim)
        
        # Channel gate: learn which channels to emphasize
        channel_gate = self.cross_channel_gate(attn_out)  # (batch, input_len, decoder_dim)
        h_gated = channel_gate * attn_out + (1 - channel_gate) * h_dec  # (batch, input_len, decoder_dim)
        
        # Norm and residual
        h_dec = self.cross_channel_norm(h_dec + h_gated)  # (batch, input_len, decoder_dim)
        h_dec = self.dropout_layer(h_dec)  # (batch, input_len, decoder_dim)
        
        # Temporal projection to output length
        h_out = h_dec.permute(0, 2, 1)  # (batch, decoder_dim, input_len)
        h_out = self.output_temporal(h_out)  # (batch, decoder_dim, output_len)
        h_out = h_out.permute(0, 2, 1)  # (batch, output_len, decoder_dim)
        
        # Feature projection to original features
        out = self.output_feature(h_out)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)

