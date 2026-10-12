import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveWavelet_Fixed2_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_scales': [2, 4, 8],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3],
            'bottleneck_ratio': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_scales = self.config.get('n_scales', 4)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout = self.config.get('dropout', 0.1)
        self.bottleneck_ratio = self.config.get('bottleneck_ratio', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Period-aware positional encoding (370-day dominant period)
        self.period = 370.0
        self.n_freqs = 16
        self.period_embed = nn.Parameter(torch.randn(self.n_freqs * 2, self.hidden_dim) * 0.02)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Wavelet decomposition with learnable filters (Haar-like)
        self.register_buffer('wavelet_lo', torch.tensor([0.5, 0.5]).float().view(1, 1, 2))
        self.register_buffer('wavelet_hi', torch.tensor([0.5, -0.5]).float().view(1, 1, 2))
        
        # Channel-adaptive soft thresholding parameters - now on hidden_dim
        self.threshold_scale = nn.Parameter(torch.ones(1, 1, self.hidden_dim) * 0.1)
        self.threshold_shift = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        
        # Hierarchical cross-scale fusion modules
        self.scale_fusions = nn.ModuleList()
        self.scale_norms = nn.ModuleList()
        for i in range(self.n_scales - 1):
            # Bottleneck fusion: 2*hidden -> hidden//ratio -> hidden
            bottleneck_dim = max(self.hidden_dim // self.bottleneck_ratio, 16)
            self.scale_fusions.append(nn.Sequential(
                nn.Linear(self.hidden_dim * 2, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(bottleneck_dim, self.hidden_dim),
            ))
            self.scale_norms.append(nn.LayerNorm(self.hidden_dim))
        
        # Period-aware temporal attention
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, self.n_heads, dropout=self.dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.hidden_dim)
        
        # Feed-forward with bottleneck
        ff_bottleneck = max(self.hidden_dim // self.bottleneck_ratio, 16)
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, ff_bottleneck),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(ff_bottleneck, self.hidden_dim),
        )
        self.ff_norm = nn.LayerNorm(self.hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        
        # Direct residual connection from input
        self.residual_proj = nn.Linear(self.input_len, self.output_len)

    def _get_period_encoding(self, seq_len, device):
        # Sinusoidal encodings based on 370-day period
        positions = torch.arange(seq_len, device=device).float().unsqueeze(1)  # (seq_len, 1)
        freqs = torch.arange(self.n_freqs, device=device).float()  # (n_freqs,)
        # Multiple harmonics of base period
        periods = self.period / (freqs + 1)  # (n_freqs,)
        
        angles = positions / periods.unsqueeze(0) * 2 * 3.14159  # (seq_len, n_freqs)
        
        sin_enc = torch.sin(angles)  # (seq_len, n_freqs)
        cos_enc = torch.cos(angles)  # (seq_len, n_freqs)
        enc = torch.cat([sin_enc, cos_enc], dim=-1)  # (seq_len, n_freqs * 2)
        
        # Project to hidden_dim
        return enc @ self.period_embed  # (seq_len, hidden_dim)

    def _wavelet_decompose(self, x, n_levels):
        # x: (batch, seq_len, hidden_dim)
        coeffs = []
        current = x
        for level in range(n_levels):
            if current.shape[1] < 2:
                break
            # Pad for even length
            if current.shape[1] % 2 == 1:
                current = torch.cat([current, current[:, -1:, :]], dim=1)  # (batch, seq_len+1, hidden_dim)
            
            # Reshape for 1D conv: (batch*hidden_dim, 1, seq_len)
            b, s, h = current.shape
            current_t = current.permute(0, 2, 1).reshape(b * h, 1, s)  # (batch*hidden_dim, 1, seq_len)
            
            # Apply filters with proper padding
            pad = 1
            current_t = F.pad(current_t, (pad, 0), mode='reflect')
            
            lo = F.conv1d(current_t, self.wavelet_lo.to(current_t.device), stride=2)  # (batch*hidden_dim, 1, seq_len//2)
            hi = F.conv1d(current_t, self.wavelet_hi.to(current_t.device), stride=2)  # (batch*hidden_dim, 1, seq_len//2)
            
            # Reshape back
            approx = lo.view(b, h, -1).permute(0, 2, 1)  # (batch, seq_len//2, hidden_dim)
            detail = hi.view(b, h, -1).permute(0, 2, 1)  # (batch, seq_len//2, hidden_dim)
            
            coeffs.append((approx, detail))
            current = approx
        
        return coeffs

    def _soft_threshold(self, x):
        # Channel-adaptive soft thresholding
        # x: (batch, seq_len, hidden_dim)
        # Compute adaptive threshold per channel based on statistics
        threshold = F.softplus(self.threshold_scale) + 0.01  # Ensure positive, shape (1, 1, hidden_dim)
        threshold = threshold + self.threshold_shift  # shape (1, 1, hidden_dim)
        
        # Apply soft thresholding: sign(x) * max(|x| - threshold, 0)
        # Both x and threshold now have compatible shapes for broadcasting
        return torch.sign(x) * F.relu(torch.abs(x) - threshold)  # (batch, seq_len, hidden_dim)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save for residual
        x_input = x  # (batch, input_len, n_features)
        
        # Project to hidden dimension
        h = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Wavelet decomposition
        wavelet_coeffs = self._wavelet_decompose(h, self.n_scales)  # list of (approx, detail) tuples
        
        # Apply channel-adaptive soft thresholding to detail coefficients
        processed_coeffs = []
        for i, (approx, detail) in enumerate(wavelet_coeffs):
            # Threshold detail coefficients
            detail_thresh = self._soft_threshold(detail)  # (batch, seq_len//2^(i+1), hidden_dim)
            processed_coeffs.append((approx, detail_thresh))
        
        # Hierarchical cross-scale fusion (from coarse to fine)
        if len(processed_coeffs) > 0:
            # Start with coarsest approximation
            fused = processed_coeffs[-1][0]  # (batch, coarsest_len, hidden_dim)
            
            # Progressively fuse with finer scales
            for i in range(len(processed_coeffs) - 2, -1, -1):
                approx, detail = processed_coeffs[i]
                
                # Upsample fused to match current scale if needed
                if fused.shape[1] != approx.shape[1]:
                    fused = F.interpolate(
                        fused.permute(0, 2, 1),  # (batch, hidden_dim, fused_len)
                        size=approx.shape[1],
                        mode='linear',
                        align_corners=False
                    ).permute(0, 2, 1)  # (batch, approx_len, hidden_dim)
                
                # Combine approximation and upsampled coarse
                combined = torch.cat([approx, fused], dim=-1)  # (batch, approx_len, hidden_dim*2)
                fused = self.scale_fusions[i](combined)  # (batch, approx_len, hidden_dim)
                fused = self.scale_norms[i](fused + approx)  # Residual + norm
                
                # Add detail information
                if detail.shape[1] == fused.shape[1]:
                    fused = fused + detail  # (batch, approx_len, hidden_dim)
        else:
            fused = h  # (batch, input_len, hidden_dim)
        
        # Interpolate to output length if needed
        if fused.shape[1] != self.output_len:
            fused = F.interpolate(
                fused.permute(0, 2, 1),  # (batch, hidden_dim, fused_len)
                size=self.output_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Period-aware temporal attention
        period_enc = self._get_period_encoding(self.output_len, fused.device)  # (output_len, hidden_dim)
        period_enc = period_enc.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, output_len, hidden_dim)
        
        # Add period encoding
        fused_with_period = fused + period_enc  # (batch, output_len, hidden_dim)
        
        # Self-attention
        attn_out, _ = self.attention(fused_with_period, fused_with_period, fused_with_period)  # (batch, output_len, hidden_dim)
        attn_out = self.attn_norm(attn_out + fused_with_period)  # Residual + norm
        
        # Feed-forward
        ff_out = self.ff(attn_out)  # (batch, output_len, hidden_dim)
        ff_out = self.ff_norm(ff_out + attn_out)  # Residual + norm
        
        # Output projection
        out = self.output_proj(ff_out)  # (batch, output_len, n_features)
        
        # Direct residual connection
        x_res = x_input.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_res = self.residual_proj(x_res)  # (batch, n_features, output_len)
        x_res = x_res.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = out + x_res  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
