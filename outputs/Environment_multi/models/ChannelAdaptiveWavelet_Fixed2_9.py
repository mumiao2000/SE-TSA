import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveWavelet_Fixed2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'wavelet_depth': [2, 4, 8],
            'hidden_dim': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3],
            'fusion_dim': [64, 256, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with wide ranges
        self.wavelet_depth = self.config.get('wavelet_depth', 4)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout = self.config.get('dropout', 0.1)
        self.fusion_dim = self.config.get('fusion_dim', 256)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Haar wavelet filters (fixed, not learnable for stability)
        self.register_buffer('low_pass', torch.tensor([0.5, 0.5]).float())
        self.register_buffer('high_pass', torch.tensor([0.5, -0.5]).float())
        
        # Channel-adaptive soft thresholding with learnable slopes
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, self.n_features)) 
            for _ in range(self.wavelet_depth)
        ])
        self.shrinkage_slopes = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, self.n_features) * 2.0)
            for _ in range(self.wavelet_depth)
        ])
        
        # Scale-specific feature projections
        self.scale_projs = nn.ModuleList([
            nn.Linear(self.n_features, self.hidden_dim)
            for _ in range(self.wavelet_depth + 1)  # +1 for approximation
        ])
        
        # Hierarchical cross-scale fusion: fuse adjacent scales progressively
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.fusion_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.fusion_dim, self.hidden_dim)
            )
            for _ in range(self.wavelet_depth)
        ])
        
        self.fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Sigmoid()
            )
            for _ in range(self.wavelet_depth)
        ])
        
        # Period-aware position encoding (370-day dominant period)
        self.period = 370.0
        self.pos_dim = self.hidden_dim
        inv_freq = 1.0 / (self.period ** (torch.arange(0, self.pos_dim, 2).float() / self.pos_dim))  # (pos_dim//2,)
        self.register_buffer('inv_freq', inv_freq)
        
        # Multi-head temporal attention with rotary position encoding
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, 
            self.n_heads, 
            dropout=self.dropout,
            batch_first=True
        )
        
        self.attn_ln = nn.LayerNorm(self.hidden_dim)
        self.attn_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Dropout(self.dropout)
        )
        self.ffn_ln = nn.LayerNorm(self.hidden_dim)
        
        # Output projection
        # Compute expected sequence length after wavelet decomposition
        self.expected_len = input_len // (2 ** self.wavelet_depth)
        if self.expected_len < 1:
            self.expected_len = 1
        
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_len * self.n_features)
        )
        
    def get_periodic_pos_encoding(self, seq_len, device):
        # Generate position encodings tuned to 370-day period
        positions = torch.arange(seq_len, device=device).float()  # (seq_len,)
        angles = torch.outer(positions, self.inv_freq)  # (seq_len, pos_dim//2)
        pos_enc = torch.zeros(seq_len, self.pos_dim, device=device)  # (seq_len, pos_dim)
        pos_enc[:, 0::2] = torch.sin(angles)
        pos_enc[:, 1::2] = torch.cos(angles)
        return pos_enc  # (seq_len, pos_dim)
    
    def apply_rotary_pos_emb(self, q, k, pos_enc):
        # q, k: (batch, n_heads, seq_len, head_dim)
        # pos_enc: (seq_len, pos_dim) where pos_dim = hidden_dim
        batch, n_heads, seq_len, head_dim = q.shape
        # Reshape pos_enc to match head_dim
        pos_enc = pos_enc[:seq_len, :head_dim]  # (seq_len, head_dim)
        cos = pos_enc[:, 0::2].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
        sin = pos_enc[:, 1::2].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
        
        # Apply rotation to q, k
        q_rot = torch.zeros_like(q)
        k_rot = torch.zeros_like(k)
        q_rot[:, :, :, 0::2] = q[:, :, :, 0::2] * cos - q[:, :, :, 1::2] * sin
        q_rot[:, :, :, 1::2] = q[:, :, :, 0::2] * sin + q[:, :, :, 1::2] * cos
        k_rot[:, :, :, 0::2] = k[:, :, :, 0::2] * cos - k[:, :, :, 1::2] * sin
        k_rot[:, :, :, 1::2] = k[:, :, :, 0::2] * sin + k[:, :, :, 1::2] * cos
        
        return q_rot, k_rot
    
    def haar_decompose(self, x):
        # x: (batch, seq_len, n_features)
        batch, seq_len, n_features = x.shape
        if seq_len % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1), mode='replicate')  # (batch, seq_len+1, n_features)
            seq_len = seq_len + 1
        
        # Reshape for 1D conv: (batch * n_features, 1, seq_len)
        x_perm = x.permute(0, 2, 1).reshape(batch * n_features, 1, seq_len)  # (batch*n_features, 1, seq_len)
        
        # Low-pass and high-pass filtering
        low = F.conv1d(x_perm, self.low_pass.view(1, 1, 2), stride=2)  # (batch*n_features, 1, seq_len//2)
        high = F.conv1d(x_perm, self.high_pass.view(1, 1, 2), stride=2)  # (batch*n_features, 1, seq_len//2)
        
        # Reshape back
        approx = low.view(batch, n_features, -1).permute(0, 2, 1)  # (batch, seq_len//2, n_features)
        detail = high.view(batch, n_features, -1).permute(0, 2, 1)  # (batch, seq_len//2, n_features)
        
        return approx, detail  # (batch, seq_len//2, n_features), (batch, seq_len//2, n_features)
    
    def soft_threshold(self, x, threshold, slope):
        # Channel-adaptive soft thresholding with learnable slope
        # x: (batch, seq_len, n_features)
        # threshold, slope: (1, 1, n_features)
        abs_x = torch.abs(x)  # (batch, seq_len, n_features)
        # Soft shrinkage: sign(x) * max(0, |x| - threshold) with soft transition via slope
        shrink = F.softplus(slope * (abs_x - threshold)) / slope  # (batch, seq_len, n_features)
        return torch.sign(x) * F.relu(abs_x - threshold + shrink - shrink.detach())  # (batch, seq_len, n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Multi-level wavelet decomposition
        coeffs = []  # List of detail coefficients at each scale
        approx = x  # (batch, input_len, n_features)
        
        for d in range(self.wavelet_depth):
            approx, detail = self.haar_decompose(approx)  # (batch, len//2, n_features), (batch, len//2, n_features)
            # Channel-adaptive thresholding on detail coefficients
            detail = self.soft_threshold(detail, self.thresholds[d], self.shrinkage_slopes[d])  # (batch, len//2, n_features)
            coeffs.append(detail)
        
        # Project approximation and all detail coefficients
        approx_proj = self.scale_projs[0](approx)  # (batch, approx_len, hidden_dim)
        scale_features = [approx_proj]  # List of (batch, scale_len, hidden_dim)
        
        for i, coeff in enumerate(coeffs):
            proj = self.scale_projs[i + 1](coeff)  # (batch, scale_len, hidden_dim)
            scale_features.append(proj)
        
        # Hierarchical fusion: from finest to coarsest scale
        # Start with finest detail, progressively fuse with coarser scales
        fused = scale_features[-1]  # (batch, finest_len, hidden_dim) - finest detail
        
        for i in range(len(scale_features) - 2, -1, -1):  # From second finest to approximation
            coarse = scale_features[i]  # (batch, coarse_len, hidden_dim)
            
            # Upsample fused to match coarse length if needed
            if fused.shape[1] != coarse.shape[1]:
                fused = F.interpolate(
                    fused.permute(0, 2, 1),  # (batch, hidden_dim, fine_len)
                    size=coarse.shape[1],
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)  # (batch, coarse_len, hidden_dim)
            
            # Concatenate and fuse
            combined = torch.cat([coarse, fused], dim=-1)  # (batch, coarse_len, hidden_dim*2)
            gate = self.fusion_gates[i](combined)  # (batch, coarse_len, hidden_dim)
            fused = gate * coarse + (1 - gate) * self.fusion_layers[i](combined)  # (batch, coarse_len, hidden_dim)
        
        # Final fused representation
        h = fused  # (batch, seq_len_fused, hidden_dim)
        
        # Period-aware temporal attention
        pos_enc = self.get_periodic_pos_encoding(h.shape[1], h.device)  # (seq_len_fused, pos_dim)
        
        # Self-attention with rotary position encoding
        h_attn = self.attn_ln(h)  # (batch, seq_len_fused, hidden_dim)
        
        # Manual multi-head attention with rotary embeddings
        batch, seq_len, _ = h_attn.shape
        qkv = h_attn  # Use same tensor for q, k, v for self-attention
        
        # Standard multi-head attention (PyTorch handles this, but we add pos enc to input)
        h_attn = h_attn + pos_enc.unsqueeze(0)  # (batch, seq_len_fused, hidden_dim)
        
        attn_out, _ = self.attention(h_attn, h_attn, h_attn)  # (batch, seq_len_fused, hidden_dim)
        h = h + attn_out  # (batch, seq_len_fused, hidden_dim) - residual
        
        # FFN
        h = h + self.attn_ffn(self.ffn_ln(h))  # (batch, seq_len_fused, hidden_dim)
        
        # Global average pooling over time
        h_pooled = h.mean(dim=1)  # (batch, hidden_dim)
        
        # Output projection
        out = self.output_proj(h_pooled)  # (batch, output_len * n_features)
        out = out.view(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
