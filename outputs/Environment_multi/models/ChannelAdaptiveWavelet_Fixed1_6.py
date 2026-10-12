import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveWavelet_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'wavelet_hidden': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3],
            'n_wavelet_layers': [1, 4, 8],
            'attention_dim': [64, 256, 1024]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.wavelet_hidden = self.config.get('wavelet_hidden', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_wavelet_layers = self.config.get('n_wavelet_layers', 4)
        self.attention_dim = self.config.get('attention_dim', 256)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Channel-adaptive threshold parameters
        self.threshold = nn.Parameter(torch.ones(self.n_features, 1) * 0.1)  # (n_features, 1)
        
        # Multi-scale wavelet processing layers
        self.wavelet_convs = nn.ModuleList()
        self.scale_gates = nn.ModuleList()
        for i in range(self.n_wavelet_layers):
            # Approximation path
            self.wavelet_convs.append(nn.Sequential(
                nn.Conv1d(self.n_features, self.wavelet_hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Conv1d(self.wavelet_hidden, self.n_features, kernel_size=3, padding=1)
            ))
            # Gating for cross-scale fusion: detail * sigmoid(gate)
            self.scale_gates.append(nn.Sequential(
                nn.Conv1d(self.n_features, self.wavelet_hidden, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(self.wavelet_hidden, self.n_features, kernel_size=1),
                nn.Sigmoid()
            ))
        
        # Multi-scale temporal attention
        self.attention_qkv = nn.Linear(self.n_features, self.attention_dim * 3)
        self.attention_out = nn.Linear(self.attention_dim, self.n_features)
        self.attention_norm = nn.LayerNorm(self.n_features)
        
        # Scale combination weights
        self.scale_weights = nn.Parameter(torch.ones(self.n_wavelet_layers + 1))
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(self.n_features, self.wavelet_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.wavelet_hidden, self.n_features)
        )
        
        # Temporal decoder for variable-length output
        self.temporal_decoder = nn.Linear(input_len, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode="norm")  # (batch, input_len, n_features)
        
        # Store original for residual
        x_orig = x  # (batch, input_len, n_features)
        
        # Multi-scale wavelet decomposition and processing
        # Approximation starts as input
        approx = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        scale_outputs = []
        for i in range(self.n_wavelet_layers):
            # Simple downsampling as wavelet approximation (average pooling)
            if approx.shape[-1] > 1:
                approx_next = F.avg_pool1d(approx, kernel_size=2, stride=2)  # (batch, n_features, len//2)
                # Detail is difference from upsampled approximation
                detail = approx - F.interpolate(approx_next, size=approx.shape[-1], mode='linear', align_corners=False)  # (batch, n_features, input_len)
            else:
                approx_next = approx
                detail = torch.zeros_like(approx)
            
            # Apply channel-adaptive thresholding to detail with correct broadcasting
            # threshold: (n_features, 1) -> (1, 1, n_features) for broadcasting across (batch, time, features)
            threshold_expanded = self.threshold.permute(1, 0).unsqueeze(0)  # (1, 1, n_features)
            detail_perm = detail.permute(0, 2, 1)  # (batch, len, n_features)
            detail_thresholded = torch.where(
                torch.abs(detail_perm) > threshold_expanded,
                detail_perm,
                torch.zeros_like(detail_perm)
            )  # (batch, len, n_features)
            detail_thresholded = detail_thresholded.permute(0, 2, 1)  # (batch, n_features, len)
            
            # Process detail with gating
            detail_processed = self.wavelet_convs[i](detail_thresholded)  # (batch, n_features, len)
            gate = self.scale_gates[i](detail_thresholded)  # (batch, n_features, len)
            detail_fused = detail_processed * gate  # (batch, n_features, len)
            
            # Upsample detail to original length
            if detail_fused.shape[-1] != x.shape[1]:
                detail_fused = F.interpolate(detail_fused, size=x.shape[1], mode='linear', align_corners=False)  # (batch, n_features, input_len)
            
            scale_outputs.append(detail_fused)
            approx = approx_next
        
        # Upsample final approximation to original length
        if approx.shape[-1] != x.shape[1]:
            approx = F.interpolate(approx, size=x.shape[1], mode='linear', align_corners=False)  # (batch, n_features, input_len)
        scale_outputs.append(approx)
        
        # Weighted combination of scales
        weights = F.softmax(self.scale_weights, dim=0)  # (n_wavelet_layers + 1,)
        fused = torch.zeros_like(scale_outputs[0])  # (batch, n_features, input_len)
        for i, scale_out in enumerate(scale_outputs):
            fused = fused + weights[i] * scale_out  # (batch, n_features, input_len)
        
        fused = fused.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Add residual
        fused = fused + x_orig  # (batch, input_len, n_features)
        
        # Multi-scale temporal self-attention
        attn_in = self.attention_norm(fused)  # (batch, input_len, n_features)
        qkv = self.attention_qkv(attn_in)  # (batch, input_len, attention_dim * 3)
        q, k, v = qkv.chunk(3, dim=-1)  # each (batch, input_len, attention_dim)
        
        # Multi-head attention
        batch_size, seq_len, _ = q.shape
        head_dim = self.attention_dim // self.n_heads
        
        q = q.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)  # (batch, n_heads, input_len, head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)  # (batch, n_heads, input_len, head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)  # (batch, n_heads, input_len, head_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # (batch, n_heads, input_len, input_len)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, input_len, input_len)
        attn_out = torch.matmul(attn_weights, v)  # (batch, n_heads, input_len, head_dim)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.attention_dim)  # (batch, input_len, attention_dim)
        attn_out = self.attention_out(attn_out)  # (batch, input_len, n_features)
        
        # Residual connection after attention
        attended = fused + attn_out  # (batch, input_len, n_features)
        
        # Final projection per time step
        projected = self.projection(attended)  # (batch, input_len, n_features)
        
        # Temporal decoder: project from input_len to output_len
        projected = projected.permute(0, 2, 1)  # (batch, n_features, input_len)
        out = self.temporal_decoder(projected)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode="denorm")  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
