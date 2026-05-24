import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class WaveletMLPMixer(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_scales': [2, 3, 4],
            'dropout': [0.0, 0.1, 0.3],
            'n_mixer_layers': [1, 2, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_scales = self.config.get('n_scales', 3)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_mixer_layers = self.config.get('n_mixer_layers', 2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Learnable wavelet filters (Haar-like with learnable parameters)
        self.wavelet_filters = nn.ParameterList([
            nn.Parameter(torch.randn(2, 1) * 0.1) for _ in range(self.n_scales)
        ])
        
        # Projection to hidden dim
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # MLP-Mixer blocks for each scale
        self.mixer_blocks = nn.ModuleList()
        for _ in range(self.n_mixer_layers):
            self.mixer_blocks.append(
                nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(self.hidden_dim),
                        nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                        nn.Dropout(self.dropout)
                    ) for _ in range(self.n_scales)
                ])
            )
        
        # Cross-scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * self.n_scales, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        
        # Temporal decoder
        decoder_len = input_len // (2 ** self.n_scales) + sum([input_len // (2 ** (i+1)) for i in range(self.n_scales)])
        self.temporal_decoder = nn.Linear(decoder_len, output_len)
        
    def wavelet_decompose(self, x, scale_idx):
        # x: (batch, seq_len, hidden_dim)
        B, T, D = x.shape  # (batch, seq_len, hidden_dim)
        filter_k = self.wavelet_filters[scale_idx]  # (2, 1)
        
        # Apply along time dimension for each feature
        x_reshaped = x.permute(0, 2, 1).unsqueeze(2)  # (batch, hidden_dim, 1, seq_len)
        
        # Simple downsampling with learnable weights
        stride = 2 ** (scale_idx + 1)
        kernel_size = 2
        
        # Approximation (low-pass)
        approx = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=kernel_size, stride=2)  # (batch, hidden_dim, T//2)
        # Detail (high-pass via difference)
        detail = x.permute(0, 2, 1)[:, :, 1::2] - x.permute(0, 2, 1)[:, :, 0::2]  # (batch, hidden_dim, T//2)
        
        approx = approx.permute(0, 2, 1)  # (batch, T//2, hidden_dim)
        detail = detail.permute(0, 2, 1)  # (batch, T//2, hidden_dim)
        
        return approx, detail
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden dimension
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Multi-scale decomposition
        scale_features = []
        current = x  # (batch, input_len, hidden_dim)
        
        for i in range(self.n_scales):
            approx, detail = self.wavelet_decompose(current, i)  # both (batch, T//2, hidden_dim)
            scale_features.append(approx)
            scale_features.append(detail)
            current = approx  # (batch, T//2, hidden_dim)
        
        # Apply MLP-Mixer to each scale
        for mixer_layer in self.mixer_blocks:
            new_features = []
            for j, (feat, mixer) in enumerate(zip(scale_features, mixer_layer)):
                # Channel mixing (across hidden_dim)
                feat = feat + mixer(feat)  # (batch, T_j, hidden_dim)
                new_features.append(feat)
            scale_features = new_features
        
        # Interpolate all scales to same length and concatenate
        target_len = self.input_len
        aligned_features = []
        for feat in scale_features:
            if feat.shape[1] != target_len:
                feat = F.interpolate(feat.permute(0, 2, 1), size=target_len, mode='linear', align_corners=False).permute(0, 2, 1)  # (batch, target_len, hidden_dim)
            aligned_features.append(feat)  # (batch, target_len, hidden_dim)
        
        # Concatenate along feature dimension
        fused = torch.cat(aligned_features, dim=-1)  # (batch, target_len, hidden_dim * n_scales * 2)
        
        # Fuse scales
        fused = self.scale_fusion(fused)  # (batch, target_len, hidden_dim)
        
        # Project to output features
        out = self.output_proj(fused)  # (batch, target_len, n_features)
        
        # Temporal decoding to output length
        out = out.permute(0, 2, 1)  # (batch, n_features, target_len)
        out = self.temporal_decoder(out)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

