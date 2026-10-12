import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class WaveletMLPMixer_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_scales': [2, 3, 4],
            'dropout': [0.0, 0.3],
            'n_mixer_layers': [1, 3, 6]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_scales = self.config.get('n_scales', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.n_mixer_layers = self.config.get('n_mixer_layers', 3)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Learnable wavelet filters (simplified as 1D convolutions with constraints)
        self.wavelet_filters = nn.ModuleList()
        self.scale_lengths = []
        scale_len = input_len
        
        for i in range(self.n_scales):
            # Ensure minimum sequence length for valid convolution
            if scale_len < 4:
                break
            # Filter length: small odd-sized filters for stable wavelet behavior
            filter_len = min(6, scale_len - 1)
            if filter_len % 2 == 0:
                filter_len -= 1
            filter_len = max(3, filter_len)
            
            # Use stride=2 with careful padding to ensure valid output
            padding = filter_len // 2
            # Compute output length: floor((L + 2*padding - filter_len) / 2) + 1
            out_len = (scale_len + 2 * padding - filter_len) // 2 + 1
            
            self.wavelet_filters.append(
                nn.Conv1d(
                    n_features, 
                    n_features * 2, 
                    filter_len, 
                    stride=2, 
                    padding=padding, 
                    groups=n_features, 
                    bias=False
                )
            )
            self.scale_lengths.append(out_len)
            scale_len = out_len
        
        # Update n_scales to actual number of valid scales
        self.n_scales = len(self.wavelet_filters)
        
        # Feature projection to hidden dim
        self.feature_proj = nn.Linear(n_features, self.hidden_dim)
        
        # MLP-Mixer layers: proper channel and temporal mixing
        self.mixer_layers = nn.ModuleList()
        for _ in range(self.n_mixer_layers):
            # Channel mixing: operates on hidden_dim (last dimension)
            # Temporal mixing: operates on time dimension using depthwise conv
            self.mixer_layers.append(nn.ModuleDict({
                'ln_channel': nn.LayerNorm(self.hidden_dim),
                'channel_expand': nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                'channel_act': nn.GELU(),
                'channel_drop': nn.Dropout(self.dropout_rate),
                'channel_squeeze': nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                'channel_drop2': nn.Dropout(self.dropout_rate),
                'ln_temporal': nn.LayerNorm(self.hidden_dim),
                'temporal_expand': nn.Conv1d(self.hidden_dim, self.hidden_dim * 4, 1, groups=1),
                'temporal_act': nn.GELU(),
                'temporal_drop': nn.Dropout(self.dropout_rate),
                'temporal_squeeze': nn.Conv1d(self.hidden_dim * 4, self.hidden_dim, 1, groups=1),
                'temporal_drop2': nn.Dropout(self.dropout_rate),
            }))
        
        # Scale mixing MLP
        self.scale_mixer = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, n_features)
        
        # Learnable upsampling for each scale to output_len
        self.upsamplers = nn.ModuleList()
        for sl in self.scale_lengths:
            self.upsamplers.append(nn.Sequential(
                nn.Linear(sl, output_len),
            ))
        
        self.scale_weights = nn.Parameter(torch.ones(self.n_scales))

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Wavelet decomposition
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        scale_features = []
        current = x_perm
        for i, wfilter in enumerate(self.wavelet_filters):
            # Apply filter and split into low/high
            filtered = wfilter(current)  # (batch, n_features*2, new_len)
            b, _, l = filtered.shape  # (batch, n_features*2, l)
            # Take low-pass (approximation) for next level
            low = filtered[:, :self.n_features, :]  # (batch, n_features, l)
            high = filtered[:, self.n_features:, :]  # (batch, n_features, l)
            
            # Use both for representation
            combined = low + high  # (batch, n_features, l)
            scale_features.append(combined)
            current = low  # (batch, n_features, l)
        
        # Process each scale with MLP-Mixer
        scale_outputs = []
        for i, (feat, sl, upsampler) in enumerate(zip(scale_features, self.scale_lengths, self.upsamplers)):
            # feat: (batch, n_features, sl)
            feat = feat.permute(0, 2, 1)  # (batch, sl, n_features)
            feat = self.feature_proj(feat)  # (batch, sl, hidden_dim)
            
            # Apply MLP-Mixer layers
            for mixer in self.mixer_layers:
                # Channel mixing: operates on last dimension (hidden_dim)
                residual = feat  # (batch, sl, hidden_dim)
                feat = mixer['ln_channel'](feat)  # (batch, sl, hidden_dim)
                feat = mixer['channel_expand'](feat)  # (batch, sl, hidden_dim*4)
                feat = mixer['channel_act'](feat)  # (batch, sl, hidden_dim*4)
                feat = mixer['channel_drop'](feat)  # (batch, sl, hidden_dim*4)
                feat = mixer['channel_squeeze'](feat)  # (batch, sl, hidden_dim)
                feat = mixer['channel_drop2'](feat)  # (batch, sl, hidden_dim)
                feat = residual + feat  # (batch, sl, hidden_dim)
                
                # Temporal mixing: operates on time dimension using 1x1 conv
                residual = feat  # (batch, sl, hidden_dim)
                feat = mixer['ln_temporal'](feat)  # (batch, sl, hidden_dim)
                feat = feat.permute(0, 2, 1)  # (batch, hidden_dim, sl)
                feat = mixer['temporal_expand'](feat)  # (batch, hidden_dim*4, sl)
                feat = mixer['temporal_act'](feat)  # (batch, hidden_dim*4, sl)
                feat = mixer['temporal_drop'](feat)  # (batch, hidden_dim*4, sl)
                feat = mixer['temporal_squeeze'](feat)  # (batch, hidden_dim, sl)
                feat = mixer['temporal_drop2'](feat)  # (batch, hidden_dim, sl)
                feat = feat.permute(0, 2, 1)  # (batch, sl, hidden_dim)
                feat = residual + feat  # (batch, sl, hidden_dim)
            
            # Upsample to output_len
            feat = feat.permute(0, 2, 1)  # (batch, hidden_dim, sl)
            feat = upsampler(feat)  # (batch, hidden_dim, output_len)
            feat = feat.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
            scale_outputs.append(feat)
        
        # Weighted combination of scales
        if len(scale_outputs) > 0:
            weights = F.softmax(self.scale_weights[:self.n_scales], dim=0)  # (n_scales,)
            combined = sum(w * out for w, out in zip(weights, scale_outputs))  # (batch, output_len, hidden_dim)
        else:
            # Fallback: use zero tensor with correct shape if no scales
            combined = torch.zeros(x.size(0), self.output_len, self.hidden_dim, device=x.device)  # (batch, output_len, hidden_dim)
        
        # Final scale mixing
        combined = self.scale_mixer(combined)  # (batch, output_len, hidden_dim)
        
        # Output projection
        out = self.output_proj(combined)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

