import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveWavelet_Fixed1_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'wavelet': ['db2', 'sym4', 'coif2'],
            'level': [2, 3, 4],
            'hidden_dim': [64, 256, 1024],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.wavelet = self.config.get('wavelet', 'db2')
        self.level = self.config.get('level', 3)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        # Initialize wavelet filter bank parameters (learnable)
        self._init_wavelet_filters()
        
        # Compute decomposition structure
        self._compute_decomposition_structure()
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Channel-adaptive threshold: MLP on per-channel statistics
        self.threshold_net = nn.Sequential(
            nn.Linear(2, 32),  # input: mean and std of wavelet coeffs
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Learnable cross-scale attention gates
        self.scale_attn = nn.ModuleList()
        for i in range(self.level + 1):  # approximation + detail levels
            # Each scale gets its own attention projection
            self.scale_attn.append(nn.Sequential(
                nn.Linear(self.n_features, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.n_features)
            ))
        
        # Attention temperature
        self.attn_temp = nn.Parameter(torch.ones(1))
        
        # Channel-separated temporal convolutions for each scale
        self.scale_convs = nn.ModuleList()
        for i in range(self.level + 1):
            convs = nn.ModuleList()
            for f in range(self.n_features):
                convs.append(nn.Sequential(
                    nn.Conv1d(1, self.hidden_dim // 4, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Conv1d(self.hidden_dim // 4, self.hidden_dim // 4, kernel_size=3, padding=1),
                    nn.ReLU(),
                ))
            self.scale_convs.append(convs)
        
        # Feature-wise projection per scale
        self.scale_proj = nn.ModuleList()
        for i in range(self.level + 1):
            self.scale_proj.append(nn.Linear(self.hidden_dim // 4, self.n_features))
        
        # Cross-scale fusion with attention
        self.fusion_attn = nn.Sequential(
            nn.Linear((self.level + 1) * self.n_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.level + 1)
        )
        
        # Final predictor: channel-separated MLP
        total_scale_features = (self.level + 1) * self.n_features
        self.predictor = nn.ModuleList()
        for f in range(self.n_features):
            self.predictor.append(nn.Sequential(
                nn.Linear(total_scale_features, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.output_len)
            ))
        
        # Residual: direct approximation-to-output skip
        self.approx_skip = nn.Linear(self.approx_len, self.output_len) if self.approx_len != self.output_len else nn.Identity()
        
    def _init_wavelet_filters(self):
        # Approximate wavelet filter coefficients for common wavelets
        # These are learnable parameters initialized to approximate the wavelet
        if self.wavelet == 'db2':
            # Daubechies-2 (db2) low-pass filter (4 taps)
            h0_init = torch.tensor([0.482962913145, 0.836516303738, 
                                    0.224143868042, -0.129409522551])
        elif self.wavelet == 'sym4':
            # Symlet-4 (sym4) low-pass filter (8 taps)
            h0_init = torch.tensor([0.0322, -0.0126, -0.0992, 0.2979,
                                    0.8037, 0.4976, -0.0296, -0.0758])
        elif self.wavelet == 'coif2':
            # Coiflet-2 (coif2) low-pass filter (12 taps)
            h0_init = torch.tensor([-0.0157, -0.0727, 0.3849, 0.8526,
                                    0.3379, -0.0727, -0.0157, 0.0000,
                                    0.0000, 0.0000, 0.0000, 0.0000])
            h0_init = h0_init[:8]  # Use first 8 for simplicity
        else:
            # Default to db2
            h0_init = torch.tensor([0.482962913145, 0.836516303738, 
                                    0.224143868042, -0.129409522551])
        
        # Ensure length is even
        if len(h0_init) % 2 == 1:
            h0_init = torch.cat([h0_init, torch.zeros(1)])
        
        self.filter_len = len(h0_init)
        
        # Compute high-pass filter from low-pass (QMF: alternating flip)
        # h1[n] = (-1)^n * h0[L-1-n]
        h1_init = torch.zeros_like(h0_init)
        for i in range(self.filter_len):
            h1_init[i] = ((-1) ** i) * h0_init[self.filter_len - 1 - i]
        
        # Make learnable parameters
        self.h0 = nn.Parameter(h0_init.float().view(1, 1, -1))  # low-pass decomposition
        self.h1 = nn.Parameter(h1_init.float().view(1, 1, -1))  # high-pass decomposition
        
    def _compute_decomposition_structure(self):
        # Compute expected lengths after decomposition
        temp_len = self.input_len
        self.detail_lens = []
        for i in range(self.level):
            # With proper padding, length after DWT with stride 2
            # For 'symmetric' equivalent padding: output = ceil(input / 2)
            temp_len = (temp_len + self.filter_len - 1) // 2
            self.detail_lens.append(temp_len)
        self.approx_len = temp_len
        
    def _dwt_single_level(self, x):
        # x: (batch, n_features, seq_len)
        batch_size, n_feat, seq_len = x.shape
        
        # Pad symmetrically (equivalent to pywt's 'symmetric' mode)
        pad_total = self.filter_len - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_padded = F.pad(x, (pad_left, pad_right), mode='reflect')  # (batch, n_features, seq_len + pad)
        
        # Apply filters per feature
        # We need to convolve each feature independently
        approx = []
        detail = []
        
        for f in range(n_feat):
            xf = x_padded[:, f:f+1, :]  # (batch, 1, seq_len + pad)
            
            # Low-pass (approximation)
            a = F.conv1d(xf, self.h0, stride=2)  # (batch, 1, ceil(seq_len/2))
            # High-pass (detail)
            d = F.conv1d(xf, self.h1, stride=2)  # (batch, 1, ceil(seq_len/2))
            
            approx.append(a)
            detail.append(d)
        
        approx = torch.cat(approx, dim=1)  # (batch, n_features, approx_len)
        detail = torch.cat(detail, dim=1)  # (batch, n_features, detail_len)
        
        return approx, detail
    
    def _dwt_decompose(self, x):
        # x: (batch, input_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        coeffs = []
        current = x
        
        for i in range(self.level):
            approx, detail = self._dwt_single_level(current)
            coeffs.append(detail)  # Store detail coefficients
            current = approx  # Continue with approximation
        
        coeffs.append(current)  # Final approximation
        coeffs = coeffs[::-1]  # Reverse: [approx, detail_level, ..., detail_1]
        
        # Convert to list of (batch, len_i, n_features)
        result = []
        for c in coeffs:
            result.append(c.permute(0, 2, 1))  # (batch, len_i, n_features)
        
        return result  # [approx, detail_level, ..., detail_1]
    
    def _soft_threshold(self, x, threshold):
        # x: (batch, len, n_features)
        # threshold: computed per channel, shape should broadcast to (1, 1, n_features)
        return torch.sign(x) * F.relu(torch.abs(x) - threshold)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Wavelet decomposition (PyTorch-based)
        scales = self._dwt_decompose(x)  # list of (batch, len_i, n_features)
        
        # Channel-adaptive thresholding with learnable threshold
        processed_scales = []
        for i, scale in enumerate(scales):
            # Compute per-channel statistics for threshold
            scale_mean = scale.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
            scale_std = scale.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
            
            # Threshold network: process each channel independently
            stats_per_channel = torch.stack([scale_mean.squeeze(1), scale_std.squeeze(1)], dim=-1)  # (batch, n_features, 2)
            
            # Apply MLP per channel
            thresh_vals = []
            for f in range(self.n_features):
                f_stats = stats_per_channel[:, f, :]  # (batch, 2)
                f_thresh = self.threshold_net(f_stats)  # (batch, 1)
                thresh_vals.append(f_thresh)
            threshold = torch.stack(thresh_vals, dim=1)  # (batch, n_features, 1)
            threshold = threshold.permute(0, 2, 1)  # (batch, 1, n_features)
            
            # Apply soft thresholding
            if i > 0:  # threshold detail coefficients
                scale = self._soft_threshold(scale, threshold)  # (batch, len_i, n_features)
            
            # Learnable scale attention gate
            attn_input = scale.mean(dim=1)  # (batch, n_features)
            attn = torch.sigmoid(self.scale_attn[i](attn_input) / self.attn_temp)  # (batch, n_features)
            scale = scale * attn.unsqueeze(1)  # (batch, len_i, n_features)
            
            processed_scales.append(scale)
        
        # Channel-separated temporal convolution per scale
        conv_outputs = []
        for i, scale in enumerate(processed_scales):
            # scale: (batch, len_i, n_features)
            scale_len = scale.shape[1]
            
            # Process each feature with its own conv
            feat_outputs = []
            for f in range(self.n_features):
                sf = scale[:, :, f:f+1]  # (batch, len_i, 1)
                sf = sf.permute(0, 2, 1)  # (batch, 1, len_i)
                
                # Interpolate to common length if needed
                if scale_len != self.approx_len:
                    sf = F.interpolate(sf, size=self.approx_len, mode='linear', align_corners=False)  # (batch, 1, approx_len)
                
                conv_out = self.scale_convs[i][f](sf)  # (batch, hidden_dim//4, approx_len)
                conv_out = conv_out.mean(dim=2)  # (batch, hidden_dim//4) - global average pool
                feat_outputs.append(conv_out)
            
            # Stack features
            stacked = torch.stack(feat_outputs, dim=1)  # (batch, n_features, hidden_dim//4)
            # Project each feature's representation - use mean for simplicity
            scale_repr = stacked.mean(dim=-1)  # (batch, n_features)
            conv_outputs.append(scale_repr)
        
        # Concatenate all scale representations
        fused = torch.cat(conv_outputs, dim=-1)  # (batch, (level+1) * n_features)
        
        # Cross-scale attention for dynamic fusion weighting
        fusion_weights = F.softmax(self.fusion_attn(fused), dim=-1)  # (batch, level+1)
        
        # Weighted combination of scale representations
        weighted_scales = []
        for i in range(self.level + 1):
            w = fusion_weights[:, i:i+1]  # (batch, 1)
            weighted_scales.append(conv_outputs[i] * w)  # (batch, n_features)
        fused_repr = torch.cat(weighted_scales, dim=-1)  # (batch, (level+1) * n_features)
        
        # Residual from approximation path
        approx_pooled = processed_scales[0].mean(dim=1)  # (batch, n_features)
        
        # Final prediction: channel-separated
        outputs = []
        for f in range(self.n_features):
            out = self.predictor[f](fused_repr)  # (batch, output_len)
            # Add residual from approximation
            out = out + approx_pooled[:, f:f+1].expand(-1, self.output_len)  # (batch, output_len)
            outputs.append(out)
        
        out = torch.stack(outputs, dim=-1)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
