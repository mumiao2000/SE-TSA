import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveWavelet_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_levels': [2, 3, 4],
            'dropout': [0.0, 0.3],
            'fusion_layers': [1, 2, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_levels = self.config.get('n_levels', 3)
        self.dropout = self.config.get('dropout', 0.1)
        self.fusion_layers = self.config.get('fusion_layers', 2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Ensure input_len is divisible by 2^n_levels
        self.padded_len = ((self.input_len + (2**self.n_levels) - 1) // (2**self.n_levels)) * (2**self.n_levels)
        
        # Learnable soft-threshold per channel per level
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, self.n_features) * 0.1) for _ in range(self.n_levels)
        ])
        
        # Detail coefficient fusion MLPs
        self.detail_fusion = nn.ModuleList()
        for _ in range(self.n_levels):
            layers = []
            for i in range(self.fusion_layers):
                in_d = self.n_features if i == 0 else self.hidden_dim
                out_d = self.hidden_dim if i < self.fusion_layers - 1 else self.n_features
                layers.append(nn.Linear(in_d, out_d))
                if i < self.fusion_layers - 1:
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(self.dropout))
            self.detail_fusion.append(nn.Sequential(*layers))
        
        # Approximation fusion
        layers = []
        for i in range(self.fusion_layers):
            in_d = self.n_features if i == 0 else self.hidden_dim
            out_d = self.hidden_dim if i < self.fusion_layers - 1 else self.n_features
            layers.append(nn.Linear(in_d, out_d))
            if i < self.fusion_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(self.dropout))
        self.approx_fusion = nn.Sequential(*layers)
        
        # Final projection from reconstructed to output
        self.output_proj = nn.Linear(self.padded_len, self.output_len)
        
    def haar_decompose(self, x, level):
        # x: (batch, len, n_features)
        # Simple Haar: avg and diff
        if x.shape[1] % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))  # pad time dim
        even = x[:, 0::2, :]  # (batch, len//2, n_features)
        odd = x[:, 1::2, :]   # (batch, len//2, n_features)
        approx = (even + odd) / 2  # (batch, len//2, n_features)
        detail = (even - odd) / 2  # (batch, len//2, n_features)
        return approx, detail
    
    def haar_reconstruct(self, approx, detail):
        # approx, detail: (batch, len, n_features)
        batch, len_, nf = approx.shape
        out = torch.zeros(batch, len_ * 2, nf, device=approx.device, dtype=approx.dtype)
        out[:, 0::2, :] = approx + detail  # (batch, len*2, n_features)
        out[:, 1::2, :] = approx - detail  # (batch, len*2, n_features)
        return out
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Pad to required length
        if x.shape[1] < self.padded_len:
            x = F.pad(x, (0, 0, 0, self.padded_len - x.shape[1]))  # (batch, padded_len, n_features)
        elif x.shape[1] > self.padded_len:
            x = x[:, :self.padded_len, :]  # (batch, padded_len, n_features)
        
        # Multi-level decomposition
        coeffs = []
        current = x  # (batch, padded_len, n_features)
        for level in range(self.n_levels):
            approx, detail = self.haar_decompose(current, level)  # both (batch, padded_len//2^(level+1), n_features)
            # Apply adaptive soft-threshold to detail
            detail = torch.sign(detail) * F.relu(torch.abs(detail) - self.thresholds[level])  # (batch, len', n_features)
            # Fuse detail
            detail = self.detail_fusion[level](detail)  # (batch, len', n_features)
            coeffs.append(detail)
            current = approx  # (batch, len', n_features)
        
        # Fuse approximation
        current = self.approx_fusion(current)  # (batch, len', n_features)
        
        # Reconstruct
        for level in range(self.n_levels - 1, -1, -1):
            detail = coeffs[level]  # (batch, len', n_features)
            # Match lengths: detail should be same length as approx for reconstruction
            if detail.shape[1] > current.shape[1]:
                detail = detail[:, :current.shape[1], :]  # (batch, current_len, n_features)
            elif detail.shape[1] < current.shape[1]:
                detail = F.pad(detail, (0, 0, 0, current.shape[1] - detail.shape[1]))  # (batch, current_len, n_features)
            current = self.haar_reconstruct(current, detail)  # (batch, current_len*2, n_features)
        
        # Trim and project
        current = current[:, :self.padded_len, :]  # (batch, padded_len, n_features)
        current = current.permute(0, 2, 1)  # (batch, n_features, padded_len)
        out = self.output_proj(current)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

