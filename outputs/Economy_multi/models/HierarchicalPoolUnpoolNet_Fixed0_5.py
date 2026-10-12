import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HierarchicalPoolUnpoolNet_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [32, 128, 512],
            'n_scales': [2, 3],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 128)
        self.n_scales = self.config.get('n_scales', 2)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Ensure we don't pool below 2
        max_scales = 0
        temp_len = self.input_len
        while temp_len >= 4:
            max_scales += 1
            temp_len //= 2
        self.n_scales = min(self.n_scales, max_scales)
        if self.n_scales < 1:
            self.n_scales = 1
        
        # Compute lengths at each scale
        self.scale_lengths = []
        current_len = self.input_len
        for s in range(self.n_scales):
            self.scale_lengths.append(current_len)
            current_len //= 2
        
        # Encoder: project from time length to hidden_dim at each scale
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for s in range(self.n_scales):
            self.encoders.append(nn.Sequential(
                nn.Linear(self.scale_lengths[s], self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ))
            if s < self.n_scales - 1:
                self.pools.append(nn.AvgPool1d(kernel_size=2, stride=2))
        
        # Decoder: merge skip and upsampled, project to hidden_dim
        self.decoders = nn.ModuleList()
        self.upsample_projs = nn.ModuleList()
        
        for s in range(self.n_scales - 1, -1, -1):
            # Decoder input: hidden_dim (from below) + hidden_dim (skip) = 2*hidden_dim
            self.decoders.append(nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ))
            
            # Projection after upsampling (doubling hidden_dim -> hidden_dim)
            if s > 0:
                self.upsample_projs.append(nn.Linear(self.hidden_dim * 2, self.hidden_dim))
        
        # Final projection from hidden_dim to output_len
        self.final_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.output_len)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose to (batch, n_features, input_len) - treat features as channels
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Encoder path
        skip_connections = []
        h = x  # (batch, n_features, input_len)
        
        for s in range(self.n_scales):
            # Encode: project time dimension to hidden_dim
            # h is (batch, n_features, scale_lengths[s])
            h_encoded = self.encoders[s](h)  # (batch, n_features, hidden_dim)
            skip_connections.append(h_encoded)
            
            # Pool for next scale (in time dimension)
            if s < self.n_scales - 1:
                h = self.pools[s](h)  # (batch, n_features, scale_lengths[s+1])
        
        # Decoder path: start from deepest level
        h = skip_connections[-1]  # (batch, n_features, hidden_dim)
        
        decoder_idx = 0
        for s in range(self.n_scales - 1, -1, -1):
            if s < self.n_scales - 1:
                # Upsample h by repeating in hidden_dim (which encodes temporal info)
                h = h.repeat_interleave(2, dim=-1)  # (batch, n_features, hidden_dim * 2)
                h = self.upsample_projs[decoder_idx - 1](h)  # (batch, n_features, hidden_dim)
                
                # Concatenate with skip connection
                skip = skip_connections[s]  # (batch, n_features, hidden_dim)
                h = torch.cat([h, skip], dim=-1)  # (batch, n_features, hidden_dim * 2)
                
                # Decode
                h = self.decoders[decoder_idx](h)  # (batch, n_features, hidden_dim)
            else:
                # Deepest level: just pass through decoder (no skip needed, but use same structure)
                # Actually at deepest level we don't have an upsampled input, just use skip alone
                # But we already set h = skip_connections[-1], so apply decoder with self-concat
                h_skip = torch.cat([h, h], dim=-1)  # (batch, n_features, hidden_dim * 2)
                h = self.decoders[decoder_idx](h_skip)  # (batch, n_features, hidden_dim)
            
            decoder_idx += 1
        
        # h is now (batch, n_features, hidden_dim)
        # Project to output length
        out = self.final_proj(h)  # (batch, n_features, output_len)
        
        # Transpose back to (batch, output_len, n_features)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

