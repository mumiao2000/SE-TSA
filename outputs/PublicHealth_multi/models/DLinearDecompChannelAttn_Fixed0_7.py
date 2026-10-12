import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DLinearDecompChannelAttn_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_groups': [4, 8, 16],
            'dropout': [0.0, 0.2, 0.4],
            'period_gates': [4, 8, 16],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_groups = self.config.get('n_groups', 8)
        self.dropout_rate = self.config.get('dropout', 0.2)
        self.n_period_gates = self.config.get('period_gates', 8)
        
        # Ensure n_groups divides n_features
        self.n_groups = min(self.n_groups, self.n_features)
        while self.n_features % self.n_groups != 0 and self.n_groups > 1:
            self.n_groups -= 1
        self.group_size = self.n_features // self.n_groups
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Decomposition: learnable moving average for trend extraction
        self.kernel_size = max(3, input_len // 4)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.trend_extractor = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        
        # Period-aware adaptive gating for seasonal component
        # Fixed frequency bins that always produce n_period_gates values
        self.period_frequencies = torch.linspace(0, input_len // 2, self.n_period_gates)
        self.period_gates = nn.Parameter(torch.ones(self.n_period_gates))
        self.period_proj = nn.Linear(self.n_period_gates, self.hidden_dim)
        
        # Trend projections: channel-dependent
        self.trend_proj = nn.Linear(input_len, output_len)
        
        # Seasonal projections with period-aware enhancement
        self.seasonal_proj = nn.Linear(input_len, output_len)
        
        # Cross-channel attention with grouping
        self.group_queries = nn.ModuleList([
            nn.Linear(output_len, output_len // self.n_groups) 
            for _ in range(self.n_groups)
        ])
        self.group_keys = nn.ModuleList([
            nn.Linear(output_len, output_len // self.n_groups)
            for _ in range(self.n_groups)
        ])
        self.group_values = nn.ModuleList([
            nn.Linear(output_len, output_len // self.n_groups)
            for _ in range(self.n_groups)
        ])
        self.group_out_proj = nn.ModuleList([
            nn.Linear(output_len // self.n_groups, output_len)
            for _ in range(self.n_groups)
        ])
        
        # Channel mixing after attention
        self.channel_mix = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.residual_scale = nn.Parameter(torch.ones(1))
        
    def extract_trend(self, x):
        # x: (batch, n_features, input_len)
        # Apply moving average for trend
        trend = self.trend_extractor(x)  # (batch, n_features, input_len)
        # Handle potential length mismatch from padding
        if trend.shape[-1] != x.shape[-1]:
            trend = F.interpolate(trend, size=x.shape[-1], mode='linear', align_corners=False)  # (batch, n_features, input_len)
        return trend
    
    def compute_period_features(self, x):
        # x: (batch, n_features, input_len)
        # Compute FFT-based period features
        x_fft = torch.fft.rfft(x, dim=-1)  # (batch, n_features, input_len//2+1)
        x_magnitude = torch.abs(x_fft)  # (batch, n_features, input_len//2+1)
        
        # Sample at fixed frequency positions, always n_period_gates values
        n_freq = x_magnitude.shape[-1]
        freq_indices = self.period_frequencies.to(x.device).clamp(0, n_freq - 1).long()  # (n_period_gates,)
        
        # Gather period features: (batch, n_features, n_period_gates)
        period_features = x_magnitude[:, :, freq_indices]  # (batch, n_features, n_period_gates)
        
        # Apply adaptive gating
        gate_weights = F.softmax(self.period_gates, dim=0)  # (n_period_gates,)
        gated_periods = period_features * gate_weights.view(1, 1, -1)  # (batch, n_features, n_period_gates)
        
        # Project to hidden dim
        period_emb = self.period_proj(gated_periods)  # (batch, n_features, hidden_dim)
        return period_emb
    
    def grouped_channel_attention(self, x):
        # x: (batch, output_len, n_features)
        batch_size = x.shape[0]
        out_len = x.shape[1]
        
        # Reshape to groups: (batch, output_len, n_groups, group_size)
        x_grouped = x.view(batch_size, out_len, self.n_groups, self.group_size)
        
        # Transpose for attention: (batch, n_groups, group_size, output_len)
        x_grouped = x_grouped.permute(0, 2, 3, 1)  # (batch, n_groups, group_size, output_len)
        
        outputs = []
        for g in range(self.n_groups):
            x_g = x_grouped[:, g, :, :]  # (batch, group_size, output_len)
            
            # Self-attention within group across channels
            q = self.group_queries[g](x_g)  # (batch, group_size, output_len//n_groups)
            k = self.group_keys[g](x_g)     # (batch, group_size, output_len//n_groups)
            v = self.group_values[g](x_g)   # (batch, group_size, output_len//n_groups)
            
            # Attention: (batch, group_size, group_size)
            attn = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            out = torch.matmul(attn, v)  # (batch, group_size, output_len//n_groups)
            out = self.group_out_proj[g](out)  # (batch, group_size, output_len)
            outputs.append(out)
        
        # Stack and reshape back
        out = torch.stack(outputs, dim=1)  # (batch, n_groups, group_size, output_len)
        out = out.permute(0, 3, 1, 2)  # (batch, output_len, n_groups, group_size)
        out = out.reshape(batch_size, out_len, self.n_features)  # (batch, output_len, n_features)
        return out
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for 1D operations: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Decomposition
        trend = self.extract_trend(x)  # (batch, n_features, input_len)
        seasonal = x - trend  # (batch, n_features, input_len)
        
        # Period-aware features for seasonal
        period_emb = self.compute_period_features(seasonal)  # (batch, n_features, hidden_dim)
        
        # Project trend and seasonal
        trend_out = self.trend_proj(trend)  # (batch, n_features, output_len)
        seasonal_out = self.seasonal_proj(seasonal)  # (batch, n_features, output_len)
        
        # Add period-aware bias to seasonal
        seasonal_out = seasonal_out + period_emb.sum(dim=-1, keepdim=True) * 0.1  # (batch, n_features, output_len)
        
        # Combine
        out = trend_out + seasonal_out  # (batch, n_features, output_len)
        
        # Transpose back: (batch, output_len, n_features)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Grouped cross-channel attention
        attn_out = self.grouped_channel_attention(out)  # (batch, output_len, n_features)
        out = out + self.dropout(attn_out)  # (batch, output_len, n_features)
        
        # Channel mixing with residual
        mix_out = self.channel_mix(out)  # (batch, output_len, n_features)
        out = out + self.residual_scale * self.dropout(mix_out)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out
