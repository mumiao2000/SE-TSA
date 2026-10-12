import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class TimesNetAdvancedV3(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 512],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.2, 0.4],
            'n_periods': [4, 8, 16],
            'bottleneck_ratio': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 256)
        self.n_layers = cfg.get('n_layers', 2)
        self.dropout = cfg.get('dropout', 0.2)
        self.n_periods = cfg.get('n_periods', 8)
        self.bottleneck_ratio = cfg.get('bottleneck_ratio', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: each feature gets its own linear map to d_model
        self.input_proj = nn.ModuleList([
            nn.Linear(self.input_len, self.d_model) for _ in range(self.n_features)
        ])  # (batch, d_model) per feature
        
        # Period discovery via FFT
        self.period_fc = nn.Linear(self.d_model, self.n_periods)  # (batch, n_periods) per feature
        
        # Frequency-aware gating: use FFT magnitude to weight periods
        self.freq_gate = nn.Sequential(
            nn.Linear(self.n_periods, self.n_periods // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_periods // 2, self.n_periods),
            nn.Sigmoid()
        )  # (batch, n_periods) per feature
        
        # Times blocks: channel-separated 1D convolutions with depthwise mixing
        self.times_blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            block = nn.ModuleDict({
                'norm1': nn.LayerNorm(self.d_model),
                'convs': nn.ModuleList([
                    nn.Sequential(
                        # Depthwise separable: each channel independent
                        nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1, groups=self.d_model),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        # Pointwise bottleneck
                        nn.Conv1d(self.d_model, self.d_model // self.bottleneck_ratio, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv1d(self.d_model // self.bottleneck_ratio, self.d_model, kernel_size=1),
                    ) for _ in range(self.n_periods)
                ]),
                'norm2': nn.LayerNorm(self.d_model),
                'mlp': nn.Sequential(
                    nn.Linear(self.d_model, self.d_model // self.bottleneck_ratio),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.d_model // self.bottleneck_ratio, self.d_model),
                ),
            })
            self.times_blocks.append(block)
        
        # Output: project back from d_model to output_len per feature
        self.output_proj = nn.ModuleList([
            nn.Linear(self.d_model, self.output_len) for _ in range(self.n_features)
        ])  # (batch, output_len) per feature
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Per-feature input projection
        feat_list = []
        for i in range(self.n_features):
            xi = x[:, :, i]  # (batch, input_len)
            xi = self.input_proj[i](xi)  # (batch, d_model)
            feat_list.append(xi)
        
        # Stack to (batch, n_features, d_model)
        h = torch.stack(feat_list, dim=1)  # (batch, n_features, d_model)
        
        # Period discovery and gating per feature
        period_logits = self.period_fc(h)  # (batch, n_features, n_periods)
        
        # Get FFT-based frequency magnitudes for gating
        # Compute FFT on input sequence for each feature
        x_fft = torch.fft.rfft(x, dim=1)  # (batch, input_len//2+1, n_features)
        x_mag = torch.abs(x_fft)  # (batch, input_len//2+1, n_features)
        # Pool to n_periods bins
        mag_pooled = F.adaptive_avg_pool1d(
            x_mag.permute(0, 2, 1),  # (batch, n_features, input_len//2+1)
            self.n_periods
        )  # (batch, n_features, n_periods)
        
        # Frequency gate
        freq_weight = self.freq_gate(mag_pooled)  # (batch, n_features, n_periods)
        period_weight = F.softmax(period_logits, dim=-1) * freq_weight  # (batch, n_features, n_periods)
        period_weight = period_weight / (period_weight.sum(dim=-1, keepdim=True) + 1e-8)  # (batch, n_features, n_periods)
        
        # TimesNet blocks
        for block in self.times_blocks:
            residual = h  # (batch, n_features, d_model)
            
            # Layer norm
            h_norm = block['norm1'](h)  # (batch, n_features, d_model)
            
            # Multi-period convolutions
            conv_outs = []
            for p_idx, conv in enumerate(block['convs']):
                # Reshape according to period: treat as 2D (period, length//period)
                # Use fixed period lengths based on dominant period ~51
                period_len = max(2, 52 // (p_idx + 1))
                if period_len > self.input_len:
                    period_len = self.input_len
                
                hp = h_norm.permute(0, 2, 1)  # (batch, d_model, n_features)
                # Apply 1D conv over feature dimension (treating each channel independently)
                hp = conv(hp)  # (batch, d_model, n_features)
                hp = hp.permute(0, 2, 1)  # (batch, n_features, d_model)
                conv_outs.append(hp)
            
            # Weighted aggregation by period importance
            conv_stack = torch.stack(conv_outs, dim=-1)  # (batch, n_features, d_model, n_periods)
            period_w = period_weight.unsqueeze(2)  # (batch, n_features, 1, n_periods)
            h = torch.sum(conv_stack * period_w, dim=-1)  # (batch, n_features, d_model)
            h = self.dropout_layer(h)
            h = h + residual  # (batch, n_features, d_model)
            
            # MLP with residual
            residual = h  # (batch, n_features, d_model)
            h = block['norm2'](h)  # (batch, n_features, d_model)
            h = block['mlp'](h)  # (batch, n_features, d_model)
            h = self.dropout_layer(h)
            h = h + residual  # (batch, n_features, d_model)
        
        # Per-feature output projection
        out_list = []
        for i in range(self.n_features):
            oi = h[:, i, :]  # (batch, d_model)
            oi = self.output_proj[i](oi)  # (batch, output_len)
            out_list.append(oi)
        
        out = torch.stack(out_list, dim=-1)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
