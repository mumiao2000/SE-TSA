import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimesNetAdvancedV2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 512],
            'd_ff': [128, 512, 1024],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.2, 0.4],
            'n_periods': [2, 4, 8],
            'bottleneck_ratio': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 256)
        self.d_ff = cfg.get('d_ff', 512)
        self.n_layers = cfg.get('n_layers', 2)
        self.dropout = cfg.get('dropout', 0.2)
        self.n_periods = cfg.get('n_periods', 4)
        self.bottleneck_ratio = cfg.get('bottleneck_ratio', 4)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to d_model per channel
        self.input_projection = nn.Linear(self.input_len, self.d_model)
        
        # Period discovery: learnable frequencies based on dominant period ~51
        # Candidate periods around the dominant period and its harmonics
        base_period = 52  # close to dominant period 51.16
        candidate_periods = []
        for i in range(self.n_periods):
            period = max(2, int(base_period / (i + 1)))
            if period not in candidate_periods:
                candidate_periods.append(period)
        # Ensure we have exactly n_periods
        while len(candidate_periods) < self.n_periods:
            candidate_periods.append(max(2, candidate_periods[-1] // 2))
        self.periods = nn.Parameter(torch.tensor(candidate_periods[:self.n_periods], dtype=torch.float32), requires_grad=False)
        
        # Period importance gating
        self.period_gate = nn.Parameter(torch.ones(self.n_periods) / self.n_periods)
        
        # TimesBlocks with channel-dependent convolutions
        self.blocks = nn.ModuleList([
            TimesBlockV2(
                d_model=self.d_model,
                d_ff=self.d_ff,
                n_features=self.n_features,
                n_periods=self.n_periods,
                dropout=self.dropout,
                bottleneck_ratio=self.bottleneck_ratio
            ) for _ in range(self.n_layers)
        ])
        
        # Layer norms
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.n_layers)
        ])
        
        # Output projection: d_model -> output_len per channel
        self.output_projection = nn.Linear(self.d_model, self.output_len)
        
        # Residual connection scaling
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection: per-channel linear transform
        # (batch, input_len, n_features) -> (batch, n_features, input_len) -> (batch, n_features, d_model)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.input_projection(x)  # (batch, n_features, d_model)
        
        # Store residual for skip connection
        residual = x
        
        # Apply TimesBlocks with period-aware processing
        for i, (block, norm) in enumerate(zip(self.blocks, self.norm_layers)):
            # Period discovery and gating
            period_list = self.periods.int().tolist()
            period_weights = F.softmax(self.period_gate, dim=0)  # (n_periods,)
            
            # Forward through block
            x_out = block(x, period_list, period_weights)  # (batch, n_features, d_model)
            x_out = norm(x_out)  # (batch, n_features, d_model)
            
            # Scaled residual connection
            x = residual + self.residual_scale * x_out  # (batch, n_features, d_model)
            residual = x  # Update residual for next layer
        
        # Output projection: per-channel linear transform to output_len
        out = self.output_projection(x)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out


class TimesBlockV2(nn.Module):
    def __init__(self, d_model, d_ff, n_features, n_periods, dropout, bottleneck_ratio):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        self.n_periods = n_periods
        
        # Bottleneck dimension for parameter efficiency
        self.bottleneck_dim = max(d_model // bottleneck_ratio, 16)
        
        # Period-specific processing: channel-dependent 1D convolutions instead of 2D
        # Each period gets its own set of channel-wise convolutions
        self.period_convs = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    # Bottleneck projection
                    nn.Conv1d(d_model, self.bottleneck_dim, 1, groups=1),
                    nn.ReLU(),
                    # Depthwise separable convolution for efficiency
                    nn.Conv1d(self.bottleneck_dim, self.bottleneck_dim, 3, padding=1, groups=self.bottleneck_dim),
                    nn.ReLU(),
                    # Expansion back
                    nn.Conv1d(self.bottleneck_dim, d_model, 1),
                    nn.Dropout(dropout)
                ) for _ in range(n_features)
            ]) for _ in range(n_periods)
        ])
        
        # Feed-forward network with bottleneck
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, period_list, period_weights):
        # x: (batch, n_features, d_model)
        batch_size = x.shape[0]
        
        # Process each period and weight by importance
        period_outputs = []
        for p_idx, (period, weight) in enumerate(zip(period_list, period_weights)):
            # Reshape for this period: (batch, n_features, d_model) -> handle as 1D conv per channel
            # For each channel, apply its period-specific convolution
            channel_outputs = []
            for c_idx in range(self.n_features):
                # Extract channel: (batch, d_model)
                x_c = x[:, c_idx, :]  # (batch, d_model)
                # Add sequence dim for conv1d: (batch, d_model, 1) - treat d_model as channels
                x_c = x_c.unsqueeze(-1)  # (batch, d_model, 1)
                
                # Apply period-specific convolution for this channel
                conv_out = self.period_convs[p_idx][c_idx](x_c)  # (batch, d_model, 1)
                conv_out = conv_out.squeeze(-1)  # (batch, d_model)
                channel_outputs.append(conv_out)
            
            # Stack channels back
            period_out = torch.stack(channel_outputs, dim=1)  # (batch, n_features, d_model)
            period_outputs.append(period_out * weight)  # Weight by period importance
        
        # Sum weighted period contributions
        out = torch.stack(period_outputs, dim=0).sum(dim=0)  # (batch, n_features, d_model)
        
        # First norm and residual
        out = self.norm1(out + x)  # (batch, n_features, d_model)
        
        # FFN with residual
        ffn_out = self.ffn(out)  # (batch, n_features, d_model)
        out = self.norm2(out + ffn_out)  # (batch, n_features, d_model)
        
        return out
