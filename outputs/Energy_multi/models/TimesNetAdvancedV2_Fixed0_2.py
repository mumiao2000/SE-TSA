import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimesNetAdvancedV2_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'd_ff': [128, 512, 2048],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'top_k': [1, 3, 5],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.d_model = self.config.get('d_model', 256)
        self.d_ff = self.config.get('d_ff', 512)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.top_k = self.config.get('top_k', 3)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: from input_len to d_model per feature
        self.input_projection = nn.Linear(self.input_len, self.d_model)
        
        # TimesNet blocks
        self.blocks = nn.ModuleList([
            TimesBlockV2(
                input_len=self.input_len,
                d_model=self.d_model,
                d_ff=self.d_ff,
                top_k=self.top_k,
                dropout=self.dropout
            ) for _ in range(self.n_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, self.output_len)
        )
        
        # Residual connection from input
        self.residual_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store for residual
        x_input = x.clone()  # (batch, input_len, n_features)
        
        # Channel-independent processing: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Input projection to d_model
        x = self.input_projection(x)  # (batch, n_features, d_model)
        
        # TimesNet blocks with residual connections
        for i, (block, norm) in enumerate(zip(self.blocks, self.layer_norms)):
            residual = x  # (batch, n_features, d_model)
            x = block(x)  # (batch, n_features, d_model)
            x = norm(x + residual)  # (batch, n_features, d_model)
        
        # Output projection: (batch, n_features, output_len)
        x = self.output_projection(x)  # (batch, n_features, output_len)
        
        # Permute back: (batch, output_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Add residual from input (trend preservation)
        last_val = x_input[:, -1:, :]  # (batch, 1, n_features)
        residual_forecast = last_val.repeat(1, self.output_len, 1)  # (batch, output_len, n_features)
        
        # Gated combination
        gate = torch.sigmoid(self.residual_weight)  # scalar
        x = gate * x + (1 - gate) * residual_forecast  # (batch, output_len, n_features)
        
        # RevIN denormalization
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x  # (batch, output_len, n_features)


class TimesBlockV2(nn.Module):
    def __init__(self, input_len, d_model, d_ff, top_k, dropout):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model
        self.top_k = top_k
        
        # Multi-scale 2D convolutions - d_model is the channel dimension
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, d_model // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(d_model // 2),
                nn.GELU(),
                nn.Dropout2d(dropout),
                nn.Conv2d(d_model // 2, d_model, kernel_size=3, padding=1),
                nn.BatchNorm2d(d_model),
            ),
            nn.Sequential(
                nn.Conv2d(d_model, d_model // 2, kernel_size=5, padding=2),
                nn.BatchNorm2d(d_model // 2),
                nn.GELU(),
                nn.Dropout2d(dropout),
                nn.Conv2d(d_model // 2, d_model, kernel_size=5, padding=2),
                nn.BatchNorm2d(d_model),
            )
        ])
        
        # Gating for multi-scale fusion
        self.gate = nn.Linear(d_model * 2, 2)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (batch, n_features, d_model)
        batch_size, n_features, d_model = x.shape
        
        # Period analysis: compute FFT on each feature's representation across d_model
        # We need to find periods in the sequence dimension. Since x is (batch, n_features, d_model),
        # we interpret n_features as the "sequence" for period discovery (channel-independent)
        
        # Compute FFT-based periods on the n_features dimension
        x_for_period = x.permute(0, 2, 1)  # (batch, d_model, n_features)
        
        # Average across d_model to get per-feature signal
        x_fft_input = x_for_period.mean(dim=1)  # (batch, n_features)
        
        # Actually, we need to find periods that divide n_features well
        # Use FFT on the feature dimension to find periodic patterns
        x_fft = torch.fft.rfft(x, dim=1)  # (batch, n_features//2+1, d_model)
        x_fft = torch.abs(x_fft).mean(dim=-1)  # (batch, n_features//2+1)
        
        # Get top-k periods for each batch element
        _, top_indices = torch.topk(x_fft, k=min(self.top_k, x_fft.size(-1)), dim=-1)  # (batch, top_k)
        
        # Use most common period across batch
        periods = []
        for b in range(batch_size):
            period_vals = []
            for idx in top_indices[b]:
                freq = idx.item()
                if freq > 0:
                    # Convert frequency to period
                    period = max(1, n_features // (freq + 1))
                    period_vals.append(period)
            if len(period_vals) == 0:
                period_vals = [max(1, n_features // 2)]
            periods.extend(period_vals)
        
        # Use median period for stability, clamped to valid range
        period = int(torch.tensor(periods).float().median().item())
        period = max(1, min(period, n_features // 2))
        if period < 1:
            period = 1
        
        # Make n_features divisible by period
        n_features_eff = (n_features // period) * period
        if n_features_eff == 0:
            n_features_eff = period
            # Pad if needed
            if n_features < n_features_eff:
                pad_size = n_features_eff - n_features
                x_padded = F.pad(x, (0, 0, 0, pad_size))  # (batch, n_features_eff, d_model)
            else:
                x_padded = x[:, :n_features_eff, :]  # (batch, n_features_eff, d_model)
        else:
            x_padded = x[:, :n_features_eff, :]  # (batch, n_features_eff, d_model)
        
        num_periods = n_features_eff // period
        
        # Reshape to 2D: (batch, d_model, num_periods, period)
        # First permute to (batch, d_model, n_features_eff)
        x_2d = x_padded.permute(0, 2, 1)  # (batch, d_model, n_features_eff)
        # Then reshape to (batch, d_model, num_periods, period)
        x_2d = x_2d.reshape(batch_size, d_model, num_periods, period)  # (batch, d_model, num_periods, period)
        
        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = conv(x_2d)  # (batch, d_model, num_periods, period)
            conv_outputs.append(conv_out)
        
        # Global average pooling for gating
        gap0 = conv_outputs[0].mean(dim=(2, 3))  # (batch, d_model)
        gap1 = conv_outputs[1].mean(dim=(2, 3))  # (batch, d_model)
        
        gate_input = torch.cat([gap0, gap1], dim=-1)  # (batch, d_model * 2)
        gate_weights = F.softmax(self.gate(gate_input), dim=-1)  # (batch, 2)
        
        # Weighted combination
        w0 = gate_weights[:, 0:1, None, None]  # (batch, 1, 1, 1)
        w1 = gate_weights[:, 1:2, None, None]  # (batch, 1, 1, 1)
        
        x_fused = w0 * conv_outputs[0] + w1 * conv_outputs[1]  # (batch, d_model, num_periods, period)
        
        # Reshape back: (batch, d_model, n_features_eff)
        x_out = x_fused.reshape(batch_size, d_model, n_features_eff)
        
        # Permute back: (batch, n_features_eff, d_model)
        x_out = x_out.permute(0, 2, 1)  # (batch, n_features_eff, d_model)
        
        # Truncate back to original n_features if we padded
        if n_features_eff > n_features:
            x_out = x_out[:, :n_features, :]  # (batch, n_features, d_model)
        
        # First residual and norm
        x = self.layer_norm1(x + x_out)  # (batch, n_features, d_model)
        
        # FFN
        ffn_out = self.ffn(x)  # (batch, n_features, d_model)
        x = self.layer_norm2(x + ffn_out)  # (batch, n_features, d_model)
        
        return x  # (batch, n_features, d_model)
