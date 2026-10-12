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
            'd_ff': [64, 256, 1024],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.2, 0.4],
            'n_periods': [2, 4, 8],
            'n_groups': [2, 4, 8],
            'bottleneck_ratio': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 256)
        self.d_ff = cfg.get('d_ff', 256)
        self.n_layers = cfg.get('n_layers', 2)
        self.dropout = cfg.get('dropout', 0.2)
        self.n_periods = cfg.get('n_periods', 4)
        self.n_groups = cfg.get('n_groups', 4)
        self.bottleneck_ratio = cfg.get('bottleneck_ratio', 4)
        
        # Ensure n_groups divides n_features
        self.n_groups = min(self.n_groups, self.n_features)
        while self.n_features % self.n_groups != 0:
            self.n_groups -= 1
        self.channels_per_group = self.n_features // self.n_groups
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: per-group linear
        self.input_proj = nn.ModuleList([
            nn.Linear(self.input_len, self.d_model)
            for _ in range(self.n_groups)
        ])  # (batch, channels_per_group, d_model)
        
        # FFT-based period discovery
        self.period_fc = nn.Linear(self.d_model, self.n_periods)  # (batch, channels_per_group, n_periods)
        
        # Period embedding (learned importance)
        self.period_embed = nn.Parameter(torch.randn(self.n_periods, self.d_model // self.bottleneck_ratio))
        
        # Temporal mixing: depthwise-separable 1D conv per group
        self.temporal_convs = nn.ModuleList()
        for _ in range(self.n_layers):
            layer_convs = nn.ModuleList()
            for _ in range(self.n_groups):
                # Depthwise + pointwise (separable)
                layer_convs.append(nn.Sequential(
                    nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1, groups=self.d_model),
                    nn.ReLU(),
                    nn.Conv1d(self.d_model, self.d_model, kernel_size=1),
                    nn.Dropout(self.dropout),
                ))
            self.temporal_convs.append(layer_convs)
        
        # Cross-period mixing
        self.period_mix = nn.ModuleList([
            nn.Linear(self.n_periods, self.n_periods)
            for _ in range(self.n_layers)
        ])  # (batch, channels_per_group, d_model, n_periods)
        
        # Output projection per group
        self.output_proj = nn.ModuleList([
            nn.Linear(self.d_model, self.output_len)
            for _ in range(self.n_groups)
        ])  # (batch, channels_per_group, output_len)
        
        # Direct linear skip for stability
        self.skip = nn.Linear(self.input_len, self.output_len)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store for skip
        x_input = x  # (batch, input_len, n_features)
        
        # Split into groups
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.input_len, self.n_groups, self.channels_per_group)  # (batch, input_len, n_groups, channels_per_group)
        x = x.permute(0, 2, 3, 1)  # (batch, n_groups, channels_per_group, input_len)
        
        # Input projection per group
        group_feats = []
        for g in range(self.n_groups):
            g_x = x[:, g, :, :]  # (batch, channels_per_group, input_len)
            g_proj = self.input_proj[g](g_x)  # (batch, channels_per_group, d_model)
            group_feats.append(g_proj)
        
        # Stack: (batch, n_groups, channels_per_group, d_model)
        h = torch.stack(group_feats, dim=1)  # (batch, n_groups, channels_per_group, d_model)
        
        # FFT-based period discovery on averaged features
        h_avg = h.mean(dim=2)  # (batch, n_groups, d_model)
        
        # Period discovery via FFT on time dimension (use input_len as proxy)
        # Compute period lengths from FFT of input
        with torch.no_grad():
            # Use first group's input for period discovery
            fft_input = x[:, 0, :, :]  # (batch, channels_per_group, input_len)
            fft_vals = torch.fft.rfft(fft_input, dim=-1)  # (batch, channels_per_group, input_len//2+1)
            fft_mag = torch.abs(fft_vals).mean(dim=1)  # (batch, input_len//2+1)
            # Get top periods
            _, top_indices = torch.topk(fft_mag[:, 1:], min(self.n_periods, fft_mag.shape[1]-1), dim=-1)  # exclude DC
            top_indices = top_indices + 1  # adjust for excluded DC
            # Convert to period lengths: period = input_len / frequency_index
            periods = self.input_len / top_indices.float()  # (batch, n_periods)
            periods = periods.clamp(min=2, max=self.input_len)
        
        # Learned period weights
        period_logits = self.period_fc(h_avg)  # (batch, n_groups, n_periods)
        period_weights = F.softmax(period_logits, dim=-1)  # (batch, n_groups, n_periods)
        
        # Multi-period processing
        for layer_idx in range(self.n_layers):
            new_h = []
            for g in range(self.n_groups):
                g_h = h[:, g, :, :]  # (batch, channels_per_group, d_model)
                
                # Apply temporal conv (depthwise-separable)
                g_h_conv = g_h.permute(0, 2, 1)  # (batch, d_model, channels_per_group)
                g_h_conv = self.temporal_convs[layer_idx][g](g_h_conv)  # (batch, d_model, channels_per_group)
                g_h_conv = g_h_conv.permute(0, 2, 1)  # (batch, channels_per_group, d_model)
                g_h_conv = self.dropout_layer(g_h_conv)
                
                # Period-aware mixing: reshape by periods
                # For simplicity, use learned period embedding weighted combination
                period_emb = torch.matmul(period_weights[:, g, :], self.period_embed)  # (batch, d_model//bottleneck_ratio)
                period_emb = period_emb.unsqueeze(1).expand(-1, self.channels_per_group, -1)  # (batch, channels_per_group, d_model//bottleneck_ratio)
                
                # Combine with bottleneck
                g_h_bottleneck = F.linear(g_h_conv, torch.randn(self.d_model // self.bottleneck_ratio, self.d_model).to(g_h_conv.device) * 0.02)
                # Use proper linear layer instead
                g_h_bottleneck = g_h_conv[..., :self.d_model // self.bottleneck_ratio]  # (batch, channels_per_group, d_model//bottleneck_ratio)
                
                g_h_mixed = g_h_bottleneck * period_emb  # (batch, channels_per_group, d_model//bottleneck_ratio)
                
                # Project back
                g_h_out = F.pad(g_h_mixed, (0, self.d_model - self.d_model // self.bottleneck_ratio))  # (batch, channels_per_group, d_model)
                # Actually use proper projection
                if not hasattr(self, 'bottleneck_up'):
                    self.bottleneck_up = nn.Linear(self.d_model // self.bottleneck_ratio, self.d_model).to(g_h_mixed.device)
                g_h_out = self.bottleneck_up(g_h_mixed)  # (batch, channels_per_group, d_model)
                
                # Residual
                g_h_out = g_h_out + g_h_conv  # (batch, channels_per_group, d_model)
                
                new_h.append(g_h_out)
            
            h = torch.stack(new_h, dim=1)  # (batch, n_groups, channels_per_group, d_model)
        
        # Output projection per group
        outputs = []
        for g in range(self.n_groups):
            g_h = h[:, g, :, :]  # (batch, channels_per_group, d_model)
            g_out = self.output_proj[g](g_h)  # (batch, channels_per_group, output_len)
            outputs.append(g_out)
        
        out = torch.stack(outputs, dim=1)  # (batch, n_groups, channels_per_group, output_len)
        out = out.permute(0, 3, 1, 2)  # (batch, output_len, n_groups, channels_per_group)
        out = out.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Add skip connection
        skip_out = self.skip(x_input.permute(0, 2, 1))  # (batch, n_features, output_len)
        skip_out = skip_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        out = out + skip_out  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

