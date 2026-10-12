import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class TimesNetAdvancedV2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 1024],
            'd_ff': [256, 1024, 2048],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.1, 0.3],
            'top_k': [2, 3, 5],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.d_model = self.config.get('d_model', 512)
        self.d_ff = self.config.get('d_ff', 1024)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.top_k = self.config.get('top_k', 3)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: (batch, input_len, n_features) -> (batch, input_len, d_model)
        self.input_projection = nn.Linear(self.n_features, self.d_model)
        
        # TimesNet blocks
        self.blocks = nn.ModuleList([
            TimesBlock(self.input_len, self.d_model, self.d_ff, self.top_k, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model)
            for _ in range(self.n_layers)
        ])
        
        # Output projection: (batch, input_len, d_model) -> (batch, output_len, n_features)
        # Use channel-wise linear for parameter efficiency
        self.output_projection = nn.Linear(self.d_model, self.n_features)
        
        # Temporal projection to get from input_len to output_len
        self.temporal_projection = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection
        x = self.input_projection(x)  # (batch, input_len, d_model)
        
        # TimesNet blocks with residual connections
        for block, ln in zip(self.blocks, self.layer_norms):
            residual = x  # (batch, input_len, d_model)
            x = block(x)  # (batch, input_len, d_model)
            x = ln(x + residual)  # (batch, input_len, d_model)
        
        # Temporal projection: (batch, input_len, d_model) -> (batch, output_len, d_model)
        x = x.permute(0, 2, 1)  # (batch, d_model, input_len)
        x = self.temporal_projection(x)  # (batch, d_model, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, d_model)
        
        # Output projection
        x = self.output_projection(x)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x


class TimesBlock(nn.Module):
    def __init__(self, input_len, d_model, d_ff, top_k, dropout):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model
        self.top_k = top_k
        
        # Period discovery parameters
        self.period_fc = nn.Linear(d_model, d_model)
        
        # 2D convolutions: channel-wise for parameter efficiency
        # We use depthwise separable convolutions
        self.conv_kernels = [3, 5, 7]
        self.convs = nn.ModuleList([
            nn.Sequential(
                # Depthwise
                nn.Conv2d(d_model, d_model, kernel_size=k, padding=k//2, groups=d_model),
                nn.BatchNorm2d(d_model),
                nn.GELU(),
                # Pointwise bottleneck
                nn.Conv2d(d_model, d_ff // 2, kernel_size=1),
                nn.BatchNorm2d(d_ff // 2),
                nn.GELU(),
                # Pointwise expansion
                nn.Conv2d(d_ff // 2, d_model, kernel_size=1),
                nn.BatchNorm2d(d_model),
            )
            for k in self.conv_kernels
        ])
        
        # Soft period gating
        self.period_gate = nn.Linear(d_model, len(self.conv_kernels))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, input_len, d_model)
        batch_size, seq_len, d_model = x.shape  # (batch, input_len, d_model)
        
        # Compute period-based FFT
        # Transpose for FFT: (batch, d_model, input_len)
        x_fft = x.permute(0, 2, 1)  # (batch, d_model, input_len)
        
        # FFT to find periods
        xf = torch.fft.rfft(x_fft, dim=-1)  # (batch, d_model, input_len//2+1)
        xf = torch.abs(xf)  # (batch, d_model, input_len//2+1)
        xf = xf.mean(dim=1)  # (batch, input_len//2+1)
        
        # Find top-k periods
        frequency_list = xf[:, 1:]  # Remove DC component, (batch, input_len//2)
        _, top_indices = torch.topk(frequency_list, self.top_k, dim=-1)  # (batch, top_k)
        
        # Convert to periods (avoid division by zero)
        periods = (self.input_len // 2) // (top_indices + 1)  # (batch, top_k)
        periods = torch.clamp(periods, min=2, max=self.input_len // 2)  # (batch, top_k)
        
        # Period gating based on input features
        gate_logits = self.period_gate(x.mean(dim=1))  # (batch, len(conv_kernels))
        gate = F.softmax(gate_logits, dim=-1)  # (batch, len(conv_kernels))
        
        # Multi-period representation with soft gating
        res = []
        for i, conv in enumerate(self.convs):
            # Use median period for reshaping
            period = periods[:, i % self.top_k].median().long().item()
            period = max(period, 2)
            
            # Pad to multiple of period
            pad_len = (period - seq_len % period) % period
            if pad_len > 0:
                x_padded = F.pad(x, (0, 0, 0, pad_len))  # (batch, seq_len+pad_len, d_model)
            else:
                x_padded = x  # (batch, seq_len, d_model)
            
            # Reshape to 2D: (batch, d_model, period, num_periods)
            new_seq_len = x_padded.shape[1]
            num_periods = new_seq_len // period
            x_2d = x_padded.reshape(batch_size, new_seq_len, d_model)  # (batch, new_seq_len, d_model)
            x_2d = x_2d[:, :num_periods * period, :]  # (batch, num_periods*period, d_model)
            x_2d = x_2d.reshape(batch_size, num_periods, period, d_model)  # (batch, num_periods, period, d_model)
            x_2d = x_2d.permute(0, 3, 2, 1)  # (batch, d_model, period, num_periods)
            
            # Apply 2D convolution
            x_conv = conv(x_2d)  # (batch, d_model, period, num_periods)
            
            # Reshape back
            x_conv = x_conv.permute(0, 3, 2, 1)  # (batch, num_periods, period, d_model)
            x_conv = x_conv.reshape(batch_size, num_periods * period, d_model)  # (batch, num_periods*period, d_model)
            x_conv = x_conv[:, :seq_len, :]  # (batch, seq_len, d_model)
            
            # Apply gate weight
            weight = gate[:, i:i+1].unsqueeze(1)  # (batch, 1, 1)
            res.append(x_conv * weight)  # (batch, seq_len, d_model)
        
        # Sum gated convolutions
        x = sum(res)  # (batch, seq_len, d_model)
        x = self.dropout(x)  # (batch, seq_len, d_model)
        
        return x  # (batch, input_len, d_model)
