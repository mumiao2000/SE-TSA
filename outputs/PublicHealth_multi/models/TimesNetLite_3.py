import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class TimesNetLite(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'd_ff': [128, 512, 2048],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.1, 0.3],
            'top_k': [1, 3, 5]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.d_ff = self.config.get('d_ff', 512)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.top_k = self.config.get('top_k', 3)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.d_model)
        
        # TimesBlocks
        self.times_blocks = nn.ModuleList([
            TimesBlockLite(self.d_model, self.d_ff, self.input_len, self.top_k, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.norm = nn.LayerNorm(self.d_model)
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.n_features)
        
        # Direct skip connection from input to output
        self.skip_proj = nn.Linear(self.input_len, self.output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save for skip
        x_skip = x  # (batch, input_len, n_features)
        
        # Input projection
        x = self.input_proj(x)  # (batch, input_len, d_model)
        
        # TimesNet blocks
        for block in self.times_blocks:
            x = x + block(x)  # (batch, input_len, d_model)
        
        x = self.norm(x)  # (batch, input_len, d_model)
        
        # Output projection to features
        x = self.output_proj(x)  # (batch, input_len, n_features)
        
        # Skip connection: project temporal dimension
        x_skip = x_skip.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_skip = self.skip_proj(x_skip)  # (batch, n_features, output_len)
        x_skip = x_skip.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Interpolate main path to output_len
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = F.interpolate(x, size=self.output_len, mode='linear', align_corners=False)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Combine
        x = x + x_skip  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class TimesBlockLite(nn.Module):
    def __init__(self, d_model, d_ff, seq_len, top_k, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.top_k = top_k
        
        # Period discovery
        self.period_fc = nn.Linear(seq_len, seq_len)
        
        # 2D convolution: treat (period, frequency) as spatial dims
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=(1, 1), groups=d_model),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=(1, 1), groups=d_model),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        
        # MLP after conv
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        B, T, C = x.shape
        
        # Compute period-based importance via FFT
        xf = torch.fft.rfft(x.permute(0, 2, 1), dim=-1)  # (batch, d_model, freq)
        freq = torch.abs(xf)
        freq = freq.mean(dim=1)  # (batch, freq)
        
        # Find top-k periods
        _, top_indices = torch.topk(freq, min(self.top_k, freq.shape[-1]), dim=-1)
        
        # Use the most dominant period
        period = top_indices[:, 0].float().mean().item()
        period = max(2, min(int(period), T // 2))
        
        # Reshape to 2D: (batch, d_model, period, T//period)
        if T % period != 0:
            pad_len = period - (T % period)
            x_pad = F.pad(x, (0, 0, 0, pad_len))  # (batch, T+pad, d_model)
        else:
            x_pad = x
            pad_len = 0
        
        T_new = x_pad.shape[1]
        x_2d = x_pad.reshape(B, T_new // period, period, C).permute(0, 3, 2, 1)  # (batch, d_model, period, T//period)
        
        # 2D convolution
        x_2d = self.conv(x_2d)  # (batch, d_model, period, T//period)
        
        # Reshape back
        x_out = x_2d.permute(0, 3, 2, 1).reshape(B, T_new, C)  # (batch, T_new, d_model)
        x_out = x_out[:, :T, :]  # (batch, T, d_model)
        
        # MLP
        x_out = self.norm(x_out + self.mlp(x_out))  # (batch, T, d_model)
        
        return x_out
