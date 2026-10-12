import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class TimesNetAdvanced(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'top_k': [2, 4, 8],
            'd_ff': [128, 512, 2048],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.top_k = self.config.get('top_k', 4)
        self.d_ff = self.config.get('d_ff', 512)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Embedding
        self.enc_embedding = nn.Linear(self.n_features, self.d_model)
        
        # TimesBlocks
        self.layer = nn.ModuleList([
            TimesBlock(self.input_len, self.output_len, self.d_model, self.d_ff, self.top_k, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Projection
        self.projection = nn.Linear(self.input_len * self.d_model, self.output_len * self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Embedding
        x = self.enc_embedding(x)  # (batch, input_len, d_model)
        
        # TimesNet encoder
        for i in range(self.n_layers):
            x = self.layer[i](x)  # (batch, input_len, d_model)
            x = self.layer_norm(x)  # (batch, input_len, d_model)
        
        # Projection
        x = x.flatten(start_dim=1)  # (batch, input_len * d_model)
        x = self.projection(x)  # (batch, output_len * n_features)
        x = x.view(-1, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class TimesBlock(nn.Module):
    def __init__(self, input_len, output_len, d_model, d_ff, top_k, dropout):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.top_k = top_k
        
        # Period discovery parameters
        self.k = top_k
        
        # 2D convolution with multiple kernel sizes (Inception-style)
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff, dropout),
            InceptionBlock(d_ff, d_model, dropout)
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, input_len, d_model)
        B, T, N = x.shape
        
        # Period discovery via FFT
        xf = torch.fft.rfft(x, dim=1)  # (batch, input_len//2+1, d_model)
        
        # Find top-k periods
        frequency_list = abs(xf).mean(0).mean(-1)  # (input_len//2+1)
        frequency_list[0] = 0  # Remove DC component
        _, top_list = torch.topk(frequency_list, self.k)  # (k,)
        period = (self.input_len / (top_list + 1)).long()
        period = torch.clamp(period, min=2)
        
        # Process each period
        out = []
        for i in range(self.k):
            period_length = period[i].item()
            
            # Reshape to 2D: (batch, num_periods, period_length, d_model)
            if self.input_len % period_length != 0:
                # Pad to make divisible
                pad_len = period_length - (self.input_len % period_length)
                x_pad = F.pad(x, (0, 0, 0, pad_len))  # (batch, padded_len, d_model)
                num_periods = x_pad.shape[1] // period_length
            else:
                x_pad = x
                num_periods = self.input_len // period_length
            
            x_2d = x_pad.reshape(B, num_periods, period_length, N)  # (batch, num_periods, period_length, d_model)
            x_2d = x_2d.permute(0, 3, 1, 2)  # (batch, d_model, num_periods, period_length)
            
            # Apply 2D convolution
            x_conv = self.conv(x_2d)  # (batch, d_model, num_periods, period_length)
            x_conv = x_conv.permute(0, 2, 3, 1)  # (batch, num_periods, period_length, d_model)
            x_conv = x_conv.reshape(B, -1, N)  # (batch, padded_len, d_model)
            x_conv = x_conv[:, :self.input_len, :]  # (batch, input_len, d_model)
            
            out.append(x_conv)
        
        # Aggregate multi-period representations
        out = torch.stack(out, dim=-1).mean(-1)  # (batch, input_len, d_model)
        
        # Residual connection
        res = x + self.dropout(self.norm(out))
        return res


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels//4, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels//4, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        # x: (batch, in_channels, H, W)
        out1 = self.conv1x1(x)  # (batch, out_channels//4, H, W)
        out2 = self.conv3x3(x)  # (batch, out_channels//4, H, W)
        out3 = self.conv5x5(x)  # (batch, out_channels//4, H, W)
        out4 = self.conv7x7(x)  # (batch, out_channels//4, H, W)
        
        out = torch.cat([out1, out2, out3, out4], dim=1)  # (batch, out_channels, H, W)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.dropout(out)
        return out
