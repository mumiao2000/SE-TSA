import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class TimesNetHybrid(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'd_ff': [128, 512, 2048],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3],
            'top_k': [1, 3, 5]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.d_ff = self.config.get('d_ff', 512)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        self.top_k = self.config.get('top_k', 3)
        
        self.top_k = min(self.top_k, input_len // 4)
        self.top_k = max(self.top_k, 1)
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Embedding
        self.enc_embedding = nn.Linear(self.n_features, self.d_model)
        
        # TimesBlocks
        self.layers = nn.ModuleList([
            TimesBlock(self.d_model, self.d_ff, self.top_k, input_len, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Prediction layer
        self.projection = nn.Linear(self.d_model, self.n_features)
        
        # Direct linear forecast for residual
        self.linear_forecast = nn.Linear(input_len, output_len)
        
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # Store for residual
        x_raw = x  # (batch, input_len, n_features)
        
        # Instance normalization
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.norm(x)  # (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Get statistics for denorm
        mean = x_raw.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        std = x_raw.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        
        # Normalize
        x = (x_raw - mean) / std  # (batch, input_len, n_features)
        
        # Embedding
        x = self.enc_embedding(x)  # (batch, input_len, d_model)
        
        # TimesNet layers
        for layer in self.layers:
            x = layer(x) + x  # residual connection  # (batch, input_len, d_model)
            x = self.layer_norm(x)  # (batch, input_len, d_model)
        
        # Projection to features
        x = self.projection(x)  # (batch, input_len, n_features)
        
        # Simple linear forecast from encoded representation
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.linear_forecast(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Denormalize
        x = x * std + mean  # (batch, output_len, n_features)
        
        # Add direct linear residual
        direct = x_raw.permute(0, 2, 1)  # (batch, n_features, input_len)
        direct = self.linear_forecast(direct)  # (batch, n_features, output_len)
        direct = direct.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        return self.residual_weight * x + (1 - self.residual_weight) * direct


class TimesBlock(nn.Module):
    def __init__(self, d_model, d_ff, top_k, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.seq_len = seq_len
        
        # Period calculation
        self.period_fc = nn.Linear(d_model, d_model)
        
        # 2D convolutions for each scale
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))
            )
            for _ in range(top_k)
        ])
        
        self.projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        B, T, C = x.shape
        
        # FFT to find periods
        xf = torch.fft.rfft(x, dim=1)  # (batch, seq_len//2+1, d_model)
        frequency_list = torch.abs(xf).mean(dim=0).mean(dim=-1)  # (seq_len//2+1,)
        frequency_list[0] = 0  # remove DC component
        
        # Get top-k periods
        _, top_indices = torch.topk(frequency_list, min(self.top_k, len(frequency_list)))
        periods = [max(T // (idx.item() + 1), 1) for idx in top_indices]
        
        # Ensure we have valid periods
        periods = [p for p in periods if p <= T // 2 and p >= 2]
        if len(periods) == 0:
            periods = [T // 4, T // 2] if T >= 4 else [2]
        
        # Multi-scale processing
        out = []
        for i, period in enumerate(periods[:self.top_k]):
            # Reshape to 2D: (batch, num_periods, period, d_model)
            if T % period != 0:
                # Pad to make divisible
                pad_len = period - (T % period)
                x_padded = F.pad(x, (0, 0, 0, pad_len))  # (batch, T+pad, d_model)
            else:
                x_padded = x
                pad_len = 0
            
            num_periods = x_padded.shape[1] // period
            x_2d = x_padded.reshape(B, num_periods, period, C)  # (batch, num_periods, period, d_model)
            
            # Process each feature dimension separately with conv2d
            x_conv = x_2d.permute(0, 3, 1, 2)  # (batch, d_model, num_periods, period)
            
            # Apply conv per channel group to save parameters
            x_conv_list = []
            for c in range(0, C, max(C // 4, 1)):
                c_end = min(c + max(C // 4, 1), C)
                x_c = x_conv[:, c:c_end, :, :].mean(dim=1, keepdim=True)  # (batch, 1, num_periods, period)
                x_c = self.conv_layers[i % len(self.conv_layers)](x_c)  # (batch, 1, num_periods, period)
                x_conv_list.append(x_c)
            
            x_conv = torch.cat(x_conv_list, dim=1)  # (batch, len, num_periods, period)
            # Average back if needed
            if x_conv.shape[1] > C:
                x_conv = x_conv[:, :C, :, :]
            elif x_conv.shape[1] < C:
                x_conv = x_conv.repeat(1, (C // x_conv.shape[1]) + 1, 1, 1)[:, :C, :, :]
            
            x_conv = x_conv.permute(0, 2, 3, 1)  # (batch, num_periods, period, d_model)
            
            # Reshape back
            x_conv = x_conv.reshape(B, -1, C)[:, :T, :]  # (batch, seq_len, d_model)
            out.append(x_conv)
        
        # Aggregate multi-scale outputs
        if len(out) > 0:
            x_out = torch.stack(out, dim=0).mean(dim=0)  # (batch, seq_len, d_model)
        else:
            x_out = x
        
        x_out = self.projection(x_out)  # (batch, seq_len, d_model)
        return self.dropout(x_out)

