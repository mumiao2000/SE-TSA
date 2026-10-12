import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class TimesNetAdvancedV3(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 2048],
            'd_ff': [256, 1024, 4096],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.3],
            'top_k': [2, 4, 8],
            'channel_mix_dim': [64, 256, 1024],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 512)
        self.d_ff = cfg.get('d_ff', 1024)
        self.n_layers = cfg.get('n_layers', 2)
        self.dropout = cfg.get('dropout', 0.1)
        self.top_k = cfg.get('top_k', 4)
        self.channel_mix_dim = cfg.get('channel_mix_dim', 256)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input embedding
        self.enc_embedding = nn.Linear(self.n_features, self.d_model)  # (batch, input_len, d_model)
        
        # TimesBlocks with multi-scale period discovery and gated convolutions
        self.times_blocks = nn.ModuleList([
            TimesBlockV3(
                input_len=self.input_len,
                output_len=self.output_len,
                d_model=self.d_model,
                d_ff=self.d_ff,
                top_k=self.top_k,
                dropout=self.dropout
            ) for _ in range(self.n_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Channel-mixing projection for cross-feature dependencies
        self.channel_mixer = nn.Sequential(
            nn.Linear(self.d_model, self.channel_mix_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.channel_mix_dim, self.d_model)
        )
        
        # Output projection
        self.projection = nn.Linear(self.d_model, self.n_features)  # (batch, output_len, n_features)
        
        # Direct residual connection from input to output for trend preservation
        self.trend_proj = nn.Linear(self.input_len, self.output_len)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Save for trend residual
        x_input = x  # (batch, input_len, n_features)
        
        # Embedding
        x = self.enc_embedding(x)  # (batch, input_len, d_model)
        
        # TimesBlocks with residual connections
        for i, block in enumerate(self.times_blocks):
            x = x + block(x)  # (batch, input_len, d_model)
            x = self.layer_norm(x)  # (batch, input_len, d_model)
        
        # Channel mixing for cross-feature dependencies
        x = x + self.channel_mixer(x)  # (batch, input_len, d_model)
        x = self.layer_norm(x)  # (batch, input_len, d_model)
        
        # Project to output length and features
        # Use the last output_len positions or project spatially
        if self.input_len >= self.output_len:
            x = x[:, -self.output_len:, :]  # (batch, output_len, d_model)
        else:
            # Interpolate or repeat if needed
            x = F.interpolate(
                x.transpose(1, 2),  # (batch, d_model, input_len)
                size=self.output_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # (batch, output_len, d_model)
        
        x = self.projection(x)  # (batch, output_len, n_features)
        
        # Add trend residual from input
        trend = x_input.permute(0, 2, 1)  # (batch, n_features, input_len)
        trend = self.trend_proj(trend)  # (batch, n_features, output_len)
        trend = trend.permute(0, 2, 1)  # (batch, output_len, n_features)
        x = x + trend * 0.1  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)


class TimesBlockV3(nn.Module):
    def __init__(self, input_len, output_len, d_model, d_ff, top_k, dropout):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.top_k = top_k
        
        # Period discovery
        self.period_fc = nn.Linear(d_model, d_model)
        
        # Multi-scale 2D convolutions with gating
        self.conv_layers = nn.ModuleList()
        self.gates = nn.ModuleList()
        for _ in range(top_k):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(d_model, d_ff // 2, kernel_size=(1, 3), padding=(0, 1)),
                    nn.BatchNorm2d(d_ff // 2),
                    nn.ReLU(),
                    nn.Conv2d(d_ff // 2, d_ff // 2, kernel_size=(3, 1), padding=(1, 0)),
                    nn.BatchNorm2d(d_ff // 2),
                    nn.ReLU(),
                    nn.Conv2d(d_ff // 2, d_model, kernel_size=(1, 3), padding=(0, 1)),
                    nn.BatchNorm2d(d_model),
                )
            )
            # Gating mechanism for adaptive filtering
            self.gates.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(d_model, d_model // 4),
                    nn.ReLU(),
                    nn.Linear(d_model // 4, d_model),
                    nn.Sigmoid()
                )
            )
        
        # Period importance weighting
        self.period_importance = nn.Parameter(torch.ones(top_k))
        
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        B, T, D = x.shape  # batch, seq_len, d_model
        
        # Period discovery via FFT
        x_for_fft = self.period_fc(x)  # (batch, seq_len, d_model)
        xf = torch.fft.rfft(x_for_fft, dim=1)  # (batch, freq_len, d_model)
        
        # Find top-k periods
        frequency_list = torch.abs(xf).mean(0).mean(-1)  # (freq_len,)
        frequency_list[0] = 0  # Exclude DC component
        _, top_list = torch.topk(frequency_list, self.top_k)  # (top_k,)
        period_list = (T / (top_list + 1e-5)).long().clamp(min=1, max=T)  # (top_k,)
        
        # Multi-scale processing
        res = []
        for i in range(self.top_k):
            period = period_list[i].item()
            
            # Pad to make divisible by period
            if T % period != 0:
                pad_len = period - (T % period)
                x_padded = F.pad(x, (0, 0, 0, pad_len))  # (batch, seq_len+pad, d_model)
            else:
                x_padded = x
                pad_len = 0
            
            new_T = x_padded.shape[1]
            n_period = new_T // period
            
            # Reshape to 2D: (batch, d_model, n_period, period)
            x_2d = x_padded.reshape(B, n_period, period, D).permute(0, 3, 1, 2)  # (batch, d_model, n_period, period)
            
            # Apply gated convolution
            conv_out = self.conv_layers[i](x_2d)  # (batch, d_model, n_period, period)
            
            # Apply channel-wise gate
            gate = self.gates[i](conv_out)  # (batch, d_model)
            gate = gate.view(B, D, 1, 1)  # (batch, d_model, 1, 1)
            conv_out = conv_out * gate  # (batch, d_model, n_period, period)
            
            # Reshape back
            conv_out = conv_out.permute(0, 2, 3, 1).reshape(B, new_T, D)  # (batch, new_T, d_model)
            
            # Remove padding
            if pad_len > 0:
                conv_out = conv_out[:, :T, :]  # (batch, T, d_model)
            
            res.append(conv_out)
        
        # Weighted aggregation across periods
        importance = F.softmax(self.period_importance, dim=0)  # (top_k,)
        out = sum(w * r for w, r in zip(importance, res))  # (batch, T, d_model)
        
        out = self.dropout(out)
        out = self.projection(out)  # (batch, T, d_model)
        return out  # (batch, T, d_model)
