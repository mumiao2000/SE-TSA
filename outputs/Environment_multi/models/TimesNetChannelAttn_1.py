import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class TimesNetChannelAttn(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.3],
            'n_kernels': [2, 4, 8],
            'd_ff': [128, 512, 2048]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_kernels = self.config.get('n_kernels', 4)
        self.d_ff = self.config.get('d_ff', 512)
        
        self.revin = RevIN(n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(n_features, self.d_model)
        
        # TimesNet blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.blocks.append(
                TimesBlock(self.d_model, self.d_ff, self.n_kernels, input_len, self.dropout)
            )
        
        # Channel attention for feature importance
        self.channel_attn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, self.d_model),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, n_features)
        
        # Predictor
        self.predictor = nn.Linear(input_len, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, input_len, d_model)
        
        # TimesNet blocks
        for block in self.blocks:
            x = x + block(x)  # (batch, input_len, d_model)
        
        # Channel attention
        attn = self.channel_attn(x.mean(dim=1, keepdim=True))  # (batch, 1, d_model)
        x = x * attn  # (batch, input_len, d_model)
        
        # Predict
        x = x.permute(0, 2, 1)  # (batch, d_model, input_len)
        x = self.predictor(x)  # (batch, d_model, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, d_model)
        
        x = self.output_proj(x)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x


class TimesBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_kernels, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_kernels = n_kernels
        self.seq_len = seq_len
        
        # Period extraction
        self.period_fc = nn.Linear(seq_len, n_kernels)
        
        # 2D convolutions for each period
        self.convs = nn.ModuleList()
        for _ in range(n_kernels):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 1, kernel_size=(3, 3), padding=1)
                )
            )
        
        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(d_model * n_kernels, d_ff),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Compute period weights
        x_for_period = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
        period_weights = self.period_fc(x_for_period)  # (batch, d_model, n_kernels)
        period_weights = F.softmax(period_weights, dim=-1)  # (batch, d_model, n_kernels)
        
        # Get top periods
        _, top_indices = torch.topk(period_weights.mean(dim=1), self.n_kernels, dim=-1)  # (batch, n_kernels)
        
        # Process each period
        outputs = []
        for i in range(self.n_kernels):
            # Use learned period length (simplified: use fixed periods based on indices)
            period_len = 2 + i * 2  # Simple period lengths: 2, 4, 6, ...
            if period_len > seq_len // 2:
                period_len = max(2, seq_len // 4)
            
            # Pad to multiple of period_len
            pad_len = (period_len - seq_len % period_len) % period_len
            x_padded = F.pad(x, (0, 0, 0, pad_len))  # (batch, seq_len+pad, d_model)
            
            new_len = x_padded.shape[1]
            n_periods = new_len // period_len
            
            # Reshape to 2D: (batch, d_model, n_periods, period_len)
            x_2d = x_padded.reshape(batch_size, d_model, n_periods, period_len)
            
            # Process each feature independently with 2D conv
            feat_outputs = []
            for j in range(d_model):
                feat = x_2d[:, j:j+1, :, :]  # (batch, 1, n_periods, period_len)
                feat = self.convs[i](feat)  # (batch, 1, n_periods, period_len)
                feat_outputs.append(feat.squeeze(1))  # (batch, n_periods, period_len)
            
            x_out = torch.stack(feat_outputs, dim=1)  # (batch, d_model, n_periods, period_len)
            x_out = x_out.reshape(batch_size, d_model, -1)  # (batch, d_model, n_periods * period_len)
            x_out = x_out[:, :, :seq_len]  # (batch, d_model, seq_len)
            x_out = x_out.permute(0, 2, 1)  # (batch, seq_len, d_model)
            
            outputs.append(x_out)
        
        # Concatenate and project
        x = torch.cat(outputs, dim=-1)  # (batch, seq_len, d_model * n_kernels)
        x = self.proj(x)  # (batch, seq_len, d_model)
        x = self.norm(x)
        
        return x
