import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class TimesNetAdvancedV3_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_periods': [3, 5, 8],
            'dropout': [0.0, 0.3],
            'expansion': [2, 4],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = config or {}
        self.d_model = cfg.get('d_model', 256)
        self.n_periods = cfg.get('n_periods', 5)
        self.dropout_rate = cfg.get('dropout', 0.3)
        self.expansion = cfg.get('expansion', 2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: per-channel linear to d_model
        self.input_proj = nn.Conv1d(self.n_features, self.n_features * self.d_model, kernel_size=1, groups=self.n_features)  # (batch, n_features*d_model, input_len)
        
        # Depthwise-separable convolutions per period for each channel
        self.period_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.d_model, self.d_model * self.expansion, kernel_size=(3, 1), padding=(1, 0), groups=self.d_model),  # (batch, d_model*expansion, period_len, num_periods)
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Conv2d(self.d_model * self.expansion, self.d_model, kernel_size=1),  # (batch, d_model, period_len, num_periods)
                nn.Dropout(self.dropout_rate),
            ) for _ in range(self.n_features)
        ])
        
        # Cross-period attention: lightweight attention across periods for each channel
        self.period_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 4),  # (batch, input_len, n_periods, d_model//4)
                nn.Tanh(),
                nn.Linear(self.d_model // 4, 1),  # (batch, input_len, n_periods, 1)
            ) for _ in range(self.n_features)
        ])
        
        # Output projection per channel
        self.output_proj = nn.Conv1d(self.n_features * self.d_model, self.n_features, kernel_size=1, groups=self.n_features)  # (batch, n_features, input_len)
        
        # Direct prediction head
        self.predictor = nn.Linear(self.input_len, self.output_len)  # (batch, n_features, output_len)
        
        # Residual skip connection
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store residual for skip connection
        x_res = x  # (batch, input_len, n_features)
        
        # Transpose for conv: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Input projection: (batch, n_features*d_model, input_len)
        x_proj = self.input_proj(x)  # (batch, n_features*d_model, input_len)
        
        # Reshape to separate channels: (batch, n_features, d_model, input_len)
        x_proj = x_proj.view(x_proj.size(0), self.n_features, self.d_model, self.input_len)  # (batch, n_features, d_model, input_len)
        
        # Period discovery: compute period lengths via FFT
        xf = torch.fft.rfft(x, dim=-1)  # (batch, n_features, input_len//2+1)
        period_amp = torch.abs(xf)  # (batch, n_features, input_len//2+1)
        period_amp = period_amp[:, :, 1:]  # Remove DC component: (batch, n_features, input_len//2)
        
        # Get top periods
        _, top_indices = torch.topk(period_amp, self.n_periods, dim=-1)  # (batch, n_features, n_periods)
        period_lengths = (self.input_len // (top_indices + 1)).clamp(min=2, max=self.input_len // 2)  # (batch, n_features, n_periods)
        
        # Use mean period lengths across batch for stability
        period_lengths = period_lengths.float().mean(dim=0).long()  # (n_features, n_periods)
        
        # Process each channel independently
        outputs = []
        for c in range(self.n_features):
            # Extract channel: (batch, d_model, input_len)
            x_c = x_proj[:, c, :, :]  # (batch, d_model, input_len)
            
            # Process each period
            period_outs = []
            for p in range(self.n_periods):
                period_len = period_lengths[c, p].item()
                num_periods = self.input_len // period_len
                if num_periods == 0:
                    num_periods = 1
                    period_len = self.input_len
                
                # Ensure we use exactly num_periods * period_len = input_len for consistent shapes
                actual_len = num_periods * period_len
                x_p = x_c[:, :, :actual_len]  # (batch, d_model, num_periods*period_len)
                x_p = x_p.view(x_p.size(0), self.d_model, num_periods, period_len)  # (batch, d_model, num_periods, period_len)
                x_p = x_p.permute(0, 1, 3, 2)  # (batch, d_model, period_len, num_periods)
                
                # Apply depthwise-separable convolution
                x_p = self.period_convs[c](x_p)  # (batch, d_model, period_len, num_periods)
                
                # Reshape back to (batch, d_model, input_len)
                x_p = x_p.permute(0, 1, 3, 2)  # (batch, d_model, num_periods, period_len)
                x_p = x_p.reshape(x_p.size(0), self.d_model, -1)  # (batch, d_model, num_periods*period_len)
                
                # Pad or truncate to exactly input_len
                if x_p.size(-1) < self.input_len:
                    x_p = F.pad(x_p, (0, self.input_len - x_p.size(-1)))  # (batch, d_model, input_len)
                else:
                    x_p = x_p[:, :, :self.input_len]  # (batch, d_model, input_len)
                
                period_outs.append(x_p)
            
            # Stack periods: (batch, n_periods, d_model, input_len)
            periods = torch.stack(period_outs, dim=1)  # (batch, n_periods, d_model, input_len)
            
            # Cross-period attention: attend over periods for each time step
            # Reshape to (batch, input_len, n_periods, d_model)
            periods_attn = periods.permute(0, 3, 1, 2)  # (batch, input_len, n_periods, d_model)
            
            # Compute attention weights: (batch, input_len, n_periods)
            attn_logits = self.period_attention[c](periods_attn)  # (batch, input_len, n_periods, 1)
            attn_weights = F.softmax(attn_logits.squeeze(-1), dim=-1)  # (batch, input_len, n_periods)
            
            # Apply attention: weighted sum over periods
            # periods is (batch, n_periods, d_model, input_len)
            # We want: for each time t, sum over p of attn_weights[b,t,p] * periods[b,p,:,t]
            # Reshape periods to (batch, n_periods, d_model, input_len) -> (batch, n_periods, input_len, d_model)
            periods_perm = periods.permute(0, 1, 3, 2)  # (batch, n_periods, input_len, d_model)
            
            # einsum: btp for attn_weights, bpit for periods_perm -> btd for output
            # b = batch, t = input_len, p = n_periods, i = input_len (must match t), d = d_model
            # Actually we need: attn_weights (b, t, p), periods_perm (b, p, t, d) -> output (b, t, d)
            x_c_out = torch.einsum('btp,bptd->btd', attn_weights, periods_perm)  # (batch, input_len, d_model)
            x_c_out = x_c_out.permute(0, 2, 1)  # (batch, d_model, input_len)
            
            outputs.append(x_c_out)
        
        # Concatenate channels: (batch, n_features*d_model, input_len)
        x_out = torch.cat(outputs, dim=1)  # (batch, n_features*d_model, input_len)
        
        # Output projection: (batch, n_features, input_len)
        x_out = self.output_proj(x_out)  # (batch, n_features, input_len)
        
        # Add residual connection
        x_out = x_out + self.residual_weight * x  # (batch, n_features, input_len)
        
        # Predict future
        x_out = self.predictor(x_out)  # (batch, n_features, output_len)
        
        # Transpose back: (batch, output_len, n_features)
        x_out = x_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Denormalize
        x_out = self.revin(x_out, mode='denorm')  # (batch, output_len, n_features)
        
        return x_out  # (batch, output_len, n_features)
