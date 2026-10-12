import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FEDformerStyle_ChannelAware_Improved(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'modes_ratio': [0.125, 0.25, 0.5],
            'trend_dim': [16, 64, 256],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.d_model = self.config.get('d_model', 512)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.modes_ratio = self.config.get('modes_ratio', 0.25)
        self.trend_dim = self.config.get('trend_dim', 64)
        
        # Ensure d_model divisible by n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads
        
        # Adaptive modes: ratio of input_len//2 + 1
        self.adaptive_modes = max(1, int((input_len // 2 + 1) * self.modes_ratio))
        
        # Feature grouping based on semantic types from column names
        # Group 0: target (OT) - 1 feature
        # Group 1: metadata (CBSA Code, Number of Sites Reporting, start_date, end_date) - 4 features
        # Group 2: historical stats (his_avg_*, his_std_*, prior_history_*) - 16 features
        # Group 3: SVD embeddings (all *_svd_*) - 20 features
        self.n_groups = 4
        self.group_sizes = [1, 4, 16, 20]
        # Verify sum matches n_features
        assert sum(self.group_sizes) == self.n_features, f"Group sizes {sum(self.group_sizes)} != n_features {self.n_features}"
        
        # Group indices for splitting
        self.group_indices = []
        idx = 0
        for gs in self.group_sizes:
            self.group_indices.append((idx, idx + gs))
            idx += gs
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection per group
        self.input_proj = nn.ModuleList()
        for gs in self.group_sizes:
            self.input_proj.append(nn.Linear(gs, self.d_model))
        
        # Trend extraction: moving average decomposition per group
        self.trend_extractors = nn.ModuleList()
        for _ in range(self.n_groups):
            self.trend_extractors.append(
                nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
            )
        
        # Trend projection per group
        self.trend_proj = nn.ModuleList()
        for _ in range(self.n_groups):
            self.trend_proj.append(nn.Sequential(
                nn.Linear(self.d_model, self.trend_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.trend_dim, self.d_model)
            ))
        
        # Frequency attention blocks per group
        self.freq_blocks = nn.ModuleList()
        for _ in range(self.n_groups):
            group_blocks = nn.ModuleList()
            for _ in range(self.n_layers):
                group_blocks.append(
                    FreqEnhancedBlock_Adaptive(
                        d_model=self.d_model,
                        n_heads=self.n_heads,
                        modes_ratio=self.modes_ratio,
                        dropout=self.dropout,
                        input_len=input_len
                    )
                )
            self.freq_blocks.append(group_blocks)
        
        # Time-domain MLP per group for local patterns
        self.time_mlps = nn.ModuleList()
        for _ in range(self.n_groups):
            self.time_mlps.append(nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * 2, self.d_model)
            ))
        
        # Gating mechanism: blend freq and time domains
        self.gates = nn.ModuleList()
        for _ in range(self.n_groups):
            self.gates.append(nn.Linear(self.d_model * 2, 2))
        
        # Layer norms
        self.norms = nn.ModuleList()
        for _ in range(self.n_groups):
            self.norms.append(nn.LayerNorm(self.d_model))
        
        # Output projection per group
        self.output_proj = nn.ModuleList()
        for gs in self.group_sizes:
            self.output_proj.append(nn.Linear(self.d_model, gs))
        
        # Final prediction: project to output_len
        self.predictors = nn.ModuleList()
        for gs in self.group_sizes:
            self.predictors.append(nn.Linear(input_len, output_len))
        
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Split into groups
        group_inputs = []
        for start, end in self.group_indices:
            group_inputs.append(x[:, :, start:end])  # (batch, input_len, group_size)
        
        # Process each group
        group_outputs = []
        for g in range(self.n_groups):
            # Input projection: (batch, input_len, group_size) -> (batch, input_len, d_model)
            x_g = self.input_proj[g](group_inputs[g])  # (batch, input_len, d_model)
            
            # Extract trend: (batch, input_len, d_model) -> (batch, d_model, input_len) -> pool -> (batch, d_model, input_len) -> transpose
            x_g_t = x_g.permute(0, 2, 1)  # (batch, d_model, input_len)
            trend_t = self.trend_extractors[g](x_g_t)  # (batch, d_model, input_len)
            trend = trend_t.permute(0, 2, 1)  # (batch, input_len, d_model)
            
            # Seasonal = input - trend
            seasonal = x_g - trend  # (batch, input_len, d_model)
            
            # Process trend through MLP
            trend_processed = self.trend_proj[g](trend)  # (batch, input_len, d_model)
            
            # Process seasonal through frequency blocks
            seasonal_freq = seasonal  # (batch, input_len, d_model)
            for block in self.freq_blocks[g]:
                seasonal_freq = block(seasonal_freq)  # (batch, input_len, d_model)
                seasonal_freq = self.dropout_layer(seasonal_freq)
            
            # Time-domain pathway
            seasonal_time = self.time_mlps[g](seasonal)  # (batch, input_len, d_model)
            
            # Gating: blend freq and time
            gate_input = torch.cat([seasonal_freq, seasonal_time], dim=-1)  # (batch, input_len, d_model*2)
            gate = F.softmax(self.gates[g](gate_input), dim=-1)  # (batch, input_len, 2)
            
            seasonal_blended = gate[:, :, 0:1] * seasonal_freq + gate[:, :, 1:2] * seasonal_time  # (batch, input_len, d_model)
            
            # Combine trend + seasonal
            combined = trend_processed + seasonal_blended  # (batch, input_len, d_model)
            combined = self.norms[g](combined)  # (batch, input_len, d_model)
            
            # Output projection: (batch, input_len, d_model) -> (batch, input_len, group_size)
            combined = self.output_proj[g](combined)  # (batch, input_len, group_size)
            
            # Predict to output_len: (batch, input_len, group_size) -> (batch, output_len, group_size)
            combined = combined.permute(0, 2, 1)  # (batch, group_size, input_len)
            pred = self.predictors[g](combined)  # (batch, group_size, output_len)
            pred = pred.permute(0, 2, 1)  # (batch, output_len, group_size)
            
            group_outputs.append(pred)
        
        # Concatenate groups
        out = torch.cat(group_outputs, dim=-1)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out


class FreqEnhancedBlock_Adaptive(nn.Module):
    def __init__(self, d_model, n_heads, modes_ratio, dropout, input_len):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.modes_ratio = modes_ratio
        
        # Adaptive modes based on input length
        self.max_modes = input_len // 2 + 1
        self.modes = max(1, int(self.max_modes * modes_ratio))
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Learnable frequency filters per head (complex)
        self.freq_real = nn.Parameter(torch.randn(n_heads, self.modes, self.head_dim) * 0.02)
        self.freq_imag = nn.Parameter(torch.randn(n_heads, self.modes, self.head_dim) * 0.02)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)  # (batch, seq_len, d_model)
        v = self.v_proj(x)  # (batch, seq_len, d_model)
        
        # Reshape to multi-head: (batch, n_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply frequency attention
        # rFFT on last dimension (head_dim)
        q_fft = torch.fft.rfft(q, dim=-1)  # (batch, n_heads, seq_len, head_dim//2+1)
        k_fft = torch.fft.rfft(k, dim=-1)  # (batch, n_heads, seq_len, head_dim//2+1)
        
        # Get actual frequency modes for this sequence
        actual_modes = min(self.modes, q_fft.shape[-1])
        
        # Truncate to adaptive modes
        q_fft = q_fft[..., :actual_modes]  # (batch, n_heads, seq_len, actual_modes)
        k_fft = k_fft[..., :actual_modes]  # (batch, n_heads, seq_len, actual_modes)
        
        # Apply learnable filters (truncate if needed)
        freq_real = self.freq_real[:, :actual_modes, :actual_modes] if actual_modes < self.freq_real.shape[1] else self.freq_real[:, :actual_modes, :]
        freq_imag = self.freq_imag[:, :actual_modes, :actual_modes] if actual_modes < self.freq_imag.shape[1] else self.freq_imag[:, :actual_modes, :]
        
        # Ensure filter shape matches: need (n_heads, actual_modes, actual_modes)
        # Pad or truncate freq filters to match actual_modes in both dimensions
        if freq_real.shape[1] != actual_modes or freq_real.shape[2] != actual_modes:
            freq_real = F.interpolate(freq_real.unsqueeze(0), size=(actual_modes, actual_modes), mode='bilinear', align_corners=False).squeeze(0)
            freq_imag = F.interpolate(freq_imag.unsqueeze(0), size=(actual_modes, actual_modes), mode='bilinear', align_corners=False).squeeze(0)
        
        # Apply filter: (batch, n_heads, seq_len, actual_modes) @ (n_heads, actual_modes, actual_modes)
        # Use einsum for clarity
        q_filtered_real = torch.einsum('bhsf,hfg->bhsg', q_fft.real, freq_real) - torch.einsum('bhsf,hfg->bhsg', q_fft.imag, freq_imag)
        q_filtered_imag = torch.einsum('bhsf,hfg->bhsg', q_fft.real, freq_imag) + torch.einsum('bhsf,hfg->bhsg', q_fft.imag, freq_real)
        q_filtered = torch.complex(q_filtered_real, q_filtered_imag)
        
        k_filtered_real = torch.einsum('bhsf,hfg->bhsg', k_fft.real, freq_real) - torch.einsum('bhsf,hfg->bhsg', k_fft.imag, freq_imag)
        k_filtered_imag = torch.einsum('bhsf,hfg->bhsg', k_fft.real, freq_imag) + torch.einsum('bhsf,hfg->bhsg', k_fft.imag, freq_real)
        k_filtered = torch.complex(k_filtered_real, k_filtered_imag)
        
        # Attention in frequency domain
        scores = torch.einsum('bhqd,bhkd->bhqk', q_filtered, k_filtered.conj())  # (batch, n_heads, seq_len, seq_len)
        scores = scores / (actual_modes ** 0.5)
        attn = F.softmax(scores.real, dim=-1)  # (batch, n_heads, seq_len, seq_len)
        
        # Apply attention to v (in time domain, simplified)
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)  # (batch, n_heads, seq_len, head_dim)
        
        # Reshape back
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)
        
        # Output projection with residual
        out = self.out_proj(out)  # (batch, seq_len, d_model)
        out = self.dropout(out)
        out = self.norm(x + out)  # (batch, seq_len, d_model)
        
        return out
