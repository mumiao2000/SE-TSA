import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FEDformerLite_RealDomain(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'freq_ratio': [0.25, 0.5, 1.0],
            'kernel_size': [3, 7, 15],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with defaults
        self.d_model = self.config.get('d_model', 512)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.freq_ratio = self.config.get('freq_ratio', 0.5)
        self.kernel_size = self.config.get('kernel_size', 7)
        
        # Ensure kernel_size is odd for symmetric moving average
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.d_model)  # (batch, input_len, d_model)
        
        # Trend extraction: learnable moving average (depthwise separable)
        self.trend_extractor = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)  # (batch, d_model, input_len)
        
        # Frequency feature extraction: number of frequency bins to keep
        self.n_freq = max(1, int(input_len * self.freq_ratio))
        
        # FFT magnitude feature projector
        self.freq_proj = nn.Sequential(
            nn.Linear(self.n_freq, self.d_model),  # (batch, d_model, d_model)
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),  # (batch, d_model, d_model)
        )
        
        # Frequency gating: channel-wise selection of frequency bands
        self.freq_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),  # (batch, d_model, d_model)
            nn.Sigmoid(),
        )
        
        # Real-domain multi-head attention layers
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(self.d_model, self.n_heads, dropout=self.dropout, batch_first=True)
            for _ in range(self.n_layers)
        ])  # (batch, input_len, d_model)
        
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model)
            for _ in range(self.n_layers)
        ])  # (batch, input_len, d_model)
        
        # Feedforward after attention
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 4),  # (batch, input_len, d_model*4)
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model * 4, self.d_model),  # (batch, input_len, d_model)
                nn.Dropout(self.dropout),
            )
            for _ in range(self.n_layers)
        ])  # (batch, input_len, d_model)
        
        self.ff_norms = nn.ModuleList([
            nn.LayerNorm(self.d_model)
            for _ in range(self.n_layers)
        ])  # (batch, input_len, d_model)
        
        # Output projection: predict trend and seasonal components separately
        self.trend_proj = nn.Linear(self.d_model, self.n_features)  # (batch, output_len, n_features)
        self.seasonal_proj = nn.Linear(self.d_model, self.n_features)  # (batch, output_len, n_features)
        
        # Direct linear forecast from input (DLinear-style residual)
        self.direct_forecast = nn.Linear(self.input_len, self.output_len)  # (batch, n_features, output_len)
        
    def extract_trend(self, x):
        # x: (batch, input_len, d_model)
        # Transpose for 1D conv/pool: (batch, d_model, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, d_model, input_len)
        trend = self.trend_extractor(x_perm)  # (batch, d_model, input_len)
        # Pad or trim to match input length
        if trend.shape[-1] != self.input_len:
            # Interpolate to match length
            trend = F.interpolate(trend, size=self.input_len, mode='linear', align_corners=False)  # (batch, d_model, input_len)
        trend = trend.permute(0, 2, 1)  # (batch, input_len, d_model)
        return trend  # (batch, input_len, d_model)
    
    def extract_freq_features(self, x):
        # x: (batch, input_len, d_model)
        # Compute FFT per channel (last dim treated as separate channels)
        x_perm = x.permute(0, 2, 1)  # (batch, d_model, input_len)
        
        # Apply rfft to get frequency components
        x_fft = torch.fft.rfft(x_perm, dim=-1)  # (batch, d_model, n_freq_full)
        
        # Take magnitude (real-valued)
        x_mag = torch.abs(x_fft)  # (batch, d_model, n_freq_full)
        
        # Keep only low frequencies (first n_freq bins)
        x_mag = x_mag[:, :, :self.n_freq]  # (batch, d_model, n_freq)
        
        # Project to d_model dimension
        freq_feat = self.freq_proj(x_mag)  # (batch, d_model, d_model)
        
        # Apply channel-wise gating
        gate = self.freq_gate(x_mag.mean(dim=-1, keepdim=True).permute(0, 2, 1))  # (batch, 1, d_model) -> (batch, d_model, d_model)
        # Actually, let's fix the gate shape
        gate = self.freq_gate(x_mag.mean(dim=-1))  # (batch, d_model)
        gate = gate.unsqueeze(1)  # (batch, 1, d_model)
        
        freq_feat = freq_feat * gate  # (batch, d_model, d_model)
        
        # Pool to get frequency-aware embedding per time step
        # Use as additive bias: project back to time dimension
        freq_feat = freq_feat.mean(dim=-1, keepdim=True)  # (batch, d_model, 1)
        freq_feat = freq_feat.expand(-1, -1, self.input_len)  # (batch, d_model, input_len)
        freq_feat = freq_feat.permute(0, 2, 1)  # (batch, input_len, d_model)
        
        return freq_feat  # (batch, input_len, d_model)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Direct linear forecast (DLinear residual)
        x_direct = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_direct = self.direct_forecast(x_direct)  # (batch, n_features, output_len)
        x_direct = x_direct.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Project to d_model
        x_proj = self.input_proj(x)  # (batch, input_len, d_model)
        
        # Extract trend
        trend = self.extract_trend(x_proj)  # (batch, input_len, d_model)
        
        # Seasonal = original - trend
        seasonal = x_proj - trend  # (batch, input_len, d_model)
        
        # Enhance seasonal with frequency features (real domain)
        freq_feat = self.extract_freq_features(seasonal)  # (batch, input_len, d_model)
        seasonal_enhanced = seasonal + freq_feat  # (batch, input_len, d_model)
        
        # Apply attention on enhanced seasonal component
        attn_out = seasonal_enhanced  # (batch, input_len, d_model)
        for attn_layer, attn_norm, ff_layer, ff_norm in zip(
            self.attn_layers, self.attn_norms, self.ff_layers, self.ff_norms
        ):
            # Self-attention with residual
            attn_residual = attn_out  # (batch, input_len, d_model)
            attn_out, _ = attn_layer(attn_out, attn_out, attn_out)  # (batch, input_len, d_model)
            attn_out = attn_norm(attn_out + attn_residual)  # (batch, input_len, d_model)
            
            # Feedforward with residual
            ff_residual = attn_out  # (batch, input_len, d_model)
            attn_out = ff_layer(attn_out)  # (batch, input_len, d_model)
            attn_out = ff_norm(attn_out + ff_residual)  # (batch, input_len, d_model)
        
        # Combine trend and attended seasonal
        combined = trend + attn_out  # (batch, input_len, d_model)
        
        # Project to output: use last input to predict future
        # Simple approach: use the combined representation to predict each future step
        # We'll use a linear decoder from the full sequence
        
        # Trend forecast: extrapolate trend (simple: last value repeat or linear)
        trend_last = trend[:, -1:, :]  # (batch, 1, d_model)
        trend_forecast = trend_last.repeat(1, self.output_len, 1)  # (batch, output_len, d_model)
        # Add slight linear trend based on recent slope
        if self.input_len > 1:
            trend_slope = (trend[:, -1:, :] - trend[:, -2:-1, :])  # (batch, 1, d_model)
            trend_steps = torch.arange(1, self.output_len + 1, device=x.device).float().view(1, self.output_len, 1)  # (1, output_len, 1)
            trend_forecast = trend_forecast + trend_slope * trend_steps  # (batch, output_len, d_model)
        
        # Seasonal forecast: project from attended features
        # Use attention output to predict future seasonal patterns
        seasonal_forecast = self.seasonal_proj(attn_out[:, -1:, :])  # (batch, 1, n_features)
        seasonal_forecast = seasonal_forecast.repeat(1, self.output_len, 1)  # (batch, output_len, n_features)
        
        # Better: use a learned decoder
        # Project combined features and decode
        combined_flat = combined.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        # Expand to output length with learned transformation
        decoder_input = combined_flat.repeat(1, self.output_len, 1)  # (batch, output_len, d_model)
        # Add positional encoding for temporal ordering
        positions = torch.arange(self.output_len, device=x.device).float().view(1, self.output_len, 1)  # (1, output_len, 1)
        decoder_input = decoder_input + positions * 0.01  # (batch, output_len, d_model)
        
        # Final projections
        trend_out = self.trend_proj(decoder_input)  # (batch, output_len, n_features)
        seasonal_out = self.seasonal_proj(decoder_input)  # (batch, output_len, n_features)
        
        # Combine with direct forecast as residual
        out = trend_out + seasonal_out + x_direct * 0.5  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
