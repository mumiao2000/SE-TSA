import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FEDformerLite_RealDomain_v2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'freq_dim': [16, 64, 256],
            'kernel_size': [3, 13, 25],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config with wide exponential spacing
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.freq_dim = self.config.get('freq_dim', 64)
        self.kernel_size = self.config.get('kernel_size', 13)
        
        # Ensure kernel_size is odd for symmetric padding
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Moving average decomposition (trend extraction)
        self.trend_extractor = nn.AvgPool1d(
            kernel_size=self.kernel_size, 
            stride=1, 
            padding=self.kernel_size // 2
        )
        
        # Feature projection to d_model
        self.input_projection = nn.Linear(self.n_features, self.d_model)
        
        # Frequency feature extraction for seasonal branch
        # Real-valued FFT magnitude features
        self.freq_projection = nn.Sequential(
            nn.Linear(self.n_features, self.freq_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.freq_dim, self.d_model)
        )
        
        # Lightweight attention: only on frequency features, not full sequence
        # Use efficient local attention with reduced complexity
        self.freq_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=max(1, self.d_model // 64),  # adaptive heads
                dim_feedforward=self.d_model * 2,
                dropout=self.dropout,
                batch_first=True
            ) for _ in range(max(1, self.n_layers // 2))
        ])
        
        # Channel-wise linear projections for trend (validated from DLinearDecomp)
        # Individual linear layer per feature for trend
        self.trend_linears = nn.ModuleList([
            nn.Linear(input_len, output_len) for _ in range(self.n_features)
        ])
        
        # Seasonal prediction with frequency enhancement
        # Shared MLP with frequency conditioning
        self.seasonal_predictor = nn.Sequential(
            nn.Linear(input_len * self.d_model, self.d_model * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model * 2, output_len * self.n_features)
        )
        
        # Adaptive fusion gate: learn to combine trend and seasonal
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.n_features * 2, self.n_features),
            nn.Sigmoid()
        )
        
        # Direct residual connection for stability
        self.residual_proj = nn.Linear(input_len, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # === TREND EXTRACTION ===
        # Moving average in time dimension
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        trend = self.trend_extractor(x_perm)  # (batch, n_features, input_len)
        trend = trend.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Seasonal = input - trend
        seasonal = x - trend  # (batch, input_len, n_features)
        
        # === TREND PREDICTION: Channel-wise linear (lightweight) ===
        trend_out = []
        for i in range(self.n_features):
            feat = trend[:, :, i]  # (batch, input_len)
            pred = self.trend_linears[i](feat)  # (batch, output_len)
            trend_out.append(pred.unsqueeze(-1))  # (batch, output_len, 1)
        trend_pred = torch.cat(trend_out, dim=-1)  # (batch, output_len, n_features)
        
        # === SEASONAL PREDICTION: Frequency-enhanced ===
        # Extract FFT magnitude features
        seasonal_fft = torch.fft.rfft(seasonal, dim=1)  # (batch, input_len//2+1, n_features)
        fft_magnitude = torch.abs(seasonal_fft)  # Real-valued magnitude
        
        # Take top frequencies (low-freq dominant for seasonality)
        fft_magnitude = fft_magnitude[:, :min(fft_magnitude.shape[1], self.input_len // 2 + 1), :]  # (batch, freq_len, n_features)
        
        # Pool frequency features across time
        fft_pooled = fft_magnitude.mean(dim=1)  # (batch, n_features)
        
        # Project to frequency embedding
        freq_embed = self.freq_projection(fft_pooled)  # (batch, d_model)
        
        # Project seasonal to d_model space
        seasonal_proj = self.input_projection(seasonal)  # (batch, input_len, d_model)
        
        # Add frequency conditioning (broadcast across time)
        seasonal_enhanced = seasonal_proj + freq_embed.unsqueeze(1)  # (batch, input_len, d_model)
        
        # Lightweight attention on frequency-enhanced seasonal
        attn_out = seasonal_enhanced  # (batch, input_len, d_model)
        for attn_layer in self.freq_attention_layers:
            attn_out = attn_layer(attn_out)  # (batch, input_len, d_model)
        
        # Flatten and predict seasonal output
        attn_flat = attn_out.reshape(batch_size, -1)  # (batch, input_len * d_model)
        seasonal_flat = self.seasonal_predictor(attn_flat)  # (batch, output_len * n_features)
        seasonal_pred = seasonal_flat.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # === RESIDUAL CONNECTION ===
        # Direct linear mapping from input to output
        x_for_res = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        res_out = self.residual_proj(x_for_res)  # (batch, n_features, output_len)
        res_pred = res_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # === ADAPTIVE FUSION ===
        # Compute gate based on input statistics
        x_stats = torch.cat([
            x.mean(dim=1),  # (batch, n_features)
            x.std(dim=1)    # (batch, n_features)
        ], dim=-1)  # (batch, n_features * 2)
        gate = self.fusion_gate(x_stats)  # (batch, n_features)
        gate = gate.unsqueeze(1)  # (batch, 1, n_features)
        
        # Combine: gate * trend + (1-gate) * seasonal + residual
        combined = gate * trend_pred + (1 - gate) * seasonal_pred + 0.1 * res_pred  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(combined, mode='denorm')  # (batch, output_len, n_features)
        
        return out
