import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveSpectralFilter(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_bands': [4, 8, 16],
            'filter_order': [2, 6, 12],
            'hidden_dim': [64, 256, 512],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.n_bands = self.config.get('n_bands', 8)
        self.filter_order = self.config.get('filter_order', 6)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Frequency resolution
        self.n_freq = self.input_len // 2 + 1
        
        # Learnable filter bank: center frequencies and bandwidths per band per feature
        self.filter_centers = nn.Parameter(torch.rand(self.n_features, self.n_bands))
        self.filter_widths = nn.Parameter(torch.ones(self.n_features, self.n_bands) * 0.1)
        
        # Gating network: predicts which bands to use
        self.gate_encoder = nn.Sequential(
            nn.Linear(self.input_len, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
        )
        self.gate_predictor = nn.Linear(self.hidden_dim // 2, self.n_bands)
        
        # Feature processing after filtering
        self.feature_processor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.n_freq, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            ) for _ in range(self.n_features)
        ])
        
        # Output heads per feature
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim // 2, self.output_len)
            ) for _ in range(self.n_features)
        ])
        
        # Global mixing
        self.global_mixer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.dropout,
            batch_first=True
        )
        
    def _create_bandpass_filters(self):
        # Create smooth bandpass filters in frequency domain
        # Returns: (n_features, n_bands, n_freq)
        freq_grid = torch.linspace(0, 1, self.n_freq, device=self.filter_centers.device)  # (n_freq,)
        
        # Normalize centers to [0, 1]
        centers = torch.sigmoid(self.filter_centers)  # (n_features, n_bands)
        widths = F.softplus(self.filter_widths) + 0.01  # (n_features, n_bands)
        
        # Gaussian bandpass: exp(-((f - c) / w)^2)
        freq_grid = freq_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, n_freq)
        centers = centers.unsqueeze(-1)  # (n_features, n_bands, 1)
        widths = widths.unsqueeze(-1)  # (n_features, n_bands, 1)
        
        filters = torch.exp(-((freq_grid - centers) / widths) ** 2)  # (n_features, n_bands, n_freq)
        return filters
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Per-feature FFT
        x_permuted = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_fft = torch.fft.rfft(x_permuted, dim=-1, norm='ortho')  # (batch, n_features, n_freq)
        x_fft_mag = torch.abs(x_fft)  # (batch, n_features, n_freq)
        x_fft_phase = torch.angle(x_fft)  # (batch, n_features, n_freq)
        
        # Get adaptive filters
        filters = self._create_bandpass_filters()  # (n_features, n_bands, n_freq)
        
        # Apply filters: (batch, n_features, n_bands, n_freq)
        x_filtered = x_fft_mag.unsqueeze(2) * filters.unsqueeze(0)  # (batch, n_features, n_bands, n_freq)
        
        # Compute gating based on input statistics
        x_stats = x_permuted.mean(dim=-1)  # (batch, n_features)
        gate_input = x_permuted.reshape(batch_size * self.n_features, self.input_len)  # (batch*n_features, input_len)
        gate_features = self.gate_encoder(gate_input)  # (batch*n_features, hidden_dim//2)
        band_gates = torch.sigmoid(self.gate_predictor(gate_features))  # (batch*n_features, n_bands)
        band_gates = band_gates.reshape(batch_size, self.n_features, self.n_bands)  # (batch, n_features, n_bands)
        
        # Apply gating and sum bands
        x_gated = x_filtered * band_gates.unsqueeze(-1)  # (batch, n_features, n_bands, n_freq)
        x_bands = x_gated.sum(dim=2)  # (batch, n_features, n_freq)
        
        # Process each feature through its MLP
        feature_outputs = []
        for f in range(self.n_features):
            feat = x_bands[:, f, :]  # (batch, n_freq)
            feat = self.feature_processor[f](feat)  # (batch, hidden_dim)
            feature_outputs.append(feat)
        
        features = torch.stack(feature_outputs, dim=1)  # (batch, n_features, hidden_dim)
        
        # Global mixing across features
        features = self.global_mixer(features)  # (batch, n_features, hidden_dim)
        
        # Output per feature
        outputs = []
        for f in range(self.n_features):
            feat = features[:, f, :]  # (batch, hidden_dim)
            out = self.output_heads[f](feat)  # (batch, output_len)
            outputs.append(out)
        
        y = torch.stack(outputs, dim=-1)  # (batch, output_len, n_features)
        
        y = self.revin(y, mode='denorm')  # (batch, output_len, n_features)
        return y
