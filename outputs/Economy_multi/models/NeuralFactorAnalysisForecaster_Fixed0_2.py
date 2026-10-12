import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralFactorAnalysisForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_factors': [4, 16, 64],
            'factor_dim': [32, 128, 512],
            'n_temporal_layers': [1, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_factors = self.config.get('n_factors', 16)
        self.factor_dim = self.config.get('factor_dim', 128)
        self.n_temporal_layers = self.config.get('n_temporal_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Learnable normalization parameters (more flexible than RevIN)
        self.feature_mean = nn.Parameter(torch.zeros(n_features))
        self.feature_std = nn.Parameter(torch.ones(n_features))
        
        # Factor extraction: from features to latent factors
        # Nonlinear loading matrix
        self.loading_encoder = nn.Sequential(
            nn.Linear(n_features, self.factor_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.factor_dim * 2, self.n_factors * self.factor_dim)
        )
        
        # Factor dynamics: temporal model for each factor
        self.factor_temporal = nn.ModuleList()
        for i in range(self.n_temporal_layers):
            # First layer takes factor_dim, subsequent layers take factor_dim*2 if we don't merge
            # But we merge after each LSTM, so all LSTMs take factor_dim
            self.factor_temporal.append(nn.LSTM(
                input_size=self.factor_dim,
                hidden_size=self.factor_dim,
                batch_first=True,
                bidirectional=True
            ))
        
        # Factor merging: combine bidirectional outputs after each LSTM
        self.factor_merges = nn.ModuleList()
        for _ in range(self.n_temporal_layers):
            self.factor_merges.append(nn.Linear(self.factor_dim * 2, self.factor_dim))
        
        # Factor to feature decoding: nonlinear unmixing
        self.loading_decoder = nn.Sequential(
            nn.Linear(self.n_factors * self.factor_dim, self.factor_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.factor_dim * 2, n_features * output_len)
        )
        
        # Residual connection: direct feature autoregression
        self.residual_proj = nn.Linear(input_len, output_len)
        
        # Feature-specific adjustment
        self.feature_adjust = nn.Parameter(torch.ones(n_features))
        
    def normalize(self, x):
        # x: (batch, seq, n_features)
        return (x - self.feature_mean.unsqueeze(0).unsqueeze(0)) / (self.feature_std.unsqueeze(0).unsqueeze(0).clamp(min=1e-6))
    
    def denormalize(self, x):
        # x: (batch, seq, n_features)
        return x * self.feature_std.unsqueeze(0).unsqueeze(0) + self.feature_mean.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x_norm = self.normalize(x)  # (batch, input_len, n_features)
        
        # Extract factors
        # Process each time step
        batch_size = x_norm.shape[0]
        
        # Encode to factors: (batch, input_len, n_factors, factor_dim)
        factor_encoded = self.loading_encoder(x_norm)  # (batch, input_len, n_factors * factor_dim)
        factor_encoded = factor_encoded.reshape(batch_size, self.input_len, self.n_factors, self.factor_dim)
        
        # Temporal modeling for each factor
        # Reshape: (batch * n_factors, input_len, factor_dim)
        factors = factor_encoded.permute(0, 2, 1, 3)  # (batch, n_factors, input_len, factor_dim)
        factors = factors.reshape(batch_size * self.n_factors, self.input_len, self.factor_dim)
        
        for i, lstm in enumerate(self.factor_temporal):
            factors, _ = lstm(factors)  # (batch*n_factors, input_len, factor_dim*2)
            factors = self.factor_merges[i](factors)  # (batch*n_factors, input_len, factor_dim)
        
        # Take last hidden state for prediction
        factors = factors[:, -1, :]  # (batch*n_factors, factor_dim)
        factors = factors.reshape(batch_size, self.n_factors, self.factor_dim)  # (batch, n_factors, factor_dim)
        factors = factors.reshape(batch_size, self.n_factors * self.factor_dim)  # (batch, n_factors * factor_dim)
        
        # Decode to forecasts
        out = self.loading_decoder(factors)  # (batch, n_features * output_len)
        out = out.reshape(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Residual: direct autoregression per feature
        x_perm = x_norm.permute(0, 2, 1)  # (batch, n_features, input_len)
        res = self.residual_proj(x_perm)  # (batch, n_features, output_len)
        res = res.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Combine with learned weighting
        out = out + res * self.feature_adjust.unsqueeze(0).unsqueeze(0)  # (batch, output_len, n_features)
        
        out = self.denormalize(out)  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

