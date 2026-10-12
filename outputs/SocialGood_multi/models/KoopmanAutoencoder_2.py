import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanAutoencoder(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [16, 64, 256],
            'encoder_dim': [64, 256, 1024],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.latent_dim = self.config.get('latent_dim', 64)
        self.encoder_dim = self.config.get('encoder_dim', 256)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Encoder: time series -> Koopman observables
        encoder_layers = []
        in_dim = self.input_len
        for i in range(self.n_layers):
            out_dim = self.encoder_dim if i < self.n_layers - 1 else self.latent_dim
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                encoder_layers.append(nn.LayerNorm(out_dim))
                encoder_layers.append(nn.GELU())
                encoder_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Koopman operator (learned linear dynamics)
        self.koopman = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        
        # Initialize near identity for stability
        with torch.no_grad():
            self.koopman.weight.copy_(torch.eye(self.latent_dim) * 0.99 + torch.randn(self.latent_dim, self.latent_dim) * 0.01)
        
        # Decoder: Koopman observables -> time series
        decoder_layers = []
        in_dim = self.latent_dim
        for i in range(self.n_layers):
            out_dim = self.encoder_dim if i < self.n_layers - 1 else self.output_len
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                decoder_layers.append(nn.LayerNorm(out_dim))
                decoder_layers.append(nn.GELU())
                decoder_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Feature-specific projections
        self.feature_proj = nn.Linear(self.n_features, self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.norm(x)  # (batch, n_features, input_len)
        
        # Encode each feature to Koopman space
        latent = self.encoder(x)  # (batch, n_features, latent_dim)
        
        # Apply Koopman operator for prediction (one step, can be iterated)
        # For direct multi-step: apply matrix power or learned transition
        future_latent = self.koopman(latent)  # (batch, n_features, latent_dim)
        
        # Decode to output length
        output = self.decoder(future_latent)  # (batch, n_features, output_len)
        
        # Feature mixing
        output = output.permute(0, 2, 1)  # (batch, output_len, n_features)
        output = self.feature_proj(output)  # (batch, output_len, n_features)
        
        return output
