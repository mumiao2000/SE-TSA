import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanNeuralOperator(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [64, 256, 1024],
            'koopman_dim': [32, 128, 512],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 256)
        self.koopman_dim = self.config.get('koopman_dim', 128)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: time-series -> latent observables
        encoder_layers = []
        in_dim = input_len
        for i in range(self.n_layers):
            out_dim = self.latent_dim if i == self.n_layers - 1 else self.latent_dim // 2
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                encoder_layers.append(nn.GELU())
                encoder_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Lift to Koopman space
        self.to_koopman = nn.Linear(self.latent_dim, self.koopman_dim)
        
        # Koopman operator (learned, non-symmetric for general dynamics)
        self.koopman_K = nn.Parameter(torch.randn(self.koopman_dim, self.koopman_dim) * 0.01)
        self.koopman_bias = nn.Parameter(torch.zeros(self.koopman_dim))
        
        # Eigenvalue stabilization (learned damping)
        self.koopman_damping = nn.Parameter(torch.ones(self.koopman_dim) * 0.9)
        
        # Decoder: Koopman -> latent -> time-series
        self.from_koopman = nn.Linear(self.koopman_dim, self.latent_dim)
        
        decoder_layers = []
        in_dim = self.latent_dim
        for i in range(self.n_layers):
            out_dim = output_len if i == self.n_layers - 1 else self.latent_dim // 2
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                decoder_layers.append(nn.GELU())
                decoder_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Direct residual path
        self.residual = nn.Linear(input_len, output_len)
        
        # Channel mixing after Koopman evolution
        self.channel_mix = nn.Sequential(
            nn.Linear(n_features, n_features * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(n_features * 2, n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        residual = self.residual(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Process per channel: (batch, n_features, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Encode to latent
        z = self.encoder(x_perm)  # (batch, n_features, latent_dim)
        
        # Lift to Koopman space
        k = self.to_koopman(z)  # (batch, n_features, koopman_dim)
        
        # Apply Koopman evolution (linear dynamics)
        # K with damping for stability
        K_damped = self.koopman_K * torch.sigmoid(self.koopman_damping).unsqueeze(0)  # (1, koopman_dim, koopman_dim)
        
        # Evolution: k_evolved = k @ K_damped + bias
        k_evolved = torch.matmul(k, K_damped) + self.koopman_bias.unsqueeze(0).unsqueeze(0)  # (batch, n_features, koopman_dim)
        
        # Add nonlinear correction (residual around linear dynamics)
        k_evolved = k_evolved + F.gelu(k_evolved) * 0.1  # small nonlinear correction
        
        # Decode back
        z_decoded = self.from_koopman(k_evolved)  # (batch, n_features, latent_dim)
        
        # Decode to time
        out = self.decoder(z_decoded)  # (batch, n_features, output_len)
        
        # Channel mixing
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        out = self.channel_mix(out)  # (batch, output_len, n_features)
        
        # Add residual
        out = out + residual  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

