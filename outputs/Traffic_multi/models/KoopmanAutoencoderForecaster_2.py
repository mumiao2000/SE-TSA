import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanAutoencoderForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [64, 256, 1024],
            'encoder_width': [128, 512, 2048],
            'n_encoder_layers': [2, 4, 8],
            'koopman_regularization': [0.0, 0.01, 0.1],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 256)
        self.encoder_width = self.config.get('encoder_width', 512)
        self.n_encoder_layers = self.config.get('n_encoder_layers', 4)
        self.koopman_reg = self.config.get('koopman_regularization', 0.01)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Build encoder: (batch, input_len, n_features) -> (batch, latent_dim)
        encoder_layers = []
        in_dim = input_len * n_features
        for i in range(self.n_encoder_layers):
            out_dim = self.encoder_width if i < self.n_encoder_layers - 1 else self.latent_dim
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < self.n_encoder_layers - 1 else nn.Identity(),
                nn.GELU() if i < self.n_encoder_layers - 1 else nn.Identity(),
                nn.Dropout(self.dropout) if i < self.n_encoder_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Koopman operator: linear dynamics in latent space
        # K matrix for one-step prediction
        self.K = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        
        # Build decoder: (batch, latent_dim) -> (batch, output_len, n_features)
        decoder_layers = []
        in_dim = self.latent_dim
        for i in range(self.n_encoder_layers):
            out_dim = self.encoder_width if i < self.n_encoder_layers - 1 else output_len * n_features
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < self.n_encoder_layers - 1 else nn.Identity(),
                nn.GELU() if i < self.n_encoder_layers - 1 else nn.Identity(),
                nn.Dropout(self.dropout) if i < self.n_encoder_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Optional: learned initial state correction
        self.init_correction = nn.Linear(input_len * n_features, self.latent_dim)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        B = x.shape[0]
        x_flat = x.reshape(B, -1)  # (batch, input_len * n_features)
        
        # Encode to Koopman observables
        z = self.encoder(x_flat)  # (batch, latent_dim)
        
        # Apply Koopman operator for multi-step prediction
        # We want to predict output_len steps ahead
        # Option 1: Apply K repeatedly
        # Option 2: Learn a direct mapping (simpler, more stable)
        
        # Use power iteration of K for multi-step
        z_future = []
        z_t = z
        for t in range(self.output_len):
            z_t = self.K(z_t)  # (batch, latent_dim)
            z_future.append(z_t)
        
        # Stack and decode
        z_future = torch.stack(z_future, dim=1)  # (batch, output_len, latent_dim)
        
        # Decode each timestep
        out = self.decoder(z_future.reshape(B * self.output_len, -1))  # (batch * output_len, output_len * n_features)
        out = out.reshape(B, self.output_len, self.output_len, self.n_features)
        
        # We predicted output_len for each of output_len steps - take diagonal
        # Actually this is wrong. Let me fix: decoder should output n_features per timestep
        
        # Fix decoder to output n_features directly
        # Re-initialize decoder properly
        if not hasattr(self, '_decoder_fixed'):
            decoder_layers = []
            in_dim = self.latent_dim
            for i in range(self.n_encoder_layers):
                is_last = (i == self.n_encoder_layers - 1)
                out_dim = self.n_features if is_last else self.encoder_width
                decoder_layers.append(nn.Linear(in_dim, out_dim))
                if not is_last:
                    decoder_layers.extend([nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(self.dropout)])
                in_dim = out_dim
            self.decoder = nn.Sequential(*decoder_layers)
            self._decoder_fixed = True
        
        # Re-decode
        out = self.decoder(z_future.reshape(B * self.output_len, -1))  # (batch * output_len, n_features)
        out = out.reshape(B, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Add skip from last input value
        last_input = x[:, -1:, :]  # (batch, 1, n_features)
        out = out + last_input  # residual prediction
        
        x = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return x
