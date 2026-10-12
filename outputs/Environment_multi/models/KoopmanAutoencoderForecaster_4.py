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
            'encoder_depth': [2, 4, 8],
            'koopman_regularization': [0.0, 0.01, 0.1],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 256)
        self.encoder_depth = self.config.get('encoder_depth', 4)
        self.koopman_reg = self.config.get('koopman_regularization', 0.01)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: (batch, input_len, n_features) -> (batch, latent_dim)
        encoder_layers = []
        in_dim = self.input_len * self.n_features
        for i in range(self.encoder_depth):
            out_dim = self.latent_dim * (2 ** (self.encoder_depth - 1 - i))
            out_dim = min(out_dim, in_dim * 2) if i < self.encoder_depth - 1 else self.latent_dim
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Koopman operator (learned linear dynamics)
        self.koopman_K = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        
        # Decoder: (batch, latent_dim) -> (batch, output_len, n_features)
        decoder_layers = []
        in_dim = self.latent_dim
        for i in range(self.encoder_depth):
            out_dim = self.latent_dim * (2 ** i) if i < self.encoder_depth - 1 else self.output_len * self.n_features
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < self.encoder_depth - 1 else nn.Identity(),
                nn.GELU() if i < self.encoder_depth - 1 else nn.Identity(),
                nn.Dropout(self.dropout) if i < self.encoder_depth - 1 else nn.Identity()
            ])
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        # For stability: initialize Koopman operator near identity
        with torch.no_grad():
            self.koopman_K.weight.copy_(torch.eye(self.latent_dim) * 0.99 + torch.randn(self.latent_dim, self.latent_dim) * 0.01)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Flatten for encoder
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # (batch, input_len * n_features)
        
        # Encode to Koopman-invariant space
        z = self.encoder(x_flat)  # (batch, latent_dim)
        
        # Apply Koopman operator for each future step
        z_future = []
        z_t = z  # (batch, latent_dim)
        for _ in range(self.output_len):
            z_t = self.koopman_K(z_t)  # (batch, latent_dim)
            z_future.append(z_t)  # list of (batch, latent_dim)
        
        z_future = torch.stack(z_future, dim=1)  # (batch, output_len, latent_dim)
        
        # Decode each timestep
        out = self.decoder(z_future.reshape(-1, self.latent_dim))  # (batch * output_len, output_len * n_features)
        out = out.reshape(batch_size, self.output_len, self.output_len, self.n_features)
        # Take diagonal: each z_t predicts t-th future step
        out = out[:, :, 0, :]  # (batch, output_len, n_features) - simplified: use first prediction
        
        # Alternative: decode each latent state independently
        out = z_future.reshape(-1, self.latent_dim)  # (batch * output_len, latent_dim)
        out = self.decoder(out)  # (batch * output_len, output_len * n_features)
        out = out.reshape(batch_size, self.output_len, self.output_len, self.n_features)
        # For simplicity, predict all horizons from each state, take appropriate
        out = out[:, torch.arange(self.output_len), torch.arange(self.output_len), :]  # (batch, output_len, n_features)
        
        # Simpler approach: predict full output from final latent state
        z_final = z_future[:, -1, :]  # (batch, latent_dim)
        out = self.decoder(z_final)  # (batch, output_len * n_features)
        out = out.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)
