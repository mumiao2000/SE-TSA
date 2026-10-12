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
            'encoder_hidden': [128, 512, 2048],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 256)
        self.encoder_hidden = self.config.get('encoder_hidden', 512)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: maps (batch, input_len, n_features) to (batch, latent_dim)
        encoder_layers = []
        in_dim = self.input_len * self.n_features
        for i in range(self.n_layers):
            out_dim = self.encoder_hidden if i < self.n_layers - 1 else self.latent_dim
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = out_dim
        # Remove last activation/dropout
        encoder_layers = encoder_layers[:-3]
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Koopman operator: linear dynamics in latent space
        # We evolve the latent state forward in time
        self.koopman = nn.Linear(self.latent_dim, self.latent_dim)
        
        # Decoder: maps latent state back to (batch, output_len, n_features)
        decoder_layers = []
        in_dim = self.latent_dim
        for i in range(self.n_layers):
            out_dim = self.encoder_hidden if i < self.n_layers - 1 else self.output_len * self.n_features
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = out_dim
        # Remove last activation/dropout
        decoder_layers = decoder_layers[:-3]
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Flatten for encoder
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # (batch, input_len * n_features)
        
        # Encode to Koopman space
        z = self.encoder(x_flat)  # (batch, latent_dim)
        
        # Apply Koopman operator for each future step
        # We want to predict output_len steps, so we evolve the latent state
        z_future = []
        z_t = z
        for _ in range(self.output_len):
            z_t = self.koopman(z_t)  # (batch, latent_dim)
            z_future.append(z_t)
        z_future = torch.stack(z_future, dim=1)  # (batch, output_len, latent_dim)
        
        # Decode each future latent state
        # Reshape for decoder
        z_future_flat = z_future.view(batch_size * self.output_len, self.latent_dim)  # (batch*output_len, latent_dim)
        y_flat = self.decoder(z_future_flat)  # (batch*output_len, output_len * n_features)
        
        # Reshape output - we only need one step per decoded state
        y = y_flat.view(batch_size, self.output_len, self.output_len, self.n_features)  # (batch, output_len, output_len, n_features)
        # Take diagonal: each z_t predicts step t
        y = y.diagonal(dim1=1, dim2=2).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Alternative simpler approach: decode z_future directly
        # Actually let's use a simpler decoder that outputs n_features per step
        # Redefine: decode each latent to n_features
        y = self.decoder(z_future_flat)  # (batch*output_len, output_len*n_features)
        # We need to adjust decoder output dim
        # Let me fix this by making decoder output n_features directly
        
        # Re-decode properly
        # Actually, let's redo with correct dimensions
        # The issue is decoder outputs output_len * n_features, but we want n_features per step
        
        # Correct approach: latent -> n_features for each step
        # We already have z_future of shape (batch, output_len, latent_dim)
        # Let's use a per-step decoder
        
        # Simple fix: reshape and decode
        z_future_reshaped = z_future.view(-1, self.latent_dim)  # (batch*output_len, latent_dim)
        
        # Create a simple output layer
        self.to(self._get_device_from_input(x))
        
        # Use the last layer of decoder or create output head
        # Let me create a proper output projection
        if not hasattr(self, 'output_proj'):
            self.output_proj = nn.Linear(self.latent_dim, self.n_features).to(x.device)
        
        y = self.output_proj(z_future_reshaped)  # (batch*output_len, n_features)
        y = y.view(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        y = self.revin(y, mode='denorm')  # (batch, output_len, n_features)
        return y
    
    def _get_device_from_input(self, x):
        return x.device
    
    def to(self, device):
        super().to(device)
        if hasattr(self, 'output_proj'):
            self.output_proj = self.output_proj.to(device)
        return self
