import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanAutoencoder(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'encoder_dim': [64, 256, 1024],
            'latent_dim': [16, 64, 256],
            'n_encoder_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.encoder_dim = self.config.get('encoder_dim', 256)
        self.latent_dim = self.config.get('latent_dim', 64)
        self.n_encoder_layers = self.config.get('n_encoder_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: maps (batch, input_len, n_features) -> (batch, latent_dim)
        encoder_layers = []
        in_dim = self.input_len * self.n_features
        for i in range(self.n_encoder_layers):
            out_dim = self.encoder_dim if i < self.n_encoder_layers - 1 else self.latent_dim
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < self.n_encoder_layers - 1 else nn.Identity(),
                nn.GELU() if i < self.n_encoder_layers - 1 else nn.Identity(),
                nn.Dropout(self.dropout) if i < self.n_encoder_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Koopman operator: linear dynamics matrix
        self.koopman_matrix = nn.Parameter(torch.randn(self.latent_dim, self.latent_dim) * 0.01)
        
        # Decoder: maps (batch, latent_dim) -> (batch, output_len, n_features)
        decoder_layers = []
        in_dim = self.latent_dim
        for i in range(self.n_encoder_layers):
            out_dim = self.encoder_dim if i < self.n_encoder_layers - 1 else self.output_len * self.n_features
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < self.n_encoder_layers - 1 else nn.Identity(),
                nn.GELU() if i < self.n_encoder_layers - 1 else nn.Identity(),
                nn.Dropout(self.dropout) if i < self.n_encoder_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Spectral normalization for Koopman stability
        self.register_buffer('koopman_eigvals', torch.zeros(self.latent_dim))
        
    def _stable_koopman_power(self, z, n_steps):
        # z: (batch, latent_dim)
        # Compute K^n * z with spectral normalization
        # Use eigendecomposition for stability
        K = self.koopman_matrix  # (latent_dim, latent_dim)
        
        # Ensure stability: normalize eigenvalues to unit circle
        with torch.no_grad():
            try:
                eigvals, eigvecs = torch.linalg.eig(K)
                # Project eigenvalues to unit circle
                eigvals_norm = eigvals / (torch.abs(eigvals) + 1e-8)
                K_stable = eigvecs @ torch.diag_embed(eigvals_norm) @ torch.linalg.inv(eigvecs)
                K_stable = K_stable.real
            except:
                K_stable = K
            # Update parameter with stable version (soft constraint via gradient)
            self.koopman_eigvals.copy_(torch.abs(eigvals).real)
        
        # Use power iteration for forward pass
        outputs = []
        z_t = z  # (batch, latent_dim)
        for _ in range(n_steps):
            z_t = F.linear(z_t, K_stable.T)  # (batch, latent_dim)
            outputs.append(z_t)
        
        return torch.stack(outputs, dim=1)  # (batch, n_steps, latent_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Flatten for encoder
        x_flat = x.reshape(batch_size, -1)  # (batch, input_len * n_features)
        
        # Encode to Koopman space
        z = self.encoder(x_flat)  # (batch, latent_dim)
        
        # Evolve with Koopman operator for output_len steps
        z_future = self._stable_koopman_power(z, self.output_len)  # (batch, output_len, latent_dim)
        
        # Decode each timestep
        z_future_flat = z_future.reshape(batch_size * self.output_len, self.latent_dim)  # (batch*output_len, latent_dim)
        y_flat = self.decoder(z_future_flat)  # (batch*output_len, output_len * n_features)
        
        # Reshape output - we need output_len timesteps, but decoder gives flat
        # Actually, decode to single timestep per step
        y = y_flat.reshape(batch_size, self.output_len, self.output_len, self.n_features)
        # Take diagonal: each step i predicts step i
        y = torch.diagonal(y, dim1=1, dim2=2).permute(2, 0, 1)  # (output_len, batch, n_features)
        y = y.permute(1, 0, 2)  # (batch, output_len, n_features)
        
        # Simpler: re-decode properly
        y_steps = []
        for t in range(self.output_len):
            z_t = z_future[:, t, :]  # (batch, latent_dim)
            y_t = self.decoder(z_t)  # (batch, output_len * n_features)
            y_t = y_t[:, :self.n_features]  # Take first n_features as prediction for step t
            y_steps.append(y_t)
        y = torch.stack(y_steps, dim=1)  # (batch, output_len, n_features)
        
        # Better approach: decoder outputs single timestep
        # Re-initialize decoder for single step output
        self._rebuild_decoder()
        
        y_steps = []
        for t in range(self.output_len):
            z_t = z_future[:, t, :]  # (batch, latent_dim)
            y_t = self.decoder_single(z_t)  # (batch, n_features)
            y_steps.append(y_t)
        y = torch.stack(y_steps, dim=1)  # (batch, output_len, n_features)
        
        y = self.revin(y, mode='denorm')  # (batch, output_len, n_features)
        return y
    
    def _rebuild_decoder(self):
        # Build single-step decoder if not exists
        if not hasattr(self, 'decoder_single'):
            layers = []
            in_dim = self.latent_dim
            for i in range(self.n_encoder_layers):
                out_dim = self.encoder_dim if i < self.n_encoder_layers - 1 else self.n_features
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim) if i < self.n_encoder_layers - 1 else nn.Identity(),
                    nn.GELU() if i < self.n_encoder_layers - 1 else nn.Identity(),
                ])
                in_dim = out_dim
            self.decoder_single = nn.Sequential(*layers).to(self.koopman_matrix.device)
