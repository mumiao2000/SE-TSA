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
            'koopman_modes': [8, 32, 128],
            'encoder_depth': [2, 4, 8],
            'decoder_depth': [2, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 256)
        self.koopman_modes = self.config.get('koopman_modes', 32)
        self.encoder_depth = self.config.get('encoder_depth', 4)
        self.decoder_depth = self.config.get('decoder_depth', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Advanced normalization: learnable affine per feature
        self.feature_norm = nn.LayerNorm(self.n_features)
        self.norm_gamma = nn.Parameter(torch.ones(self.n_features))
        self.norm_beta = nn.Parameter(torch.zeros(self.n_features))
        
        # Encoder: maps (batch, input_len, n_features) -> (batch, latent_dim)
        encoder_layers = []
        in_dim = self.input_len * self.n_features
        for d in range(self.encoder_depth):
            out_dim = self.latent_dim if d == self.encoder_depth - 1 else self.latent_dim * 2
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Koopman operator: block-diagonal with Fourier-inspired structure
        # Represent as eigenvalues and eigenvectors for stability
        self.koopman_eigenvalues_real = nn.Parameter(torch.randn(self.koopman_modes) * 0.01)
        self.koopman_eigenvalues_imag = nn.Parameter(torch.randn(self.koopman_modes) * 0.01)
        self.koopman_eigenvectors = nn.Parameter(torch.randn(self.latent_dim, self.koopman_modes * 2) * 0.01)
        
        # Residual Koopman for nonlinear corrections
        self.koopman_residual = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        # Decoder: maps (batch, latent_dim) -> (batch, output_len, n_features)
        decoder_layers = []
        in_dim = self.latent_dim
        for d in range(self.decoder_depth):
            out_dim = self.output_len * self.n_features if d == self.decoder_depth - 1 else self.latent_dim * 2
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if d < self.decoder_depth - 1 else nn.Identity(),
                nn.GELU() if d < self.decoder_depth - 1 else nn.Identity(),
                nn.Dropout(self.dropout) if d < self.decoder_depth - 1 else nn.Identity()
            ])
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Temporal interpolation for multi-step forecasting
        self.temporal_interp = nn.Parameter(torch.randn(self.output_len, self.input_len) * 0.01)
        
    def apply_koopman(self, z, n_steps):
        # z: (batch, latent_dim)
        # Compute Koopman operator K = V * diag(lambda) * V^+
        eigenvalues = torch.complex(
            self.koopman_eigenvalues_real,
            self.koopman_eigenvalues_imag
        )  # (koopman_modes,)
        
        # Construct block-diagonal rotation-scaling matrices
        # For stability, ensure |eigenvalue| < 1
        eigenvalues = eigenvalues / (torch.abs(eigenvalues) + 1.0)  # Normalize to unit circle
        
        # Power iteration for n_steps
        eigenvalues_power = eigenvalues ** n_steps  # (koopman_modes,)
        
        # Reconstruct K^n
        # V: (latent_dim, 2*koopman_modes) [real, imag parts]
        V = self.koopman_eigenvectors  # (latent_dim, 2*koopman_modes)
        
        # Project to eigenbasis, apply powers, project back
        z_proj = torch.matmul(z, V)  # (batch, 2*koopman_modes)
        
        # Apply complex multiplication via real representation
        z_real = z_proj[:, :self.koopman_modes]  # (batch, koopman_modes)
        z_imag = z_proj[:, self.koopman_modes:]  # (batch, koopman_modes)
        
        ev_real = eigenvalues_power.real  # (koopman_modes,)
        ev_imag = eigenvalues_power.imag  # (koopman_modes,)
        
        z_new_real = z_real * ev_real.unsqueeze(0) - z_imag * ev_imag.unsqueeze(0)  # (batch, koopman_modes)
        z_new_imag = z_real * ev_imag.unsqueeze(0) + z_imag * ev_real.unsqueeze(0)  # (batch, koopman_modes)
        
        z_new = torch.cat([z_new_real, z_new_imag], dim=-1)  # (batch, 2*koopman_modes)
        z_out = torch.matmul(z_new, V.t())  # (batch, latent_dim)
        
        # Add residual correction
        z_out = z_out + self.koopman_residual(z_out)  # (batch, latent_dim)
        
        return z_out
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch = x.shape[0]
        
        # Advanced normalization
        x = self.feature_norm(x)  # (batch, input_len, n_features)
        x = x * self.norm_gamma + self.norm_beta  # (batch, input_len, n_features)
        
        # Flatten and encode
        x_flat = x.view(batch, -1)  # (batch, input_len * n_features)
        z = self.encoder(x_flat)  # (batch, latent_dim)
        
        # Apply Koopman operator for forecasting
        z_future = self.apply_koopman(z, self.output_len)  # (batch, latent_dim)
        
        # Decode
        out_flat = self.decoder(z_future)  # (batch, output_len * n_features)
        out = out_flat.view(batch, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Temporal refinement with learned interpolation
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        interp = torch.matmul(x_perm, self.temporal_interp.t())  # (batch, n_features, output_len)
        interp = interp.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Residual connection
        out = out + 0.1 * interp  # (batch, output_len, n_features)
        
        # Denormalize
        out = (out - self.norm_beta) / (self.norm_gamma + 1e-6)  # (batch, output_len, n_features)
        
        return out

