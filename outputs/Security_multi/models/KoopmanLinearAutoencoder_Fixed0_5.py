import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanLinearAutoencoder_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [16, 64, 256],
            'encoder_width': [64, 256, 512],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 64)
        self.encoder_width = self.config.get('encoder_width', 256)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: input_len * n_features -> latent_dim
        encoder_layers = []
        in_dim = self.input_len * self.n_features
        for i in range(self.n_layers):
            out_dim = self.encoder_width if i < self.n_layers - 1 else self.latent_dim
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_layers - 1:
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Koopman operator: linear dynamics in latent space
        self.koopman = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        
        # Decoder: latent_dim -> n_features (per timestep)
        self.decoder = nn.Linear(self.latent_dim, self.n_features)
        
        # Optional: learned initial state correction
        self.init_correction = nn.Linear(self.latent_dim, self.latent_dim)
        
        # Residual connection: direct linear mapping
        self.residual = nn.Linear(self.input_len, self.output_len)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # (batch, input_len * n_features)
        
        # Encode to latent space
        z = self.encoder(x_flat)  # (batch, latent_dim)
        z = F.relu(z)  # (batch, latent_dim)
        
        # Apply initial correction
        z = z + self.init_correction(z)  # (batch, latent_dim)
        
        # Predict future latent states using Koopman operator
        z_future = []
        z_t = z  # (batch, latent_dim)
        for t in range(self.output_len):
            z_t = self.koopman(z_t)  # (batch, latent_dim)
            z_future.append(z_t)
        
        z_future = torch.stack(z_future, dim=1)  # (batch, output_len, latent_dim)
        
        # Decode each future latent state to n_features
        out = self.decoder(z_future.reshape(-1, self.latent_dim))  # (batch * output_len, n_features)
        out = out.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Add residual from direct linear mapping
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        res = self.residual(x_perm)  # (batch, n_features, output_len)
        res = res.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = out + 0.1 * res  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

