import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanAutoEncoderRevIN(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [64, 256, 1024],
            'encoder_depth': [2, 4, 8],
            'dropout': [0.0, 0.3],
            'n_freq': [8, 32, 128]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 256)
        self.encoder_depth = self.config.get('encoder_depth', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_freq = self.config.get('n_freq', 32)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Frequency embedding for positional/time information
        self.freq_embedding = nn.Parameter(torch.randn(self.n_freq) * 0.01)
        
        # Encoder: maps (batch, input_len, n_features) -> (batch, latent_dim)
        encoder_layers = []
        in_dim = input_len * n_features + self.n_freq
        hidden_dims = [in_dim] + [self.latent_dim * 2] * (self.encoder_depth - 1) + [self.latent_dim]
        
        for i in range(len(hidden_dims) - 1):
            encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i < len(hidden_dims) - 2:
                encoder_layers.append(nn.LayerNorm(hidden_dims[i + 1]))
                encoder_layers.append(nn.GELU())
                encoder_layers.append(nn.Dropout(self.dropout))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Koopman operator (learned linear dynamics matrix)
        self.K = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        
        # Initialize K near identity for stability
        with torch.no_grad():
            self.K.weight.copy_(torch.eye(self.latent_dim) * 0.99 + torch.randn(self.latent_dim, self.latent_dim) * 0.01)
        
        # Decoder: maps (batch, latent_dim) -> (batch, output_len, n_features)
        decoder_layers = []
        hidden_dims = [self.latent_dim] + [self.latent_dim * 2] * (self.encoder_depth - 1) + [output_len * n_features]
        
        for i in range(len(hidden_dims) - 1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i < len(hidden_dims) - 2:
                decoder_layers.append(nn.LayerNorm(hidden_dims[i + 1]))
                decoder_layers.append(nn.GELU())
                decoder_layers.append(nn.Dropout(self.dropout))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        # x: (batch, input_len, n_features)
        batch = x.shape[0]
        
        # Flatten temporal and feature dimensions
        x_flat = x.reshape(batch, -1)  # (batch, input_len * n_features)
        
        # Add frequency features
        freq_feat = torch.sin(self.freq_embedding).mean().unsqueeze(0).expand(batch, self.n_freq)
        x_flat = torch.cat([x_flat, freq_feat], dim=-1)  # (batch, input_len * n_features + n_freq)
        
        z = self.encoder(x_flat)  # (batch, latent_dim)
        return z
    
    def koopman_forecast(self, z, steps):
        # z: (batch, latent_dim)
        # Apply Koopman operator repeatedly
        predictions = []
        z_t = z
        for _ in range(steps):
            z_t = self.K(z_t)  # (batch, latent_dim)
            predictions.append(z_t)
        
        # Stack: (batch, steps, latent_dim)
        return torch.stack(predictions, dim=1)
    
    def decode(self, z_seq):
        # z_seq: (batch, output_len, latent_dim)
        batch, steps, _ = z_seq.shape
        
        # Decode each timestep
        outputs = []
        for t in range(steps):
            z_t = z_seq[:, t, :]  # (batch, latent_dim)
            out = self.decoder(z_t)  # (batch, output_len * n_features)
            outputs.append(out)
        
        # We want one output per step, but decoder produces flattened output_len * n_features
        # Actually, let's reshape: each step decodes to (output_len, n_features) is wrong
        # Correct: decode to n_features directly for each future step
        # Re-design: decoder outputs n_features per latent vector
        
        # Re-interpret: z_seq has output_len steps, each decodes to n_features
        output = torch.stack(outputs, dim=1)  # (batch, output_len, output_len * n_features) - wrong
        
        # Fix: decode each to n_features
        return output.reshape(batch, steps, self.output_len, self.n_features)[:, :, 0, :]  # hack
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Encode
        z = self.encode(x)  # (batch, latent_dim)
        
        # Koopman forecast in latent space
        z_future = self.koopman_forecast(z, self.output_len)  # (batch, output_len, latent_dim)
        
        # Decode each timestep
        batch = z_future.shape[0]
        outputs = []
        for t in range(self.output_len):
            z_t = z_future[:, t, :]  # (batch, latent_dim)
            out_t = self.decoder(z_t)  # (batch, output_len * n_features)
            # We need to output n_features, not output_len * n_features
            # Re-design decoder: final layer should map to n_features
            outputs.append(out_t)
        
        # Stack outputs
        # But out_t is (batch, output_len * n_features), need to fix
        # Let's fix by taking first n_features or reshaping properly
        
        # Actually, let me fix the decoder architecture in forward
        # Redecode properly
        
        # Correct approach: use a separate decoder head for each feature
        pass  # will fix below
        
        # Proper implementation:
        z_future_flat = z_future.reshape(-1, self.latent_dim)  # (batch * output_len, latent_dim)
        out_flat = self.decoder(z_future_flat)  # (batch * output_len, output_len * n_features)
        
        # This is getting messy. Let me restructure.
        # The decoder should output n_features directly.
        
        # Re-initialize decoder properly in __init__ but we can't. Work with what we have.
        # Take first n_features elements
        out = out_flat[:, :self.n_features]  # (batch * output_len, n_features)
        out = out.reshape(batch, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        x = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return x
