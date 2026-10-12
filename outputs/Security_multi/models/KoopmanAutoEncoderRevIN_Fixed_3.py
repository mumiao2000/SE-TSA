import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanAutoEncoderRevIN_Fixed(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [64, 256, 1024],
            'hidden_dim': [128, 512, 2048],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 256)
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Build encoder: (batch, input_len, n_features) -> (batch, latent_dim)
        self.encoder_input = nn.Linear(input_len * n_features, self.hidden_dim)
        self.encoder_norms = nn.ModuleList()
        self.encoder_layers = nn.ModuleList()
        
        for i in range(self.n_layers):
            self.encoder_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim if i < self.n_layers - 1 else self.latent_dim))
            if i < self.n_layers - 1:
                self.encoder_norms.append(nn.LayerNorm(self.hidden_dim))
        
        self.encoder_act = nn.GELU()
        self.encoder_drop = nn.Dropout(self.dropout)
        
        # Koopman operator
        self.K = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        with torch.no_grad():
            self.K.weight.copy_(torch.eye(self.latent_dim) * 0.99 + torch.randn(self.latent_dim, self.latent_dim) * 0.01)
        
        # Build decoder: (batch, latent_dim) -> (batch, n_features)
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_norms = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        for i in range(self.n_layers):
            self.decoder_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim if i < self.n_layers - 1 else n_features))
            if i < self.n_layers - 1:
                self.decoder_norms.append(nn.LayerNorm(self.hidden_dim))
        
        self.decoder_act = nn.GELU()
        self.decoder_drop = nn.Dropout(self.dropout)
    
    def encode(self, x):
        # x: (batch, input_len, n_features)
        batch = x.shape[0]
        x = x.reshape(batch, -1)  # (batch, input_len * n_features)
        x = self.encoder_input(x)  # (batch, hidden_dim)
        
        for i, layer in enumerate(self.encoder_layers):
            if i < len(self.encoder_layers) - 1:
                x = self.encoder_norms[i](x)
                x = self.encoder_act(x)
                x = self.encoder_drop(x)
            x = layer(x)  # (batch, hidden_dim) or (batch, latent_dim)
        
        return x  # (batch, latent_dim)
    
    def decode(self, z):
        # z: (batch, latent_dim)
        x = self.decoder_input(z)  # (batch, hidden_dim)
        
        for i, layer in enumerate(self.decoder_layers):
            if i < len(self.decoder_layers) - 1:
                x = self.decoder_norms[i](x)
                x = self.decoder_act(x)
                x = self.decoder_drop(x)
            x = layer(x)  # (batch, hidden_dim) or (batch, n_features)
        
        return x  # (batch, n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Encode
        z = self.encode(x)  # (batch, latent_dim)
        
        # Koopman evolution and decode
        outputs = []
        z_t = z
        for _ in range(self.output_len):
            z_t = self.K(z_t)  # (batch, latent_dim)
            out_t = self.decode(z_t)  # (batch, n_features)
            outputs.append(out_t)
        
        out = torch.stack(outputs, dim=1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
