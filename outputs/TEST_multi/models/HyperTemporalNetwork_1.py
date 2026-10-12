import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperTemporalNetwork(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hyper_hidden': [64, 256, 1024],
            'hyper_depth': [1, 3, 5],
            'base_network_width': [8, 32, 128],
            'embedding_dim': [16, 64, 256]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hyper_hidden = self.config.get('hyper_hidden', 256)
        self.hyper_depth = self.config.get('hyper_depth', 3)
        self.base_width = self.config.get('base_network_width', 32)
        self.embed_dim = self.config.get('embedding_dim', 64)
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Input embedding
        self.input_embed = nn.Linear(self.n_features, self.embed_dim)
        
        # Hypernetwork: processes input sequence to generate weights
        hyper_layers = []
        for i in range(self.hyper_depth):
            in_d = self.embed_dim * self.input_len if i == 0 else self.hyper_hidden
            out_d = self.hyper_hidden
            hyper_layers.extend([
                nn.Linear(in_d, out_d),
                nn.ReLU(),
                nn.LayerNorm(out_d)
            ])
        self.hyper_encoder = nn.Sequential(*hyper_layers)
        
        # Calculate weight sizes for base network
        # Base network: input -> hidden -> output for each timestep
        # We'll use a 2-layer network per timestep
        self.w1_size = self.embed_dim * self.base_width
        self.b1_size = self.base_width
        self.w2_size = self.base_width * self.n_features
        self.b2_size = self.n_features
        self.total_weight_size = self.w1_size + self.b1_size + self.w2_size + self.b2_size
        
        # Hypernetwork output: weights for all output timesteps
        self.hyper_head = nn.Linear(self.hyper_hidden, self.output_len * self.total_weight_size)
        
        # Static bias for stability
        self.static_bias = nn.Parameter(torch.zeros(self.output_len, self.n_features))

    def _apply_base_network(self, x_embed, weights, t_idx):
        # x_embed: (batch, embed_dim)
        # weights: (batch, total_weight_size)
        batch_size = x_embed.shape[0]
        
        # Extract weights for this timestep
        start = t_idx * self.total_weight_size
        end = start + self.total_weight_size
        w = weights[:, start:end]  # (batch, total_weight_size)
        
        # Split into individual weights
        idx = 0
        w1 = w[:, idx:idx+self.w1_size].reshape(batch_size, self.embed_dim, self.base_width)  # (batch, embed_dim, base_width)
        idx += self.w1_size
        b1 = w[:, idx:idx+self.b1_size]  # (batch, base_width)
        idx += self.b1_size
        w2 = w[:, idx:idx+self.w2_size].reshape(batch_size, self.base_width, self.n_features)  # (batch, base_width, n_features)
        idx += self.w2_size
        b2 = w[:, idx:idx+self.b2_size]  # (batch, n_features)
        
        # Forward pass
        h = torch.bmm(x_embed.unsqueeze(1), w1).squeeze(1) + b1  # (batch, base_width)
        h = F.relu(h)  # (batch, base_width)
        out = torch.bmm(h.unsqueeze(1), w2).squeeze(1) + b2  # (batch, n_features)
        
        return out

    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Instance normalization
        x_norm = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, input_len, n_features)
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        x_std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        
        # Embed input
        x_embedded = self.input_embed(x_norm)  # (batch, input_len, embed_dim)
        x_flat = x_embedded.reshape(batch_size, -1)  # (batch, input_len * embed_dim)
        
        # Generate hyper weights
        hyper_features = self.hyper_encoder(x_flat)  # (batch, hyper_hidden)
        all_weights = self.hyper_head(hyper_features)  # (batch, output_len * total_weight_size)
        
        # Use last timestep embedding as input to base networks
        last_embed = x_embedded[:, -1, :]  # (batch, embed_dim)
        
        # Generate predictions for each timestep
        predictions = []
        for t in range(self.output_len):
            pred = self._apply_base_network(last_embed, all_weights, t)  # (batch, n_features)
            pred = pred + self.static_bias[t:t+1, :]  # (batch, n_features)
            predictions.append(pred)
        
        out = torch.stack(predictions, dim=1)  # (batch, output_len, n_features)
        
        # Denormalize
        out = out * x_std + x_mean  # (batch, output_len, n_features)
        return out

