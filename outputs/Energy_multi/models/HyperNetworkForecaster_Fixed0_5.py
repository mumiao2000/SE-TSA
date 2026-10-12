import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperNetworkForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hyper_dim': [32, 128, 512],
            'target_hidden': [16, 64, 256],
            'n_hyper_layers': [1, 2, 4]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hyper_dim = self.config.get('hyper_dim', 128)
        self.target_hidden = self.config.get('target_hidden', 64)
        self.n_hyper_layers = self.config.get('n_hyper_layers', 2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input statistics encoder for hypernetwork conditioning
        self.stat_encoder = nn.Sequential(
            nn.Linear(n_features * 4, self.hyper_dim),  # mean, std, min, max per feature
            nn.GELU(),
            nn.LayerNorm(self.hyper_dim)
        )
        
        # Hypernetwork: generates target network weights
        # Target network: input_len -> target_hidden -> output_len (per feature)
        # Weights: W1: (input_len, target_hidden), b1: (target_hidden,)
        #          W2: (target_hidden, output_len), b2: (output_len,)
        
        target_params = input_len * self.target_hidden + self.target_hidden + \
                       self.target_hidden * output_len + output_len
        
        self.hyper_layers = nn.ModuleList()
        in_dim = self.hyper_dim
        for i in range(self.n_hyper_layers):
            out_dim = target_params if i == self.n_hyper_layers - 1 else self.hyper_dim
            self.hyper_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_hyper_layers - 1:
                self.hyper_layers.append(nn.GELU())
            in_dim = out_dim
        
        # Also generate feature-interaction mixing weights
        self.mixer_hyper = nn.Linear(self.hyper_dim, n_features * n_features)
        
        # Temporal embedding for position awareness
        self.temporal_embed = nn.Parameter(torch.randn(1, input_len, n_features) * 0.02)
        
    def _extract_stats(self, x):
        # x: (batch, input_len, n_features)
        mean = x.mean(dim=1)  # (batch, n_features)
        std = x.std(dim=1, unbiased=False) + 1e-5  # (batch, n_features)
        min_val = x.min(dim=1)[0]  # (batch, n_features)
        max_val = x.max(dim=1)[0]  # (batch, n_features)
        return torch.cat([mean, std, min_val, max_val], dim=-1)  # (batch, 4*n_features)
    
    def _generate_weights(self, stats):
        # stats: (batch, 4*n_features)
        z = self.stat_encoder(stats)  # (batch, hyper_dim)
        
        # Generate target network parameters
        params = z
        for layer in self.hyper_layers:
            params = layer(params)
        
        # Generate mixer weights
        mixer_w = self.mixer_hyper(z)  # (batch, n_features * n_features)
        
        return params, mixer_w, z
    
    def _apply_target_network(self, x, weights, batch_size):
        # x: (batch, n_features, input_len)
        # weights: (batch, target_params)
        
        # Parse weights
        idx = 0
        w1_size = self.input_len * self.target_hidden
        b1_size = self.target_hidden
        w2_size = self.target_hidden * self.output_len
        b2_size = self.output_len
        
        # Reshape weights for batch operations
        W1 = weights[:, idx:idx+w1_size].view(batch_size, self.input_len, self.target_hidden)  # (batch, input_len, target_hidden)
        idx += w1_size
        b1 = weights[:, idx:idx+b1_size].view(batch_size, 1, self.target_hidden)  # (batch, 1, target_hidden)
        idx += b1_size
        W2 = weights[:, idx:idx+w2_size].view(batch_size, self.target_hidden, self.output_len)  # (batch, target_hidden, output_len)
        idx += w2_size
        b2 = weights[:, idx:idx+b2_size].view(batch_size, 1, self.output_len)  # (batch, 1, output_len)
        
        # Apply target network per feature using einsum for clarity and correctness
        # x: (batch, n_features, input_len)
        # For each feature, apply the same MLP (shared weights across features, but generated per batch)
        
        # First layer: (batch, n_features, input_len) @ (batch, input_len, target_hidden) -> (batch, n_features, target_hidden)
        h = torch.einsum('bfi,bih->bfh', x, W1) + b1  # (batch, n_features, target_hidden)
        h = F.gelu(h)  # (batch, n_features, target_hidden)
        
        # Second layer: (batch, n_features, target_hidden) @ (batch, target_hidden, output_len) -> (batch, n_features, output_len)
        out = torch.einsum('bfh,bho->bfo', h, W2) + b2  # (batch, n_features, output_len)
        
        # Permute to (batch, output_len, n_features)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        return out
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Extract statistics for conditioning
        stats = self._extract_stats(x)  # (batch, 4*n_features)
        
        # Generate adaptive weights
        weights, mixer_w, z = self._generate_weights(stats)  # weights: (batch, target_params), mixer_w: (batch, n_features^2)
        
        # Add temporal embedding
        x = x + self.temporal_embed  # (batch, input_len, n_features)
        
        # Apply target network with generated weights
        pred = self._apply_target_network(x.permute(0, 2, 1), weights, batch_size)  # (batch, output_len, n_features)
        
        # Apply learned feature mixing
        mixer_w = mixer_w.view(batch_size, self.n_features, self.n_features)  # (batch, n_features, n_features)
        mixer_w = F.softmax(mixer_w, dim=-1)  # (batch, n_features, n_features)
        pred = torch.bmm(pred, mixer_w.transpose(1, 2))  # (batch, output_len, n_features)
        
        x = self.revin(pred, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)
