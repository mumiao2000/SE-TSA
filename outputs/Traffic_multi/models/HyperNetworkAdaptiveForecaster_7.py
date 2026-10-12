import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperNetworkAdaptiveForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hyper_dim': [32, 128, 512],
            'hyper_layers': [1, 2, 4],
            'main_hidden': [16, 64, 256],
            'n_regimes': [2, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.hyper_dim = cfg.get('hyper_dim', 128)
        self.hyper_layers = cfg.get('hyper_layers', 2)
        self.main_hidden = cfg.get('main_hidden', 64)
        self.n_regimes = cfg.get('n_regimes', 4)
        self.dropout = cfg.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Regime embedding: learnable prototypes
        self.regime_embeds = nn.Parameter(torch.randn(self.n_regimes, self.hyper_dim))
        
        # Hypernetwork: generates main network weights
        # First, encode input statistics
        self.stat_encoder = nn.Sequential(
            nn.Linear(n_features * 2, self.hyper_dim),  # mean and std per feature
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Hypernetwork MLP
        hyper_layers = []
        for i in range(self.hyper_layers):
            in_dim = self.hyper_dim if i == 0 else self.hyper_dim
            out_dim = self.hyper_dim
            hyper_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
        self.hyper_net = nn.Sequential(*hyper_layers)
        
        # Generate weights for main network
        # Main network: input_len -> main_hidden -> output_len, per feature
        # Weight shapes: (n_features, main_hidden, input_len) and (n_features, output_len, main_hidden)
        # Bias shapes: (n_features, main_hidden) and (n_features, output_len)
        
        self.weight_gen1 = nn.Linear(self.hyper_dim, n_features * self.main_hidden * input_len)
        self.bias_gen1 = nn.Linear(self.hyper_dim, n_features * self.main_hidden)
        self.weight_gen2 = nn.Linear(self.hyper_dim, n_features * output_len * self.main_hidden)
        self.bias_gen2 = nn.Linear(self.hyper_dim, n_features * output_len)
        
        # Regime selector
        self.regime_selector = nn.Sequential(
            nn.Linear(self.hyper_dim, self.n_regimes),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Compute input statistics for regime detection
        x_mean = x.mean(dim=1)  # (batch, n_features)
        x_std = x.std(dim=1, unbiased=False) + 1e-5  # (batch, n_features)
        stats = torch.cat([x_mean, x_std], dim=-1)  # (batch, n_features*2)
        
        # Encode statistics
        z = self.stat_encoder(stats)  # (batch, hyper_dim)
        
        # Select regime
        regime_probs = self.regime_selector(z)  # (batch, n_regimes)
        
        # Get regime-conditioned hypernetwork input
        regime_cond = torch.matmul(regime_probs, self.regime_embeds)  # (batch, hyper_dim)
        hyper_input = z + regime_cond  # (batch, hyper_dim)
        
        # Generate hypernetwork output
        h = self.hyper_net(hyper_input)  # (batch, hyper_dim)
        
        # Generate main network weights
        w1 = self.weight_gen1(h).reshape(batch_size, self.n_features, self.main_hidden, self.input_len)
        b1 = self.bias_gen1(h).reshape(batch_size, self.n_features, self.main_hidden)
        w2 = self.weight_gen2(h).reshape(batch_size, self.n_features, self.output_len, self.main_hidden)
        b2 = self.bias_gen2(h).reshape(batch_size, self.n_features, self.output_len)
        
        # Apply main network per feature (channel-independent)
        # x: (batch, input_len, n_features) -> (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # First layer: (batch, n_features, input_len) @ (batch, n_features, input_len, main_hidden)^T
        # Actually: for each feature f: x[:, f, :] (batch, input_len) @ w1[:, f, :, :].T (input_len, main_hidden)
        h = torch.einsum('bfi,bfhi->bfh', x, w1) + b1  # (batch, n_features, main_hidden)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Second layer
        out = torch.einsum('bfh,bfoh->bfo', h, w2) + b2  # (batch, n_features, output_len)
        
        # Reshape to output format
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

