import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NBeatsKoopman_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [64, 256, 1024],
            'n_koopman_layers': [2, 4, 8],
            'trend_dim': [4, 8, 32],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 256)
        self.n_koopman_layers = self.config.get('n_koopman_layers', 4)
        self.trend_dim = self.config.get('trend_dim', 8)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Koopman encoder: lifts to latent space
        encoder_layers = []
        in_dim = self.input_len
        for i in range(self.n_koopman_layers):
            out_dim = self.latent_dim if i == self.n_koopman_layers - 1 else self.latent_dim // 2
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_koopman_layers - 1:
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.koopman_encoder = nn.Sequential(*encoder_layers)
        
        # Koopman operator (linear dynamics in latent space)
        self.koopman_K = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        
        # Koopman decoder: projects back
        decoder_layers = []
        in_dim = self.latent_dim
        for i in range(self.n_koopman_layers):
            out_dim = self.output_len if i == self.n_koopman_layers - 1 else self.latent_dim // 2
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.n_koopman_layers - 1:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.koopman_decoder = nn.Sequential(*decoder_layers)
        
        # N-BEATS-style trend stack
        self.trend_basis = nn.Parameter(torch.randn(self.trend_dim, self.input_len), requires_grad=False)
        for i in range(self.trend_dim):
            self.trend_basis[i] = torch.arange(self.input_len).float() ** (i + 1)
        self.trend_basis = nn.Parameter(F.normalize(self.trend_basis, dim=1), requires_grad=False)
        
        self.trend_coeff = nn.Linear(self.input_len, self.trend_dim)
        self.trend_forecast = nn.Linear(self.trend_dim, self.output_len)
        
        # Fusion layer: maps concatenated output_len*2 to output_len per feature
        self.fusion = nn.Linear(self.output_len * 2, self.output_len)
        
        # Feature-wise projection
        self.feature_proj = nn.Conv1d(self.n_features, self.n_features, 1, groups=1)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]  # (batch,)
        
        # Instance normalization
        x_norm = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Store stats for denormalization
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        x_std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        
        # Permute for per-feature processing
        x_perm = x_norm.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Koopman branch: process each feature independently
        koopman_out = []
        for i in range(self.n_features):
            feat = x_perm[:, i, :]  # (batch, input_len)
            
            # Encode to latent space
            z = self.koopman_encoder(feat)  # (batch, latent_dim)
            
            # Apply Koopman operator (one-step prediction in latent space)
            z_next = self.koopman_K(z)  # (batch, latent_dim)
            
            # Decode to output length
            k_out = self.koopman_decoder(z_next)  # (batch, output_len)
            koopman_out.append(k_out)
        
        koopman_stack = torch.stack(koopman_out, dim=1)  # (batch, n_features, output_len)
        
        # Trend branch
        trend_feats = []
        # Precompute future trend basis (device-aware, expanded to batch)
        future_basis = torch.arange(self.input_len, self.input_len + self.output_len, device=x.device).float()  # (output_len,)
        future_basis = future_basis.unsqueeze(0)  # (1, output_len)
        exponents = torch.arange(1, self.trend_dim + 1, device=x.device).float().unsqueeze(1)  # (trend_dim, 1)
        future_basis = future_basis ** exponents  # (trend_dim, output_len)
        future_basis = F.normalize(future_basis, dim=1)  # (trend_dim, output_len)
        future_basis = future_basis.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, trend_dim, output_len)
        
        for i in range(self.n_features):
            feat = x_perm[:, i, :]  # (batch, input_len)
            # Project onto polynomial basis
            coeffs = self.trend_coeff(feat)  # (batch, trend_dim)
            
            # Forecast as linear combination of future basis functions
            t_out = torch.bmm(coeffs.unsqueeze(1), future_basis)  # (batch, 1, output_len)
            trend_feats.append(t_out.squeeze(1))  # (batch, output_len)
        
        trend_stack = torch.stack(trend_feats, dim=1)  # (batch, n_features, output_len)
        
        # Fuse Koopman and trend predictions
        combined = torch.cat([koopman_stack, trend_stack], dim=2)  # (batch, n_features, output_len * 2)
        # Apply fusion directly on last dimension: (batch, n_features, output_len*2) -> (batch, n_features, output_len)
        fused = self.fusion(combined)  # (batch, n_features, output_len)
        
        # Feature interaction via 1x1 conv
        out = self.feature_proj(fused)  # (batch, n_features, output_len)
        
        # Permute back
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Denormalize: compute mean and std from input
        out = out * x_std + x_mean  # (batch, output_len, n_features)
        
        return out

