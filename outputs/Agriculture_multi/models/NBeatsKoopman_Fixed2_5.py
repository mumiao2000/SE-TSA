import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NBeatsKoopman_Fixed2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'koopman_dim': [16, 64, 256],
            'dropout': [0.0, 0.3],
            'n_stacks': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.koopman_dim = self.config.get('koopman_dim', 64)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_stacks = self.config.get('n_stacks', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # N-BEATS style blocks with Koopman decomposition
        self.stacks = nn.ModuleList()
        for _ in range(self.n_stacks):
            self.stacks.append(NBeatsKoopmanBlock(
                input_len, output_len, n_features,
                self.hidden_dim, self.n_layers, self.koopman_dim, self.dropout
            ))
        
        # Fusion: combine stack outputs with proper dimension handling
        # Each stack outputs (batch, output_len, n_features), we sum them with learned weights
        self.stack_weights = nn.Parameter(torch.ones(self.n_stacks))
        
        # Residual connection for trend
        self.trend_proj = nn.Linear(input_len, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Compute trend residual
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        trend = self.trend_proj(x_perm)  # (batch, n_features, output_len)
        trend = trend.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Stack outputs
        stack_outputs = []
        for stack in self.stacks:
            stack_out = stack(x)  # (batch, output_len, n_features)
            stack_outputs.append(stack_out)
        
        # Weighted combination
        weights = F.softmax(self.stack_weights, dim=0)  # (n_stacks,)
        out = sum(w * s for w, s in zip(weights, stack_outputs))  # (batch, output_len, n_features)
        
        # Add trend residual
        out = out + trend * 0.1  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class NBeatsKoopmanBlock(nn.Module):
    def __init__(self, input_len, output_len, n_features, hidden_dim, n_layers, koopman_dim, dropout):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.koopman_dim = koopman_dim
        
        # N-BEATS style fully connected backbone
        layers = []
        in_dim = input_len
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.backbone = nn.Sequential(*layers)
        
        # Koopman encoder: project to Koopman observable space per feature
        self.koopman_encoder = nn.Linear(hidden_dim, koopman_dim)
        
        # Koopman operator: linear evolution (learned matrix per feature)
        self.koopman_K = nn.Parameter(torch.randn(n_features, koopman_dim, koopman_dim) * 0.01)
        
        # Koopman decoder: project back from Koopman space
        self.koopman_decoder = nn.Linear(koopman_dim, hidden_dim)
        
        # Forecast heads
        self.theta_f = nn.Linear(hidden_dim, output_len)  # forecast
        self.theta_b = nn.Linear(hidden_dim, input_len)   # backcast (for residual)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Process per feature
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Apply backbone per feature
        features = []
        for i in range(self.n_features):
            feat = x_perm[:, i, :]  # (batch, input_len)
            h = self.backbone(feat)  # (batch, hidden_dim)
            features.append(h)
        
        h = torch.stack(features, dim=1)  # (batch, n_features, hidden_dim)
        
        # Koopman encoding per feature
        z = self.koopman_encoder(h)  # (batch, n_features, koopman_dim)
        
        # Apply Koopman operator: linear evolution in latent space
        # z_next = K @ z for each feature
        z_evolved = torch.einsum('fcd,bfd->bfc', self.koopman_K, z)  # (batch, n_features, koopman_dim)
        
        # Decode back
        h_evolved = self.koopman_decoder(z_evolved)  # (batch, n_features, hidden_dim)
        
        # Generate forecast per feature
        forecasts = []
        for i in range(self.n_features):
            f = self.theta_f(h_evolved[:, i, :])  # (batch, output_len)
            forecasts.append(f)
        
        forecast = torch.stack(forecasts, dim=2)  # (batch, output_len, n_features)
        
        return forecast
