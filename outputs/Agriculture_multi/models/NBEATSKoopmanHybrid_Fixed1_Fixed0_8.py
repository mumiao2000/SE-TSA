import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NBEATSKoopmanHybrid_Fixed1_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_stacks': [2, 4, 8],
            'n_blocks': [1, 2, 4],
            'koopman_dim': [16, 64, 256],
            'dropout': [0.0, 0.3],
            'gate_dim': [32, 128, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.hidden_dim = cfg.get('hidden_dim', 256)
        self.n_stacks = cfg.get('n_stacks', 4)
        self.n_blocks = cfg.get('n_blocks', 2)
        self.koopman_dim = cfg.get('koopman_dim', 64)
        self.dropout = cfg.get('dropout', 0.1)
        self.gate_dim = cfg.get('gate_dim', 128)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # N-BEATS style decomposition stacks
        self.trend_stacks = nn.ModuleList()
        self.seasonal_stacks = nn.ModuleList()
        self.koopman_stacks = nn.ModuleList()
        
        for _ in range(self.n_stacks):
            # Trend stack
            trend_blocks = nn.ModuleList()
            for _ in range(self.n_blocks):
                trend_blocks.append(nn.Sequential(
                    nn.Linear(self.input_len * self.n_features, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.output_len * self.n_features)
                ))
            self.trend_stacks.append(trend_blocks)
            
            # Seasonal stack
            seasonal_blocks = nn.ModuleList()
            for _ in range(self.n_blocks):
                seasonal_blocks.append(nn.Sequential(
                    nn.Linear(self.input_len * self.n_features, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, self.output_len * self.n_features)
                ))
            self.seasonal_stacks.append(seasonal_blocks)
            
            # Koopman stack with proper nn.Module
            koopman_blocks = nn.ModuleList()
            for _ in range(self.n_blocks):
                koopman_blocks.append(KoopmanOperatorModule(
                    input_len=self.input_len,
                    output_len=self.output_len,
                    n_features=self.n_features,
                    hidden_dim=self.hidden_dim,
                    koopman_dim=self.koopman_dim,
                    dropout=self.dropout
                ))
            self.koopman_stacks.append(koopman_blocks)
        
        # Channel-adaptive gating for fusion
        self.gate_encoder = nn.Sequential(
            nn.Linear(self.n_features * 3, self.gate_dim),  # 3 components: trend, seasonal, koopman
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.gate_dim, self.n_features * 3),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.n_features * 3, self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # (batch, input_len * n_features)
        
        all_trend = []
        all_seasonal = []
        all_koopman = []
        
        for stack_idx in range(self.n_stacks):
            # Trend component: each block takes original x_flat, sum their outputs
            trend_out = torch.zeros(batch_size, self.output_len * self.n_features, device=x.device)  # (batch, output_len * n_features)
            for block in self.trend_stacks[stack_idx]:
                trend_out = trend_out + block(x_flat)  # (batch, output_len * n_features)
            all_trend.append(trend_out.reshape(batch_size, self.output_len, self.n_features))  # (batch, output_len, n_features)
            
            # Seasonal component: each block takes original x_flat, sum their outputs
            seasonal_out = torch.zeros(batch_size, self.output_len * self.n_features, device=x.device)  # (batch, output_len * n_features)
            for block in self.seasonal_stacks[stack_idx]:
                seasonal_out = seasonal_out + block(x_flat)  # (batch, output_len * n_features)
            all_seasonal.append(seasonal_out.reshape(batch_size, self.output_len, self.n_features))  # (batch, output_len, n_features)
            
            # Koopman component: each block takes original x, sum their outputs
            koopman_out = torch.zeros(batch_size, self.output_len, self.n_features, device=x.device)  # (batch, output_len, n_features)
            for block in self.koopman_stacks[stack_idx]:
                koopman_out = koopman_out + block(x)  # (batch, output_len, n_features)
            all_koopman.append(koopman_out)  # (batch, output_len, n_features)
        
        # Aggregate across stacks
        trend_agg = torch.stack(all_trend, dim=0).mean(dim=0)  # (batch, output_len, n_features)
        seasonal_agg = torch.stack(all_seasonal, dim=0).mean(dim=0)  # (batch, output_len, n_features)
        koopman_agg = torch.stack(all_koopman, dim=0).mean(dim=0)  # (batch, output_len, n_features)
        
        # Channel-adaptive gating
        combined = torch.cat([trend_agg, seasonal_agg, koopman_agg], dim=-1)  # (batch, output_len, n_features * 3)
        
        # Per-timestep gating
        gate_input = combined  # (batch, output_len, n_features * 3)
        gates = self.gate_encoder(gate_input)  # (batch, output_len, n_features * 3)
        
        gated = combined * gates  # (batch, output_len, n_features * 3)
        
        # Project to output features
        out = self.output_proj(gated)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class KoopmanOperatorModule(nn.Module):
    """Proper nn.Module containing Koopman operator parameters."""
    def __init__(self, input_len, output_len, n_features, hidden_dim, koopman_dim, dropout):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.koopman_dim = koopman_dim
        
        # Encoder to Koopman space
        self.encoder = nn.Sequential(
            nn.Linear(input_len * n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, koopman_dim)
        )
        
        # Koopman operator (learned linear dynamics)
        self.K = nn.Parameter(torch.randn(koopman_dim, koopman_dim) * 0.01)
        
        # Decoder from Koopman space
        self.decoder = nn.Sequential(
            nn.Linear(koopman_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len * n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)  # (batch, input_len * n_features)
        
        # Encode to Koopman space
        z = self.encoder(x_flat)  # (batch, koopman_dim)
        
        # Apply Koopman operator (linear dynamics evolution)
        z_next = z @ self.K  # (batch, koopman_dim)
        
        # Decode to output space
        out = self.decoder(z_next)  # (batch, output_len * n_features)
        out = out.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        return out

