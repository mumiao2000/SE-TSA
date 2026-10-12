import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DLinearResidualChannelGated(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'dropout': [0.0, 0.3],
            'gate_dim': [16, 64, 256],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.gate_dim = self.config.get('gate_dim', 64)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Moving average kernel for trend extraction (learnable)
        self.trend_kernel_size = max(3, input_len // 4)
        if self.trend_kernel_size % 2 == 0:
            self.trend_kernel_size += 1
        
        # Learnable decomposition: each feature has its own trend weight
        self.trend_weight = nn.Parameter(torch.ones(self.n_features, 1))  # (n_features, 1)
        self.season_weight = nn.Parameter(torch.ones(self.n_features, 1))  # (n_features, 1)
        
        # Trend projector: moving average + linear
        self.trend_avg = nn.AvgPool1d(kernel_size=self.trend_kernel_size, stride=1, padding=self.trend_kernel_size // 2)
        self.trend_fc = nn.Linear(input_len, output_len)  # (input_len -> output_len)
        
        # Seasonal projector: direct linear mapping
        self.season_fc = nn.Linear(input_len, output_len)  # (input_len -> output_len)
        
        # Channel-adaptive feature-wise MLPs for trend and season
        self.trend_feature_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),  # (1 -> hidden_dim)
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1),  # (hidden_dim -> 1)
        )
        
        self.season_feature_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),  # (1 -> hidden_dim)
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 1),  # (hidden_dim -> 1)
        )
        
        # Cross-channel gating mechanism
        # Project all features to a shared gate space
        self.gate_proj = nn.Linear(self.n_features, self.gate_dim)  # (n_features -> gate_dim)
        self.gate_act = nn.Sequential(
            nn.Linear(self.gate_dim, self.gate_dim),  # (gate_dim -> gate_dim)
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
        )
        self.gate_out = nn.Linear(self.gate_dim, self.n_features)  # (gate_dim -> n_features)
        
        # Residual connection weights
        self.residual_weight = nn.Parameter(torch.ones(1))
        
        # Final fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(self.n_features * 2, self.hidden_dim),  # (2*n_features -> hidden_dim)
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.n_features),  # (hidden_dim -> n_features)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store residual
        residual = x  # (batch, input_len, n_features)
        
        # Permute for 1D operations: (batch, n_features, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Extract trend via moving average (per feature)
        trend = self.trend_avg(x_perm)  # (batch, n_features, input_len)
        
        # Seasonal = original - trend
        season = x_perm - trend  # (batch, n_features, input_len)
        
        # Apply learnable decomposition weights (feature-wise)
        trend = trend * self.trend_weight.unsqueeze(0)  # (batch, n_features, input_len) * (1, n_features, 1)
        season = season * self.season_weight.unsqueeze(0)  # (batch, n_features, input_len) * (1, n_features, 1)
        
        # Project trend and season to output length
        trend_out = self.trend_fc(trend)  # (batch, n_features, output_len)
        season_out = self.season_fc(season)  # (batch, n_features, output_len)
        
        # Apply feature-wise MLPs (process each feature independently)
        # Reshape for per-feature MLP: (batch * n_features, output_len, 1)
        trend_mlp_in = trend_out.permute(0, 2, 1).reshape(-1, 1)  # (batch * output_len * n_features, 1)
        season_mlp_in = season_out.permute(0, 2, 1).reshape(-1, 1)  # (batch * output_len * n_features, 1)
        
        trend_mlp_out = self.trend_feature_mlp(trend_mlp_in)  # (batch * output_len * n_features, 1)
        season_mlp_out = self.season_feature_mlp(season_mlp_in)  # (batch * output_len * n_features, 1)
        
        # Reshape back
        trend_out = trend_mlp_out.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        season_out = season_mlp_out.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Combine trend and season
        combined = trend_out + season_out  # (batch, output_len, n_features)
        
        # Cross-channel gating: compute gates from all features
        # Average pool over time for efficiency, then project
        time_pooled = combined.mean(dim=1)  # (batch, n_features)
        gate = self.gate_proj(time_pooled)  # (batch, gate_dim)
        gate = self.gate_act(gate)  # (batch, gate_dim)
        gate = torch.sigmoid(self.gate_out(gate))  # (batch, n_features)
        
        # Apply gates: each feature gets scaled by learned interaction with others
        gated = combined * gate.unsqueeze(1)  # (batch, output_len, n_features) * (batch, 1, n_features)
        
        # Residual connection with learnable weight
        # Project residual to output length
        residual_proj = self.season_fc(residual.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Fusion with residual
        fused_input = torch.cat([gated, residual_proj * self.residual_weight], dim=-1)  # (batch, output_len, 2*n_features)
        output = self.fusion(fused_input)  # (batch, output_len, n_features)
        
        # Add residual
        output = output + residual_proj * 0.1  # (batch, output_len, n_features)
        
        # RevIN denormalization
        output = self.revin(output, mode='denorm')  # (batch, output_len, n_features)
        
        return output  # (batch, output_len, n_features)

