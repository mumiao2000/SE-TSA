import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveExpertNet(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_experts': [2, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_experts = self.config.get('n_experts', 4)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Shared (channel-independent) expert: processes all features together
        self.shared_expert = nn.Sequential(
            nn.Linear(input_len, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, output_len)
        )
        
        # Channel-dependent experts: one set per feature group
        self.n_groups = min(self.n_experts, self.n_features)
        features_per_group = (self.n_features + self.n_groups - 1) // self.n_groups
        
        self.group_experts = nn.ModuleList()
        for g in range(self.n_groups):
            n_feat_in_group = min(features_per_group, self.n_features - g * features_per_group)
            if n_feat_in_group > 0:
                expert = nn.Sequential(
                    nn.Linear(input_len, self.hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.hidden_dim // 2, output_len)
                )
                self.group_experts.append(expert)
        
        # Gating network: decides blend for each feature
        # Uses simple statistics of each feature's history
        self.gate_encoder = nn.Sequential(
            nn.Linear(input_len, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.n_groups + 1)  # +1 for shared expert
        )
        
        # Feature assignment to groups
        self.register_buffer('feature_to_group', torch.arange(self.n_features) % self.n_groups)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for per-feature processing
        x_t = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Compute gates for each feature
        gates = self.gate_encoder(x_t)  # (batch, n_features, n_groups + 1)
        gates = F.softmax(gates, dim=-1)  # (batch, n_features, n_groups + 1)
        
        # Shared expert output
        shared_out = self.shared_expert(x_t)  # (batch, n_features, output_len)
        
        # Group expert outputs
        group_outs = []
        for g, expert in enumerate(self.group_experts):
            # Find features in this group
            mask = (self.feature_to_group == g)  # (n_features,)
            if mask.sum() == 0:
                continue
            # Process all features through this expert (will be gated later)
            g_out = expert(x_t)  # (batch, n_features, output_len)
            group_outs.append(g_out)
        
        # Blend outputs
        # Start with shared expert contribution
        out = gates[:, :, 0:1] * shared_out  # (batch, n_features, output_len)
        
        # Add group expert contributions
        for g, g_out in enumerate(group_outs):
            out = out + gates[:, :, g+1:g+2] * g_out  # (batch, n_features, output_len)
        
        # Transpose back
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
