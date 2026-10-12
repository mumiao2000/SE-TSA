import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveMixture_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_experts': [2, 4, 8],
            'expert_depth': [1, 3, 6],
            'dropout': [0.0, 0.3],
            'gating_hidden': [16, 64, 256]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_experts = self.config.get('n_experts', 4)
        self.expert_depth = self.config.get('expert_depth', 3)
        self.dropout = self.config.get('dropout', 0.1)
        self.gating_hidden = self.config.get('gating_hidden', 64)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Gating network: predicts mixture weights based on input statistics
        self.gating_encoder = nn.Sequential(
            nn.Linear(self.n_features * 2, self.gating_hidden),  # mean and std per feature
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.gating_hidden, self.gating_hidden // 2),
            nn.ReLU(),
            nn.Linear(self.gating_hidden // 2, self.n_experts)
        )
        
        # Channel-independent experts: separate processing per feature
        self.ci_experts = nn.ModuleList()
        for _ in range(self.n_experts):
            layers = []
            for d in range(self.expert_depth):
                in_ch = self.input_len if d == 0 else self.hidden_dim
                out_ch = self.output_len if d == self.expert_depth - 1 else self.hidden_dim
                layers.extend([
                    nn.Conv1d(self.n_features, self.n_features, kernel_size=3, padding=1, groups=self.n_features),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ])
            # Final projection
            self.ci_experts.append(nn.Sequential(
                nn.Linear(self.input_len, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.output_len)
            ))
        
        # Channel-dependent experts: full cross-channel mixing
        self.cd_experts = nn.ModuleList()
        for _ in range(self.n_experts):
            layers = []
            for d in range(self.expert_depth):
                layers.extend([
                    nn.Linear(self.input_len if d == 0 else self.hidden_dim, 
                             self.output_len if d == self.expert_depth - 1 else self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout)
                ])
            self.cd_experts.append(nn.Sequential(*layers))
        
        # Project to hidden_dim for cross-channel attention
        self.proj_to_hidden = nn.Linear(self.n_features, self.hidden_dim)
        # Cross-channel attention for channel-dependent path
        self.cross_channel_attn = nn.MultiheadAttention(self.hidden_dim, num_heads=4, batch_first=True, dropout=self.dropout)
        # Project back to n_features
        self.proj_from_hidden = nn.Linear(self.hidden_dim, self.n_features)
        
        # Feature-wise linear modulation for adaptive feature interaction
        self.film_gamma = nn.Linear(self.n_features, self.n_features)
        self.film_beta = nn.Linear(self.n_features, self.n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Compute gating inputs: per-feature mean and std
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        x_std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        gate_input = torch.cat([x_mean.squeeze(1), x_std.squeeze(1)], dim=-1)  # (batch, n_features * 2)
        gate_logits = self.gating_encoder(gate_input)  # (batch, n_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (batch, n_experts)
        
        # Channel-independent path: process each feature independently
        # x_ci: (batch, n_features, input_len)
        x_ci = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        ci_outputs = []
        for expert in self.ci_experts:
            out = expert(x_ci)  # (batch, n_features, output_len)
            ci_outputs.append(out)
        ci_outputs = torch.stack(ci_outputs, dim=0)  # (n_experts, batch, n_features, output_len)
        
        # Channel-dependent path: full feature mixing
        x_cd = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        cd_outputs = []
        for expert in self.cd_experts:
            out = expert(x_cd)  # (batch, n_features, output_len)
            cd_outputs.append(out)
        cd_outputs = torch.stack(cd_outputs, dim=0)  # (n_experts, batch, n_features, output_len)
        
        # Apply FiLM conditioning to blend paths
        film_gamma = self.film_gamma(x_mean.squeeze(1)).unsqueeze(-1)  # (batch, n_features, 1)
        film_beta = self.film_beta(x_mean.squeeze(1)).unsqueeze(-1)  # (batch, n_features, 1)
        
        # Weighted mixture of experts
        gate_w = gate_weights.permute(1, 0).unsqueeze(-1).unsqueeze(-1)  # (n_experts, batch, 1, 1)
        ci_mixed = (ci_outputs * gate_w).sum(dim=0)  # (batch, n_features, output_len)
        cd_mixed = (cd_outputs * gate_w).sum(dim=0)  # (batch, n_features, output_len)
        
        # Adaptive blend with FiLM
        output = film_gamma * ci_mixed + (1 - film_gamma) * cd_mixed + film_beta  # (batch, n_features, output_len)
        
        # Cross-channel attention refinement
        output_t = output.permute(0, 2, 1)  # (batch, output_len, n_features)
        output_proj = self.proj_to_hidden(output_t)  # (batch, output_len, hidden_dim)
        attn_out, _ = self.cross_channel_attn(output_proj, output_proj, output_proj)  # (batch, output_len, hidden_dim)
        output_proj = output_proj + attn_out  # (batch, output_len, hidden_dim)
        output = self.proj_from_hidden(output_proj)  # (batch, output_len, n_features)
        
        output = self.revin(output, mode='denorm')  # (batch, output_len, n_features)
        return output

