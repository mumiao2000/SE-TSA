import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveMixtureOfExperts(BaseForecastModel):
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
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to hidden space
        self.input_proj = nn.Linear(self.input_len, self.hidden_dim)
        
        # Channel-Independent experts: per-feature MLPs
        self.ci_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ) for _ in range(self.n_experts // 2)
        ])
        
        # Channel-Dependent experts: lightweight cross-feature attention
        self.cd_experts = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(self.hidden_dim, self.hidden_dim // 4),
                'k_proj': nn.Linear(self.hidden_dim, self.hidden_dim // 4),
                'v_proj': nn.Linear(self.hidden_dim, self.hidden_dim),
                'out_proj': nn.Linear(self.hidden_dim, self.hidden_dim)
            }) for _ in range(self.n_experts - self.n_experts // 2)
        ])
        
        # Gating network: decides CI vs CD weight per sample and feature
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 2),  # 2 types: CI and CD
            nn.Softmax(dim=-1)
        )
        
        # Expert weighting within each type
        self.ci_gate = nn.Linear(self.hidden_dim, len(self.ci_experts))
        self.cd_gate = nn.Linear(self.hidden_dim, len(self.cd_experts))
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.output_len)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project: (batch, n_features, hidden_dim)
        x_proj = self.input_proj(x.permute(0, 2, 1))  # (batch, n_features, hidden_dim)
        
        # Compute CI and CD features
        batch_size, n_features, hidden_dim = x_proj.shape
        
        # Channel-Independent path
        ci_outputs = []
        for expert in self.ci_experts:
            ci_out = expert(x_proj)  # (batch, n_features, hidden_dim)
            ci_outputs.append(ci_out)
        ci_stack = torch.stack(ci_outputs, dim=-1)  # (batch, n_features, hidden_dim, n_ci_experts)
        
        # Channel-Dependent path (cross-feature attention)
        cd_outputs = []
        for expert in self.cd_experts:
            q = expert['q_proj'](x_proj)  # (batch, n_features, hidden_dim//4)
            k = expert['k_proj'](x_proj)  # (batch, n_features, hidden_dim//4)
            v = expert['v_proj'](x_proj)  # (batch, n_features, hidden_dim)
            
            # Cross-feature attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)  # (batch, n_features, n_features)
            attn = F.softmax(scores, dim=-1)
            cd_out = torch.matmul(attn, v)  # (batch, n_features, hidden_dim)
            cd_out = expert['out_proj'](cd_out)  # (batch, n_features, hidden_dim)
            cd_outputs.append(cd_out)
        cd_stack = torch.stack(cd_outputs, dim=-1)  # (batch, n_features, hidden_dim, n_cd_experts)
        
        # Gating: decide CI vs CD ratio per feature
        # Global feature stats for gating
        feat_mean = x_proj.mean(dim=-1, keepdim=True)  # (batch, n_features, 1)
        feat_std = x_proj.std(dim=-1, keepdim=True)  # (batch, n_features, 1)
        gate_input = torch.cat([feat_mean.expand(-1, -1, hidden_dim), 
                                feat_std.expand(-1, -1, hidden_dim)], dim=-1)  # (batch, n_features, hidden_dim*2)
        type_weights = self.gate(gate_input)  # (batch, n_features, 2)
        
        # Expert weights within each type
        ci_weights = F.softmax(self.ci_gate(x_proj), dim=-1)  # (batch, n_features, n_ci_experts)
        cd_weights = F.softmax(self.cd_gate(x_proj), dim=-1)  # (batch, n_features, n_cd_experts)
        
        # Weighted combination of CI experts
        ci_combined = torch.sum(ci_stack * ci_weights.unsqueeze(-2), dim=-1)  # (batch, n_features, hidden_dim)
        
        # Weighted combination of CD experts
        cd_combined = torch.sum(cd_stack * cd_weights.unsqueeze(-2), dim=-1)  # (batch, n_features, hidden_dim)
        
        # Blend CI and CD
        ci_weight = type_weights[:, :, 0:1]  # (batch, n_features, 1)
        cd_weight = type_weights[:, :, 1:2]  # (batch, n_features, 1)
        mixed = ci_weight * ci_combined + cd_weight * cd_combined  # (batch, n_features, hidden_dim)
        
        # Decode to output
        out = self.decoder(mixed)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)
