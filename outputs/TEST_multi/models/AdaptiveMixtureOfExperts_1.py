import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class AdaptiveMixtureOfExperts(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'expert_hidden': [32, 128, 512],
            'n_experts_per_feature': [2, 4, 8],
            'gate_hidden': [16, 64, 256],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.expert_hidden = self.config.get('expert_hidden', 128)
        self.n_experts = self.config.get('n_experts_per_feature', 4)
        self.gate_hidden = self.config.get('gate_hidden', 64)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Per-feature experts: each feature has n_experts MLPs
        self.experts = nn.ModuleList()
        for _ in range(self.n_features):
            feature_experts = nn.ModuleList()
            for _ in range(self.n_experts):
                expert = nn.Sequential(
                    nn.Linear(self.input_len, self.expert_hidden),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.expert_hidden, self.output_len)
                )
                feature_experts.append(expert)
            self.experts.append(feature_experts)
        
        # Global gating network: looks at all features to decide expert weights
        self.gate_encoder = nn.Sequential(
            nn.Linear(self.n_features * self.input_len, self.gate_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.gate_hidden, self.n_features * self.n_experts)
        )
        
        # Feature interaction layer after expert mixing
        self.feature_mixer = nn.Sequential(
            nn.Linear(self.n_features, self.n_features * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.n_features * 2, self.n_features)
        )

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Compute gating weights from global context
        x_flat = x.reshape(batch_size, -1)  # (batch, input_len * n_features)
        gate_logits = self.gate_encoder(x_flat)  # (batch, n_features * n_experts)
        gate_weights = F.softmax(gate_logits.reshape(batch_size, self.n_features, self.n_experts), dim=-1)  # (batch, n_features, n_experts)
        
        # Apply per-feature experts
        outputs = []
        for feat_idx in range(self.n_features):
            feat_input = x[:, :, feat_idx]  # (batch, input_len)
            expert_outputs = []
            for expert_idx in range(self.n_experts):
                expert_out = self.experts[feat_idx][expert_idx](feat_input)  # (batch, output_len)
                expert_outputs.append(expert_out)
            # Stack and weight by gate
            expert_stack = torch.stack(expert_outputs, dim=-1)  # (batch, output_len, n_experts)
            gate_w = gate_weights[:, feat_idx, :].unsqueeze(1)  # (batch, 1, n_experts)
            mixed = (expert_stack * gate_w).sum(dim=-1)  # (batch, output_len)
            outputs.append(mixed)
        
        # Combine features
        out = torch.stack(outputs, dim=-1)  # (batch, output_len, n_features)
        
        # Feature interaction
        out = self.feature_mixer(out)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

