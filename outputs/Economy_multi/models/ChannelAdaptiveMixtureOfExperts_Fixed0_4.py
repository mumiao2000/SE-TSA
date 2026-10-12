import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveMixtureOfExperts_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'expert_depth': [1, 3, 6],
            'dropout': [0.0, 0.3],
            'gate_hidden': [16, 64],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.expert_depth = self.config.get('expert_depth', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.gate_hidden = self.config.get('gate_hidden', 32)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Channel-independent expert: processes each feature independently
        self.ci_experts = nn.ModuleList()
        for _ in range(self.n_features):
            layers = []
            for d in range(self.expert_depth):
                in_dim = self.input_len if d == 0 else self.hidden_dim
                out_dim = self.output_len if d == self.expert_depth - 1 else self.hidden_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if d < self.expert_depth - 1:
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(self.dropout_rate))
            self.ci_experts.append(nn.Sequential(*layers))
        
        # Channel-dependent expert: processes all features together
        self.cd_input_proj = nn.Linear(self.input_len * self.n_features, self.hidden_dim)
        cd_layers = []
        for d in range(self.expert_depth - 1):
            cd_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            cd_layers.append(nn.GELU())
            cd_layers.append(nn.Dropout(self.dropout_rate))
        cd_layers.append(nn.Linear(self.hidden_dim, self.output_len * self.n_features))
        self.cd_expert = nn.Sequential(*cd_layers)
        
        # Gating network: decides CI vs CD weight per feature
        self.gate_net = nn.Sequential(
            nn.Linear(self.input_len, self.gate_hidden),
            nn.GELU(),
            nn.Linear(self.gate_hidden, 2),  # 2 experts
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Channel-independent outputs
        ci_outputs = []
        for i in range(self.n_features):
            xi = x[:, :, i]  # (batch, input_len)
            ci_out = self.ci_experts[i](xi)  # (batch, output_len)
            ci_outputs.append(ci_out)
        ci_out = torch.stack(ci_outputs, dim=-1)  # (batch, output_len, n_features)
        
        # Channel-dependent output
        x_flat = x.reshape(batch_size, -1)  # (batch, input_len * n_features)
        x_proj = self.cd_input_proj(x_flat)  # (batch, hidden_dim)
        cd_out = self.cd_expert(x_proj)  # (batch, output_len * n_features)
        cd_out = cd_out.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Compute gates per feature based on input statistics
        gate_inputs = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        gates = self.gate_net(gate_inputs)  # (batch, n_features, 2)
        gates = gates.unsqueeze(2)  # (batch, n_features, 1, 2)
        
        # Blend: stack CI and CD outputs
        ci_cd = torch.stack([ci_out, cd_out], dim=-1)  # (batch, output_len, n_features, 2)
        ci_cd = ci_cd.permute(0, 2, 1, 3)  # (batch, n_features, output_len, 2)
        
        # Apply gates
        out = (ci_cd * gates).sum(dim=-1)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

