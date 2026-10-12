import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class RevINNBeats_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_stacks': [2, 4, 8],
            'n_blocks_per_stack': [1, 3, 5],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_stacks = self.config.get('n_stacks', 4)
        self.n_blocks_per_stack = self.config.get('n_blocks_per_stack', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Create stacks for each feature (channel-independent)
        self.stacks = nn.ModuleList()
        for _ in range(self.n_stacks):
            stack_blocks = nn.ModuleList()
            for _ in range(self.n_blocks_per_stack):
                stack_blocks.append(
                    NBeatsBlock(
                        self.input_len, self.output_len, self.hidden_dim, self.dropout
                    )
                )
            self.stacks.append(stack_blocks)
        
        # Final projection for each feature
        self.feature_proj = nn.Linear(self.n_stacks, 1)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Process each feature independently
        outputs = []
        for feat_idx in range(self.n_features):
            feat = x[:, :, feat_idx:feat_idx+1]  # (batch, input_len, 1)
            feat = feat.squeeze(-1)  # (batch, input_len)
            
            stack_outputs = []
            for stack in self.stacks:
                backcast = feat
                for block in stack:
                    backcast, forecast = block(backcast)
                stack_outputs.append(forecast)  # (batch, output_len)
            
            # Combine stack outputs
            stack_tensor = torch.stack(stack_outputs, dim=-1)  # (batch, output_len, n_stacks)
            feat_out = self.feature_proj(stack_tensor).squeeze(-1)  # (batch, output_len)
            outputs.append(feat_out)
        
        out = torch.stack(outputs, dim=-1)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out


class NBeatsBlock(nn.Module):
    def __init__(self, input_len, output_len, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        
        self.theta_b = nn.Linear(hidden_dim, input_len, bias=False)
        self.theta_f = nn.Linear(hidden_dim, output_len, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, input_len)
        h = F.relu(self.fc1(x))  # (batch, hidden_dim)
        h = self.dropout(h)
        h = F.relu(self.fc2(h))  # (batch, hidden_dim)
        h = self.dropout(h)
        h = F.relu(self.fc3(h))  # (batch, hidden_dim)
        h = self.dropout(h)
        h = F.relu(self.fc4(h))  # (batch, hidden_dim)
        
        backcast = self.theta_b(h)  # (batch, input_len)
        forecast = self.theta_f(h)  # (batch, output_len)
        
        return x - backcast, forecast

