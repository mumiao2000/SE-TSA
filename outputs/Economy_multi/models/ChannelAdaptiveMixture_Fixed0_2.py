import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveMixture_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'expert_dim': [64, 256, 1024],
            'n_expert_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'gate_hidden': [32, 128, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.expert_dim = self.config.get('expert_dim', 256)
        self.n_expert_layers = self.config.get('n_expert_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.gate_hidden = self.config.get('gate_hidden', 128)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Gating network: per-feature decision on expert usage
        self.gate_encoder = nn.Sequential(
            nn.Linear(input_len, self.gate_hidden),
            nn.ReLU(),
            nn.Linear(self.gate_hidden, 2),  # 2 experts
            nn.Softmax(dim=-1)
        )
        
        # Expert 1: Channel-independent temporal convolutions
        # Each feature processed independently
        # First layer: 1 -> expert_dim
        self.ci_first_conv = nn.Conv1d(1, self.expert_dim, kernel_size=3, padding=1)
        self.ci_first_norm = nn.LayerNorm(self.expert_dim)
        
        # Middle layers: expert_dim -> expert_dim (residual)
        self.ci_expert = nn.ModuleList()
        self.ci_norms = nn.ModuleList()
        for i in range(self.n_expert_layers - 1):
            self.ci_expert.append(nn.Sequential(
                nn.Conv1d(self.expert_dim, self.expert_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Conv1d(self.expert_dim, self.expert_dim, kernel_size=3, padding=1),
            ))
            self.ci_norms.append(nn.LayerNorm(self.expert_dim))
        
        # Expert 2: Channel-dependent graph attention
        # Features interact through attention
        self.cd_query = nn.Linear(1, self.expert_dim)
        self.cd_key = nn.Linear(1, self.expert_dim)
        self.cd_value = nn.Linear(1, self.expert_dim)
        self.cd_n_heads = 4
        self.cd_head_dim = self.expert_dim // self.cd_n_heads
        
        self.cd_expert = nn.ModuleList()
        for i in range(self.n_expert_layers):
            self.cd_expert.append(nn.Sequential(
                nn.Linear(self.expert_dim, self.expert_dim * 4),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.expert_dim * 4, self.expert_dim)
            ))
        self.cd_norms = nn.ModuleList([nn.LayerNorm(self.expert_dim) for _ in range(self.n_expert_layers)])
        
        # Output projections
        self.ci_output = nn.Linear(self.expert_dim * input_len, output_len)
        self.cd_output = nn.Linear(self.expert_dim, output_len)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_len * 2, self.expert_dim),
            nn.ReLU(),
            nn.Linear(self.expert_dim, output_len)
        )
        
    def ci_forward(self, x):
        # x: (batch, n_features, input_len)
        # Process each feature independently
        batch_size = x.shape[0]
        
        # Reshape: each feature is a separate sample
        x_reshaped = x.reshape(batch_size * self.n_features, 1, self.input_len)  # (batch*n_features, 1, input_len)
        
        # First conv: 1 -> expert_dim
        h = self.ci_first_conv(x_reshaped)  # (batch*n_features, expert_dim, input_len)
        h = h.permute(0, 2, 1)  # (batch*n_features, input_len, expert_dim)
        h = self.ci_first_norm(h)
        h = F.relu(h)
        h = h.permute(0, 2, 1)  # (batch*n_features, expert_dim, input_len)
        
        # Residual blocks
        for conv, norm in zip(self.ci_expert, self.ci_norms):
            residual = h
            h = conv(h)  # (batch*n_features, expert_dim, input_len)
            h = h + residual  # residual connection
            h = h.permute(0, 2, 1)  # (batch*n_features, input_len, expert_dim)
            h = norm(h)
            h = F.relu(h)
            h = h.permute(0, 2, 1)  # (batch*n_features, expert_dim, input_len)
        
        # Flatten and project
        h = h.reshape(batch_size * self.n_features, self.expert_dim * self.input_len)  # (batch*n_features, expert_dim*input_len)
        out = self.ci_output(h)  # (batch*n_features, output_len)
        out = out.reshape(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        return out  # (batch, n_features, output_len)
    
    def cd_forward(self, x):
        # x: (batch, n_features, input_len)
        # Process with cross-feature attention
        batch_size = x.shape[0]
        
        # Embed: (batch, n_features, input_len, 1)
        x = x.unsqueeze(-1)  # (batch, n_features, input_len, 1)
        
        # For each time step, apply attention across features
        h = x.permute(0, 2, 1, 3)  # (batch, input_len, n_features, 1)
        h = h.reshape(batch_size * self.input_len, self.n_features, 1)  # (batch*input_len, n_features, 1)
        
        # Multi-head attention
        q = self.cd_query(h)  # (batch*input_len, n_features, expert_dim)
        k = self.cd_key(h)  # (batch*input_len, n_features, expert_dim)
        v = self.cd_value(h)  # (batch*input_len, n_features, expert_dim)
        
        # Reshape for multi-head
        q = q.reshape(batch_size * self.input_len, self.n_features, self.cd_n_heads, self.cd_head_dim)  # (batch*input_len, n_features, n_heads, head_dim)
        k = k.reshape(batch_size * self.input_len, self.n_features, self.cd_n_heads, self.cd_head_dim)  # (batch*input_len, n_features, n_heads, head_dim)
        v = v.reshape(batch_size * self.input_len, self.n_features, self.cd_n_heads, self.cd_head_dim)  # (batch*input_len, n_features, n_heads, head_dim)
        
        q = q.permute(0, 2, 1, 3)  # (batch*input_len, n_heads, n_features, head_dim)
        k = k.permute(0, 2, 1, 3)  # (batch*input_len, n_heads, n_features, head_dim)
        v = v.permute(0, 2, 1, 3)  # (batch*input_len, n_heads, n_features, head_dim)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.cd_head_dim ** 0.5)  # (batch*input_len, n_heads, n_features, n_features)
        attn = F.softmax(scores, dim=-1)  # (batch*input_len, n_heads, n_features, n_features)
        
        out = torch.matmul(attn, v)  # (batch*input_len, n_heads, n_features, head_dim)
        out = out.permute(0, 2, 1, 3)  # (batch*input_len, n_features, n_heads, head_dim)
        out = out.reshape(batch_size * self.input_len, self.n_features, self.expert_dim)  # (batch*input_len, n_features, expert_dim)
        
        # MLP layers
        for mlp, norm in zip(self.cd_expert, self.cd_norms):
            out = out + mlp(norm(out))  # (batch*input_len, n_features, expert_dim)
        
        # Reshape and project to output
        out = out.reshape(batch_size, self.input_len, self.n_features, self.expert_dim)  # (batch, input_len, n_features, expert_dim)
        out = out.mean(dim=1)  # (batch, n_features, expert_dim) - average over time
        
        out = self.cd_output(out)  # (batch, n_features, output_len)
        return out  # (batch, n_features, output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Compute gating weights per feature
        # Use statistics of each feature's time series
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        gate_input = x_perm  # (batch, n_features, input_len)
        gates = self.gate_encoder(gate_input)  # (batch, n_features, 2)
        # gates[:, :, 0] = weight for CI, gates[:, :, 1] = weight for CD
        
        # Expert outputs
        ci_out = self.ci_forward(x_perm)  # (batch, n_features, output_len)
        cd_out = self.cd_forward(x_perm)  # (batch, n_features, output_len)
        
        # Mix according to gates
        gates = gates.unsqueeze(-1)  # (batch, n_features, 2, 1)
        ci_weight = gates[:, :, 0:1, :]  # (batch, n_features, 1, 1)
        cd_weight = gates[:, :, 1:2, :]  # (batch, n_features, 1, 1)
        
        # Weighted combination
        mixed = ci_out * ci_weight.squeeze(2) + cd_out * cd_weight.squeeze(2)  # (batch, n_features, output_len)
        
        out = mixed.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

