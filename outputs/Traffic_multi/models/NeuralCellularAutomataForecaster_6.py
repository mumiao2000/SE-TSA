import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralCellularAutomataForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_rules': [4, 8, 16],
            'state_dim': [32, 128, 512],
            'n_steps': [4, 8, 16],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.n_rules = self.config.get('n_rules', 8)
        self.state_dim = self.config.get('state_dim', 128)
        self.n_steps = self.config.get('n_steps', 8)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Embed input to CA state space
        self.input_embed = nn.Linear(self.n_features, self.state_dim)
        
        # Multiple CA rule kernels (like different convolution filters)
        self.rule_convs = nn.ModuleList([
            nn.Conv1d(self.state_dim, self.state_dim, kernel_size=ks, padding=ks//2, groups=self.state_dim)
            for ks in [3, 5, 7]  # Different receptive fields
        ])
        
        # Rule selection network (which rule to apply where)
        self.rule_selector = nn.Sequential(
            nn.Linear(self.state_dim, self.n_rules),
            nn.Softmax(dim=-1)
        )
        
        # Rule parameterization: each rule is a learned MLP
        self.rule_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.state_dim * len(self.rule_convs), self.state_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.state_dim * 2, self.state_dim)
            ) for _ in range(self.n_rules)
        ])
        
        # Global coordination: pooled state influences all cells
        self.global_pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.state_dim, self.state_dim),
            nn.Tanh()
        )
        
        # State update with residual
        self.update_gate = nn.Linear(self.state_dim * 2, self.state_dim)
        
        # Output projection from state to prediction
        self.output_head = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim // 2),
            nn.GELU(),
            nn.Linear(self.state_dim // 2, self.n_features)
        )
        
        # Learned step sizes for unrolling
        self.step_sizes = nn.Parameter(torch.ones(self.n_steps) * 0.5)
        
    def ca_step(self, state, global_bias):
        # state: (batch, seq_len, state_dim)
        # global_bias: (batch, state_dim)
        
        # Apply conv rules
        state_perm = state.permute(0, 2, 1)  # (batch, state_dim, seq_len)
        rule_outputs = []
        for conv in self.rule_convs:
            conv_out = conv(state_perm)  # (batch, state_dim, seq_len)
            rule_outputs.append(conv_out.permute(0, 2, 1))  # (batch, seq_len, state_dim)
        
        # Concatenate rule outputs
        rules_combined = torch.cat(rule_outputs, dim=-1)  # (batch, seq_len, state_dim * n_convs)
        
        # Select which rule to apply at each position
        selector = self.rule_selector(state)  # (batch, seq_len, n_rules)
        
        # Apply selected rules
        rule_results = torch.stack([mlp(rules_combined) for mlp in self.rule_mlps], dim=-1)  # (batch, seq_len, state_dim, n_rules)
        selected = (rule_results * selector.unsqueeze(2)).sum(dim=-1)  # (batch, seq_len, state_dim)
        
        # Add global bias
        biased = selected + global_bias.unsqueeze(1)  # (batch, seq_len, state_dim)
        
        # Gated update
        update_input = torch.cat([state, biased], dim=-1)  # (batch, seq_len, state_dim * 2)
        gate = torch.sigmoid(self.update_gate(update_input))  # (batch, seq_len, state_dim)
        new_state = (1 - gate) * state + gate * biased  # (batch, seq_len, state_dim)
        
        return new_state
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Embed to state space
        state = self.input_embed(x)  # (batch, input_len, state_dim)
        
        # Pad or interpolate to working sequence length (use input_len as spatial dim)
        seq_len = state.size(1)
        
        # Run CA dynamics
        for step in range(self.n_steps):
            # Compute global bias from pooled state
            state_perm = state.permute(0, 2, 1)  # (batch, state_dim, seq_len)
            global_bias = self.global_pool_proj(state_perm)  # (batch, state_dim)
            
            # CA update with learned step size
            new_state = self.ca_step(state, global_bias)
            state = state + self.step_sizes[step] * (new_state - state)  # (batch, seq_len, state_dim)
        
        # Extract predictions: use last positions to forecast future
        # Take the last few states and project to output
        last_states = state[:, -self.output_len:, :]  # (batch, min(output_len, seq_len), state_dim)
        
        # If needed, pad or use final state repeated
        if last_states.size(1) < self.output_len:
            final_state = state[:, -1:, :]  # (batch, 1, state_dim)
            repeat = self.output_len - last_states.size(1)
            last_states = torch.cat([last_states, final_state.repeat(1, repeat, 1)], dim=1)
        
        out = self.output_head(last_states)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

