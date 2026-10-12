import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralControlledODE_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3],
            'n_attention_heads': [2, 8, 32]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_attention_heads = self.config.get('n_attention_heads', 8)
        
        # Ensure hidden_dim is divisible by n_attention_heads for attention
        # This is guaranteed by proper config selection, but we also adjust if needed
        self.attn_dim = self.hidden_dim
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Input projection to attention dimension
        self.input_proj = nn.Linear(self.n_features, self.attn_dim)
        
        # Control path encoder: attention over history
        self.control_attention = nn.MultiheadAttention(
            self.attn_dim, self.n_attention_heads, dropout=self.dropout, batch_first=True
        )
        
        # Initial projection from raw features to hidden
        self.initial_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Control projection after attention
        self.control_proj = nn.Sequential(
            nn.Linear(self.attn_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # CDE vector field (drift): f(h, t) -> dh/dt
        self.drift_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.drift_layers.append(nn.Sequential(
                nn.Linear(self.hidden_dim + 1, self.hidden_dim),  # +1 for time
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            ))
        
        # Diffusion/Control influence
        self.control_influence = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Output decoder
        self.output_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
    def cde_solve(self, h0, control_path, n_steps=10):
        # Simplified Euler integration for CDE
        # h0: (batch, hidden_dim)
        # control_path: (batch, input_len, hidden_dim)
        
        batch_size = h0.size(0)  # (batch,)
        h = h0  # (batch, hidden_dim)
        
        # Interpolate control
        control_interp = F.interpolate(
            control_path.transpose(1, 2),  # (batch, hidden_dim, input_len)
            size=n_steps,
            mode='linear',
            align_corners=True
        ).transpose(1, 2)  # (batch, n_steps, hidden_dim)
        
        # Simple Euler: dh = f(h, t) * dt + g(h, control) * dcontrol
        dt = 1.0 / n_steps
        outputs = []
        
        for i in range(n_steps):
            t = torch.full((batch_size, 1), i * dt, device=h0.device)  # (batch, 1)
            h_t = torch.cat([h, t], dim=-1)  # (batch, hidden_dim + 1)
            
            # Drift
            dh = self.drift_layers[0](h_t)  # (batch, hidden_dim)
            for layer in self.drift_layers[1:]:
                dh = dh + layer(h_t)  # (batch, hidden_dim)
            
            # Control influence
            ctrl = control_interp[:, min(i, control_interp.size(1)-1), :]  # (batch, hidden_dim)
            ctrl_influence = self.control_influence(torch.cat([h, ctrl], dim=-1))  # (batch, hidden_dim)
            
            # Update
            h = h + dh * dt + ctrl_influence * dt  # (batch, hidden_dim)
            
            # Collect outputs at appropriate times
            if i % max(1, n_steps // self.output_len) == 0 and len(outputs) < self.output_len:
                outputs.append(h)
        
        # Pad if needed
        while len(outputs) < self.output_len:
            outputs.append(h)
        
        return torch.stack(outputs, dim=1)  # (batch, output_len, hidden_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.norm(x)  # (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Project to attention dimension
        x_attn = self.input_proj(x)  # (batch, input_len, attn_dim)
        
        # Encode control path with attention
        ctrl_encoded, _ = self.control_attention(x_attn, x_attn, x_attn)  # (batch, input_len, attn_dim)
        ctrl_path = self.control_proj(ctrl_encoded)  # (batch, input_len, hidden_dim)
        
        # Initial hidden state from last observation
        h0 = self.initial_proj(x[:, -1, :])  # (batch, hidden_dim)
        
        # Solve CDE
        trajectory = self.cde_solve(h0, ctrl_path)  # (batch, output_len, hidden_dim)
        
        # Decode to features
        output = self.output_net(trajectory)  # (batch, output_len, n_features)
        
        return output
