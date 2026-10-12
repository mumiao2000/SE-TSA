import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralODEHybridForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [32, 128, 512],
            'hidden_dim': [64, 256, 1024],
            'n_steps': [4, 8, 16],
            'n_heads': [2, 4, 8],
            'temporal_scales': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 128)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_steps = self.config.get('n_steps', 8)
        self.n_heads = self.config.get('n_heads', 4)
        self.temporal_scales = self.config.get('temporal_scales', 4)
        
        # Ensure latent_dim divisible by n_heads for attention
        self.attn_dim = (self.latent_dim // self.n_heads) * self.n_heads
        if self.attn_dim == 0:
            self.attn_dim = self.n_heads
            self.latent_dim = self.n_heads
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.latent_dim)
        
        # Multi-scale temporal encoder
        self.scale_convs = nn.ModuleList()
        for scale in range(1, self.temporal_scales + 1):
            kernel_size = min(2 * scale + 1, self.input_len)
            padding = kernel_size // 2
            self.scale_convs.append(
                nn.Conv1d(self.latent_dim, self.latent_dim // self.temporal_scales, 
                         kernel_size=kernel_size, padding=padding, groups=1)
            )
        
        self.scale_fusion = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Sigmoid()
        )
        
        # Cross-channel attention
        self.cross_attn_q = nn.Linear(self.latent_dim, self.attn_dim)
        self.cross_attn_k = nn.Linear(self.latent_dim, self.attn_dim)
        self.cross_attn_v = nn.Linear(self.latent_dim, self.attn_dim)
        self.cross_attn_out = nn.Linear(self.attn_dim, self.latent_dim)
        
        # ODE dynamics network
        self.ode_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.latent_dim, self.n_features)
        
        # Learnable time embeddings for prediction horizons
        self.time_embed = nn.Embedding(self.output_len, self.latent_dim)
        
    def _cross_channel_attention(self, x):
        # x: (batch, seq_len, latent_dim)
        B, T, D = x.shape  # (batch, seq_len, latent_dim)
        
        # Reshape for multi-head attention across features (channels)
        q = self.cross_attn_q(x).view(B, T, self.n_heads, -1).permute(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        k = self.cross_attn_k(x).view(B, T, self.n_heads, -1).permute(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        v = self.cross_attn_v(x).view(B, T, self.n_heads, -1).permute(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)  # (batch, n_heads, seq_len, seq_len)
        attn = F.softmax(scores, dim=-1)  # (batch, n_heads, seq_len, seq_len)
        
        out = torch.matmul(attn, v)  # (batch, n_heads, seq_len, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.attn_dim)  # (batch, seq_len, attn_dim)
        out = self.cross_attn_out(out)  # (batch, seq_len, latent_dim)
        
        return out + x  # residual
    
    def _ode_step(self, z, dt=1.0):
        # Simple Euler integration step
        dz = self.ode_net(z)  # (batch, latent_dim)
        return z + dt * dz  # (batch, latent_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        B = x.size(0)  # batch size
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to latent space
        z = self.input_proj(x)  # (batch, input_len, latent_dim)
        
        # Multi-scale temporal encoding
        z_perm = z.permute(0, 2, 1)  # (batch, latent_dim, input_len)
        scale_outputs = []
        for conv in self.scale_convs:
            scale_out = conv(z_perm)  # (batch, latent_dim//scales, input_len)
            # Adaptive pooling to match input_len
            if scale_out.size(-1) != self.input_len:
                scale_out = F.interpolate(scale_out, size=self.input_len, mode='linear', align_corners=False)  # (batch, latent_dim//scales, input_len)
            scale_outputs.append(scale_out)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(scale_outputs, dim=1)  # (batch, latent_dim, input_len)
        multi_scale = multi_scale.permute(0, 2, 1)  # (batch, input_len, latent_dim)
        
        # Gating mechanism to fuse multi-scale information
        gate = self.scale_fusion(z)  # (batch, input_len, latent_dim)
        z = gate * z + (1 - gate) * multi_scale  # (batch, input_len, latent_dim)
        
        # Cross-channel attention
        z = self._cross_channel_attention(z)  # (batch, input_len, latent_dim)
        
        # Pool to get initial condition for ODE (use last timestep)
        z0 = z[:, -1, :]  # (batch, latent_dim)
        
        # Evolve through ODE for prediction horizon
        predictions = []
        z_t = z0  # (batch, latent_dim)
        
        dt = self.output_len / self.n_steps
        
        for t in range(self.output_len):
            # Multiple ODE steps per prediction point for stability
            n_internal = max(1, self.n_steps // self.output_len) if self.n_steps >= self.output_len else 1
            for _ in range(n_internal):
                z_t = self._ode_step(z_t, dt=dt / n_internal)  # (batch, latent_dim)
            
            # Add time-specific embedding
            t_embed = self.time_embed(torch.tensor(t, device=x.device))  # (latent_dim,)
            t_embed = t_embed.unsqueeze(0).expand(B, -1)  # (batch, latent_dim)
            z_t_timed = z_t + t_embed  # (batch, latent_dim)
            
            pred = self.output_proj(z_t_timed)  # (batch, n_features)
            predictions.append(pred)
        
        out = torch.stack(predictions, dim=1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
