import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralCDEAttention_Improved_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'cde_width': [64, 256, 512],
            'n_spectral_modes': [8, 32, 128],
            'n_heads': [2, 4, 8],
            'dropout': [0.0, 0.2],
            'channel_mlp_dim': [64, 256, 1024],
            'n_cde_layers': [1, 2, 4],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.cde_width = cfg.get('cde_width', 256)
        self.n_spectral_modes = cfg.get('n_spectral_modes', 32)
        self.n_heads = cfg.get('n_heads', 4)
        self.dropout = cfg.get('dropout', 0.2)
        self.channel_mlp_dim = cfg.get('channel_mlp_dim', 256)
        self.n_cde_layers = cfg.get('n_cde_layers', 2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to CDE space
        self.input_proj = nn.Linear(self.n_features, self.cde_width)
        
        # CDE dynamics function: deep MLP with residual connections
        # Always concatenate [h, dx] so input is cde_width * 2
        cde_layers = []
        for i in range(self.n_cde_layers):
            cde_layers.append(nn.Sequential(
                nn.Linear(self.cde_width * 2, self.cde_width * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.cde_width * 2, self.cde_width * 2)
            ))
        self.cde_dynamics = nn.ModuleList(cde_layers)
        
        # Channel-dependent spectral weights (complex-valued as separate real/imag)
        self.spectral_real = nn.Parameter(
            torch.randn(self.n_features, self.n_spectral_modes) * 0.02
        )  # (n_features, n_spectral_modes)
        self.spectral_imag = nn.Parameter(
            torch.randn(self.n_features, self.n_spectral_modes) * 0.02
        )  # (n_features, n_spectral_modes)
        
        # Adaptive frequency gating per channel
        self.freq_gate = nn.Sequential(
            nn.Linear(self.cde_width, self.n_spectral_modes),
            nn.Sigmoid()
        )  # outputs (batch, n_spectral_modes) per channel after processing
        
        # Actually, we need per-channel gating based on CDE state
        # Let's use a different approach: project CDE state to channel-specific gates
        self.channel_gate_proj = nn.Linear(self.cde_width, self.n_features * self.n_spectral_modes)
        
        # Cross-attention for reading from evolving hidden state
        self.query_proj = nn.Linear(self.n_features, self.cde_width)
        self.key_proj = nn.Linear(self.cde_width, self.cde_width)
        self.value_proj = nn.Linear(self.cde_width, self.cde_width)
        self.attention = nn.MultiheadAttention(
            self.cde_width, self.n_heads, dropout=self.dropout, batch_first=True
        )
        
        # Channel-specific MLPs for spectral processing
        self.channel_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.n_spectral_modes * 2, self.channel_mlp_dim),  # *2 for real+imag
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.channel_mlp_dim, self.channel_mlp_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.channel_mlp_dim // 2, self.n_spectral_modes)
            )
            for _ in range(self.n_features)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.cde_width + self.n_spectral_modes, self.cde_width),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.cde_width, self.n_features)
        )
        
        # Temporal interpolation for output horizon
        self.temporal_embed = nn.Parameter(torch.randn(self.output_len, self.cde_width) * 0.02)
        
    def _cde_solve(self, x, t):
        # x: (batch, input_len, cde_width)
        # t: (input_len,) - normalized time points
        batch_size, seq_len, _ = x.shape  # (batch, input_len, cde_width)
        
        # Simple Euler integration for CDE
        h = torch.zeros(batch_size, self.cde_width, device=x.device)  # (batch, cde_width)
        
        dt = t[1:] - t[:-1]  # (input_len - 1,)
        
        for i in range(seq_len - 1):
            x_i = x[:, i, :]  # (batch, cde_width)
            dx = x[:, i + 1, :] - x_i  # (batch, cde_width)
            
            # Concatenate current state with input increment
            z = torch.cat([h, dx], dim=-1)  # (batch, cde_width * 2)
            
            # Apply dynamics
            for layer in self.cde_dynamics:
                dz = layer(z)  # (batch, cde_width * 2)
                # Split into drift and diffusion
                drift, diffusion = dz.chunk(2, dim=-1)  # each (batch, cde_width)
                h = h + drift * dt[i] + diffusion * dx  # (batch, cde_width)
                z = torch.cat([h, dx], dim=-1)  # (batch, cde_width * 2)
        
        return h  # (batch, cde_width)
    
    def _channel_spectral_transform(self, h, gates):
        # h: (batch, cde_width)
        # gates: (batch, n_features, n_spectral_modes)
        
        batch_size = h.shape[0]  # (batch,)
        
        # Project hidden state to get channel-specific queries
        # Use FFT-based spectral processing per channel
        
        # Create complex spectral weights with gating
        spectral_weights = torch.view_as_complex(
            torch.stack([
                self.spectral_real.unsqueeze(0) * gates,  # (batch, n_features, n_spectral_modes)
                self.spectral_imag.unsqueeze(0) * gates   # (batch, n_features, n_spectral_modes)
            ], dim=-1)
        )  # (batch, n_features, n_spectral_modes)
        
        # Pad to full frequency representation
        full_spectrum = torch.zeros(
            batch_size, self.n_features, self.cde_width // 2 + 1,
            dtype=torch.complex64, device=h.device
        )  # (batch, n_features, cde_width//2+1)
        
        n_modes = min(self.n_spectral_modes, self.cde_width // 2 + 1)
        full_spectrum[:, :, :n_modes] = spectral_weights[:, :, :n_modes]
        
        # Inverse FFT to get time-domain representation
        time_domain = torch.fft.irfft(full_spectrum, n=self.cde_width, dim=-1)  # (batch, n_features, cde_width)
        
        # Apply channel-specific MLPs
        channel_outputs = []
        for c in range(self.n_features):
            ch_in = torch.cat([
                self.spectral_real[c].unsqueeze(0).expand(batch_size, -1),  # (batch, n_spectral_modes)
                self.spectral_imag[c].unsqueeze(0).expand(batch_size, -1)   # (batch, n_spectral_modes)
            ], dim=-1)  # (batch, n_spectral_modes * 2)
            
            # Apply gating
            ch_in = ch_in * gates[:, c, :].mean(dim=-1, keepdim=True)  # (batch, n_spectral_modes * 2)
            ch_out = self.channel_mlps[c](ch_in)  # (batch, n_spectral_modes)
            channel_outputs.append(ch_out)
        
        channel_features = torch.stack(channel_outputs, dim=1)  # (batch, n_features, n_spectral_modes)
        
        return channel_features.mean(dim=1)  # (batch, n_spectral_modes)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]  # (batch,)
        
        x = self.revin(x, mode="norm")  # (batch, input_len, n_features)
        
        # Project to CDE space
        x_cde = self.input_proj(x)  # (batch, input_len, cde_width)
        
        # Create normalized time points
        t = torch.linspace(0, 1, self.input_len, device=x.device)  # (input_len,)
        
        # Solve CDE to get continuous representation
        h_final = self._cde_solve(x_cde, t)  # (batch, cde_width)
        
        # Compute adaptive frequency gates per channel
        gate_logits = self.channel_gate_proj(h_final)  # (batch, n_features * n_spectral_modes)
        gates = gate_logits.view(batch_size, self.n_features, self.n_spectral_modes)  # (batch, n_features, n_spectral_modes)
        gates = torch.sigmoid(gates)  # (batch, n_features, n_spectral_modes)
        
        # Channel-dependent spectral processing
        spectral_features = self._channel_spectral_transform(h_final, gates)  # (batch, n_spectral_modes)
        
        # Expand for output sequence
        h_expanded = h_final.unsqueeze(1).expand(-1, self.output_len, -1)  # (batch, output_len, cde_width)
        spectral_expanded = spectral_features.unsqueeze(1).expand(-1, self.output_len, -1)  # (batch, output_len, n_spectral_modes)
        
        # Add temporal embeddings
        temporal = self.temporal_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, output_len, cde_width)
        h_temporal = h_expanded + temporal  # (batch, output_len, cde_width)
        
        # Cross-attention: query from temporal positions, key/value from CDE evolution
        # Use original CDE states as memory
        queries = self.query_proj(
            x.mean(dim=1, keepdim=True).expand(-1, self.output_len, -1)
        )  # (batch, output_len, cde_width)
        keys = self.key_proj(h_temporal)  # (batch, output_len, cde_width)
        values = self.value_proj(h_temporal)  # (batch, output_len, cde_width)
        
        attn_out, _ = self.attention(queries, keys, values)  # (batch, output_len, cde_width)
        
        # Combine with spectral features
        combined = torch.cat([attn_out, spectral_expanded], dim=-1)  # (batch, output_len, cde_width + n_spectral_modes)
        
        # Output projection
        out = self.output_proj(combined)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode="denorm")  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)

