import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SpectralHybrid_Fixed1_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_modes': [16, 64, 256],
            'dropout': [0.0, 0.3],
            'n_heads': [2, 8, 32],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_modes = self.config.get('n_modes', 64)
        self.dropout = self.config.get('dropout', 0.3)
        self.n_heads = self.config.get('n_heads', 8)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        # (batch, input_len, hidden_dim)
        
        # Positional encoding for temporal information
        self.temporal_emb = nn.Parameter(torch.randn(1, self.input_len, self.hidden_dim) * 0.02)
        # (1, input_len, hidden_dim)
        
        # Channel-dependent spectral weights - complex-valued for FFT compatibility
        class SpectralWeightModule(nn.Module):
            def __init__(self, modes, hidden_dim):
                super().__init__()
                # Initialize as complex tensor using real and imag parts
                real = torch.randn(hidden_dim, hidden_dim, modes) * 0.02
                imag = torch.randn(hidden_dim, hidden_dim, modes) * 0.02
                self.weight = nn.Parameter(torch.complex(real, imag))
                # (hidden_dim, hidden_dim, modes) complex
            def forward(self):
                return self.weight
        
        self.spectral_weights = nn.ModuleList([
            SpectralWeightModule(self.n_modes, self.hidden_dim)
            for _ in range(self.n_features)
        ])
        # n_features modules, each with (hidden_dim, hidden_dim, n_modes) complex
        
        # Adaptive frequency gating - per-channel importance for frequency bands
        self.freq_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_heads),
            nn.Sigmoid()
        )
        # (batch, input_len, n_heads)
        
        # Frequency band projections (coarse to fine)
        self.band_projs = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim // self.n_heads)
            for _ in range(self.n_heads)
        ])
        # n_heads projections
        
        # Residual frequency bypass
        self.freq_bypass = nn.Linear(self.input_len, self.output_len)
        # (batch, hidden_dim, output_len)
        
        # Output MLP with spectral features
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        # (batch, output_len, n_features)
        
        # Direct temporal projection for residual
        self.temporal_proj = nn.Linear(self.input_len, self.output_len)
        # (batch, n_features, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')
        # (batch, input_len, n_features)
        
        # Store for residual
        x_input = x
        # (batch, input_len, n_features)
        
        # Project to hidden dim
        h = self.input_proj(x)
        # (batch, input_len, hidden_dim)
        
        # Add temporal embedding
        h = h + self.temporal_emb
        # (batch, input_len, hidden_dim)
        
        # FFT for frequency domain
        h_fft = torch.fft.rfft(h, dim=1)
        # (batch, input_len//2+1, hidden_dim) complex
        
        # Truncate to n_modes
        h_fft_trunc = h_fft[:, :self.n_modes, :]
        # (batch, n_modes, hidden_dim) complex
        
        # Apply channel-dependent spectral mixing with adaptive gating
        # Compute per-head frequency importance
        gate_input = h.mean(dim=1)  # (batch, hidden_dim)
        freq_gates = self.freq_gate(gate_input)  # (batch, n_heads)
        # (batch, n_heads)
        
        # Spectral convolution per feature group
        h_fft_out = torch.zeros_like(h_fft_trunc)
        # (batch, n_modes, hidden_dim) complex
        
        # Split hidden dim into heads
        head_dim = self.hidden_dim // self.n_heads
        
        for head_idx in range(self.n_heads):
            start_idx = head_idx * head_dim
            end_idx = start_idx + head_dim
            
            # Get spectral weights for this head (cycle through features)
            feat_idx = head_idx % self.n_features
            W = self.spectral_weights[feat_idx]()
            # (hidden_dim, hidden_dim, n_modes) complex
            
            # Apply to corresponding head dimension
            h_head = h_fft_trunc[:, :, start_idx:end_idx]
            # (batch, n_modes, head_dim) complex
            
            # Prepare for batch matrix multiply per mode
            h_head_perm = h_head.permute(0, 2, 1)  # (batch, head_dim, n_modes) complex
            
            W_head = W[start_idx:end_idx, start_idx:end_idx, :]
            # (head_dim, head_dim, n_modes) complex
            
            # Batch matrix multiply per mode using complex matmul
            h_head_out = torch.zeros_like(h_head_perm)
            # (batch, head_dim, n_modes) complex
            
            for m in range(min(self.n_modes, h_head_perm.size(-1))):
                W_m = W_head[:, :, m]  # (head_dim, head_dim) complex
                h_m = h_head_perm[:, :, m]  # (batch, head_dim) complex
                # Complex matmul: (batch, head_dim) @ (head_dim, head_dim)^T
                # Use conj().T for proper complex matrix multiplication
                h_head_out[:, :, m] = torch.matmul(h_m, W_m.conj().t())
                # (batch, head_dim) complex
            
            # Apply gating (gate is real, broadcast to complex)
            gate = freq_gates[:, head_idx:head_idx+1]  # (batch, 1)
            # Convert gate to complex for multiplication
            h_head_out = h_head_out * gate.unsqueeze(-1).to(h_head_out.dtype)
            # (batch, head_dim, n_modes) complex
            
            h_fft_out[:, :, start_idx:end_idx] = h_head_out.permute(0, 2, 1)
            # (batch, n_modes, head_dim) complex
        
        # Inverse FFT
        h_fft_full = torch.zeros_like(h_fft)
        # (batch, input_len//2+1, hidden_dim) complex
        h_fft_full[:, :self.n_modes, :] = h_fft_out
        # (batch, input_len//2+1, hidden_dim) complex
        
        h_time = torch.fft.irfft(h_fft_full, n=self.input_len, dim=1)
        # (batch, input_len, hidden_dim) real
        
        # Residual connection
        h = h + h_time
        # (batch, input_len, hidden_dim)
        
        # Frequency bypass: direct temporal projection in frequency space
        h_for_bypass = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        bypass = self.freq_bypass(h_for_bypass)  # (batch, hidden_dim, output_len)
        bypass = bypass.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Output projection
        out = self.output_mlp(bypass)
        # (batch, output_len, n_features)
        
        # Direct temporal residual from input
        x_res = x_input.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_res = self.temporal_proj(x_res)  # (batch, n_features, output_len)
        x_res = x_res.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = out + x_res * 0.5
        # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')
        # (batch, output_len, n_features)
        
        return out

