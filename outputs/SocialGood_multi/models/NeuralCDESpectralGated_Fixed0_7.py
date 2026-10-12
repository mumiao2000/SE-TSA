import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralCDESpectralGated_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'n_spectral_modes': [8, 32, 128],
            'cde_width': [32, 128, 512],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.3)
        self.n_spectral_modes = self.config.get('n_spectral_modes', 32)
        self.cde_width = self.config.get('cde_width', 128)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Initial projection to CDE width
        self.input_proj = nn.Linear(self.n_features, self.cde_width)
        
        # CDE vector field: defines dynamics dz/dt = f(z, x)
        # Simplified as GRU-like update for stability
        self.cde_gru = nn.GRU(
            input_size=self.cde_width,
            hidden_size=self.cde_width,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.n_layers > 1 else 0.0,
        )
        
        # Spectral gating: channel-dependent frequency filtering
        self.spectral_proj_in = nn.Linear(self.cde_width, self.hidden_dim)
        
        # Learnable spectral filters per channel (feature)
        # We operate on hidden_dim channels, each with its own frequency response
        self.spectral_real = nn.Parameter(torch.randn(self.hidden_dim, self.n_spectral_modes) * 0.02)
        self.spectral_imag = nn.Parameter(torch.randn(self.hidden_dim, self.n_spectral_modes) * 0.02)
        
        # Multi-scale frequency attention
        self.coarse_attn = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True, dropout=self.dropout_rate)
        self.fine_attn = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True, dropout=self.dropout_rate)
        
        # Gating to combine coarse and fine
        self.scale_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, self.n_features)
        )
        
        # For output length generation: project hidden to each timestep
        self.temporal_decoder = nn.Linear(self.cde_width, self.output_len * self.hidden_dim)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode="norm")  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Project to CDE width
        z = self.input_proj(x)  # (batch, input_len, cde_width)
        
        # Evolve through CDE (GRU approximation for stability)
        z_cde, _ = self.cde_gru(z)  # (batch, input_len, cde_width)
        z_cde = self.dropout(z_cde)
        
        # Take final hidden state as summary
        z_final = z_cde[:, -1, :]  # (batch, cde_width)
        
        # Project to hidden_dim for spectral processing
        h = self.spectral_proj_in(z_final)  # (batch, hidden_dim)
        
        # Generate output sequence via temporal decoder
        h_expanded = self.temporal_decoder(z_final)  # (batch, output_len * hidden_dim)
        h_seq = h_expanded.view(batch_size, self.output_len, self.hidden_dim)  # (batch, output_len, hidden_dim)
        h_seq = self.dropout(h_seq)
        
        # Spectral gating in frequency domain
        # Apply FFT along time dimension for each channel
        h_fft = torch.fft.rfft(h_seq, dim=1)  # (batch, output_len//2+1, hidden_dim)
        
        # Create spectral mask with learnable parameters
        n_freq = h_fft.shape[1]
        modes_to_use = min(self.n_spectral_modes, n_freq)
        
        # Build complex spectral filter
        spectral_filter = torch.zeros(self.hidden_dim, n_freq, device=h_fft.device, dtype=torch.complex64)  # FIXED: self.hidden_dim
        spectral_filter[:, :modes_to_use] = torch.complex(
            self.spectral_real[:, :modes_to_use],
            self.spectral_imag[:, :modes_to_use]
        )
        # (hidden_dim, n_freq)
        
        # Apply channel-dependent filtering: (batch, n_freq, hidden_dim) * (1, 1, hidden_dim)
        h_fft = h_fft * spectral_filter.T.unsqueeze(0)  # (batch, n_freq, hidden_dim)
        
        # Inverse FFT back to time domain
        h_filtered = torch.fft.irfft(h_fft, n=self.output_len, dim=1)  # (batch, output_len, hidden_dim)
        h_filtered = self.layer_norm(h_filtered)
        
        # Multi-scale frequency attention
        # Coarse: low-pass (first half of modes)
        h_coarse_fft = h_fft.clone()
        h_coarse_fft[:, modes_to_use//2:, :] = 0
        h_coarse = torch.fft.irfft(h_coarse_fft, n=self.output_len, dim=1)  # (batch, output_len, hidden_dim)
        
        # Fine: high-pass (second half of modes)
        h_fine_fft = h_fft.clone()
        h_fine_fft[:, :modes_to_use//2, :] = 0
        h_fine = torch.fft.irfft(h_fine_fft, n=self.output_len, dim=1)  # (batch, output_len, hidden_dim)
        
        # Self-attention on each scale
        h_coarse_attn, _ = self.coarse_attn(h_coarse, h_coarse, h_coarse)  # (batch, output_len, hidden_dim)
        h_fine_attn, _ = self.fine_attn(h_fine, h_fine, h_fine)  # (batch, output_len, hidden_dim)
        
        # Learned gating between scales
        gate_input = h_filtered  # (batch, output_len, hidden_dim)
        gates = self.scale_gate(gate_input)  # (batch, output_len, 2)
        
        h_combined = gates[:, :, 0:1] * h_coarse_attn + gates[:, :, 1:2] * h_fine_attn  # (batch, output_len, hidden_dim)
        h_combined = self.layer_norm(h_combined + h_filtered)  # residual + norm
        
        # Final projection to features
        out = self.output_proj(h_combined)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode="denorm")  # (batch, output_len, n_features)
        return out

