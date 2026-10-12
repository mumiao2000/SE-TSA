import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanAutoencoder(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [64, 256, 1024],
            'n_koopman_modes': [8, 32, 128],
            'encoder_layers': [2, 4, 8],
            'dropout': [0.0, 0.3],
            'freq_dim': [16, 64, 256]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.latent_dim = cfg.get('latent_dim', 256)
        self.n_koopman_modes = cfg.get('n_koopman_modes', 32)
        self.encoder_layers = cfg.get('encoder_layers', 4)
        self.dropout = cfg.get('dropout', 0.1)
        self.freq_dim = cfg.get('freq_dim', 64)
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(n_features, affine=True)
        
        # Encoder: maps (batch, input_len, n_features) -> Koopman observables
        encoder_modules = []
        in_dim = n_features
        for i in range(self.encoder_layers):
            out_dim = self.latent_dim if i == self.encoder_layers - 1 else self.latent_dim // 2
            encoder_modules.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_modules)
        
        # Koopman operator: diagonal with complex eigenvalues
        # Represent as magnitude and phase
        self.koopman_magnitude = nn.Parameter(torch.rand(self.latent_dim) * 0.9 + 0.05)  # (0.05, 0.95)
        self.koopman_phase = nn.Parameter(torch.rand(self.latent_dim) * 2 * 3.14159)
        
        # Frequency-aware embedding for seasonal patterns
        self.freq_encoder = nn.Linear(n_features, self.freq_dim)
        self.freq_proj = nn.Linear(self.freq_dim, self.latent_dim)
        
        # Mode gating: which Koopman modes to use
        self.mode_gate = nn.Sequential(
            nn.Linear(self.latent_dim, self.n_koopman_modes),
            nn.Softmax(dim=-1)
        )
        
        # Mode-specific projections
        self.mode_proj = nn.ModuleList([
            nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_koopman_modes)
        ])
        
        # Decoder
        decoder_modules = []
        in_dim = self.latent_dim
        for i in range(self.encoder_layers):
            out_dim = n_features if i == self.encoder_layers - 1 else self.latent_dim // 2
            if i < self.encoder_layers - 1:
                decoder_modules.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout)
                ])
            else:
                decoder_modules.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_modules)
        
        # Temporal integration: how to go from input_len to output_len
        self.temporal_mixer = nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1)
        
        # Direct prediction head
        self.predict_head = nn.Linear(input_len, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch, seq_len, _ = x.shape
        
        # Normalize
        x_norm = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Encode each timestep
        z = self.encoder(x_norm)  # (batch, input_len, latent_dim)
        
        # Frequency-aware enhancement
        freq_emb = torch.fft.rfft(x_norm, dim=1).real  # (batch, input_len//2+1, n_features)
        freq_emb = freq_emb[:, :seq_len, :]  # truncate to match
        if freq_emb.shape[1] < seq_len:
            freq_emb = F.pad(freq_emb, (0, 0, 0, seq_len - freq_emb.shape[1]))
        freq_emb = self.freq_encoder(freq_emb)  # (batch, input_len, freq_dim)
        freq_emb = self.freq_proj(freq_emb)  # (batch, input_len, latent_dim)
        z = z + freq_emb  # (batch, input_len, latent_dim)
        
        # Apply Koopman evolution for each future timestep
        # Koopman operator: K = diag(magnitude * exp(i * phase))
        # For stability: ensure magnitude < 1
        magnitude = torch.sigmoid(self.koopman_magnitude)  # (latent_dim,)
        phase = self.koopman_phase  # (latent_dim,)
        
        # Complex eigenvalue: lambda = magnitude * (cos(phase) + i*sin(phase))
        # Real evolution: apply rotation with decay
        
        # Get initial state (last timestep)
        z0 = z[:, -1, :]  # (batch, latent_dim)
        
        # Evolve with Koopman operator for output_len steps
        z_future = []
        z_t = z0
        for t in range(self.output_len):
            # Apply Koopman operator (element-wise for diagonal structure)
            # Real part: rotate and decay
            cos_p = torch.cos(phase)  # (latent_dim,)
            sin_p = torch.sin(phase)  # (latent_dim,)
            
            # Simple real evolution: z_{t+1} = magnitude * (cos * z_t + sin * shift)
            # Use learned oscillation
            z_t_rot = magnitude * (cos_p * z_t + sin_p * torch.roll(z_t, 1, dims=-1))
            z_t = z_t_rot
            
            # Apply mode gating
            gate = self.mode_gate(z_t)  # (batch, n_koopman_modes)
            
            # Weighted combination of modes
            z_mode = torch.zeros_like(z_t)  # (batch, latent_dim)
            for m in range(self.n_koopman_modes):
                z_m = self.mode_proj[m](z_t)  # (batch, latent_dim)
                z_mode = z_mode + gate[:, m:m+1] * z_m  # (batch, latent_dim)
            
            z_future.append(z_mode)
        
        z_future = torch.stack(z_future, dim=1)  # (batch, output_len, latent_dim)
        
        # Decode
        out = self.decoder(z_future)  # (batch, output_len, n_features)
        
        # Add residual from input
        x_last = x[:, -1:, :]  # (batch, 1, n_features)
        out = out + x_last  # (batch, output_len, n_features)
        
        return out
