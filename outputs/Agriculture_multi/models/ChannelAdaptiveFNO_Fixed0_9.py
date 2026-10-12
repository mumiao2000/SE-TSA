import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveFNO_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_modes': [8, 32, 128],
            'dropout': [0.0, 0.3],
            'decoder_dim': [64, 256, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_modes = min(self.config.get('n_modes', 32), self.d_model // 2 + 1)
        self.dropout = self.config.get('dropout', 0.1)
        self.decoder_dim = self.config.get('decoder_dim', 256)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Project to d_model per channel (channel-independent)
        self.input_proj = nn.Linear(self.input_len, self.d_model)
        
        # Complex-valued frequency weights (learnable) - shape: (n_features, n_modes)
        self.freq_weights = nn.Parameter(torch.randn(self.n_features, self.n_modes) * 0.02)
        
        # Channel-adaptive frequency gating
        # Input: per-channel mean of d_model dimension -> (batch, n_features)
        self.freq_gate = nn.Sequential(
            nn.Linear(self.n_features, self.n_features * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_features * 2, self.n_features * self.n_modes)
        )
        
        # Decoder: spectral -> time -> prediction
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model * 2, self.decoder_dim),  # *2 for real+imag after IFFT
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_dim, self.output_len)
        )
        
        self.residual_proj = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store residual
        residual = x  # (batch, input_len, n_features)
        
        # Permute for channel-wise processing: (batch, n_features, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project to d_model
        x_proj = self.input_proj(x_perm)  # (batch, n_features, d_model)
        
        # FFT along d_model dimension
        x_fft = torch.fft.rfft(x_proj, dim=-1)  # (batch, n_features, d_model//2+1)
        
        # Truncate to n_modes
        x_fft = x_fft[:, :, :self.n_modes]  # (batch, n_features, n_modes)
        
        # Compute feature context for gating: mean across d_model (before FFT truncation)
        feat_mean = x_proj.mean(dim=-1)  # (batch, n_features)
        
        # Compute frequency gates
        gates = self.freq_gate(feat_mean)  # (batch, n_features * n_modes)
        gates = gates.view(-1, self.n_features, self.n_modes)  # (batch, n_features, n_modes)
        gates = torch.sigmoid(gates)  # (batch, n_features, n_modes)
        
        # Apply channel-adaptive spectral weights with gating
        # Expand weights and gates for broadcasting with complex tensor
        weights = self.freq_weights.unsqueeze(0)  # (1, n_features, n_modes)
        gates_expanded = gates  # (batch, n_features, n_modes)
        
        # Apply weights and gates to FFT coefficients
        x_fft_weighted = x_fft * weights * gates_expanded  # (batch, n_features, n_modes)
        
        # Pad back to full FFT size for IFFT
        x_fft_full = torch.zeros(
            x_fft_weighted.size(0), self.n_features, x_fft.size(-1),
            dtype=x_fft_weighted.dtype, device=x_fft_weighted.device
        )  # (batch, n_features, d_model//2+1)
        x_fft_full[:, :, :self.n_modes] = x_fft_weighted  # (batch, n_features, d_model//2+1)
        
        # IFFT back to time domain
        x_time = torch.fft.irfft(x_fft_full, n=self.d_model, dim=-1)  # (batch, n_features, d_model)
        
        # Also keep the real and imag parts of truncated FFT as features
        x_fft_real = x_fft_weighted.real  # (batch, n_features, n_modes)
        x_fft_imag = x_fft_weighted.imag  # (batch, n_features, n_modes)
        
        # Pad real/imag to d_model for concatenation
        x_fft_real_pad = torch.zeros(
            x_fft_real.size(0), self.n_features, self.d_model,
            dtype=x_fft_real.dtype, device=x_fft_real.device
        )  # (batch, n_features, d_model)
        x_fft_imag_pad = torch.zeros(
            x_fft_imag.size(0), self.n_features, self.d_model,
            dtype=x_fft_imag.dtype, device=x_fft_imag.device
        )  # (batch, n_features, d_model)
        x_fft_real_pad[:, :, :self.n_modes] = x_fft_real  # (batch, n_features, d_model)
        x_fft_imag_pad[:, :, :self.n_modes] = x_fft_imag  # (batch, n_features, d_model)
        
        # Combine: time domain + spectral features
        x_processed = torch.cat([
            x_time,  # (batch, n_features, d_model)
            x_fft_real_pad + x_fft_imag_pad  # (batch, n_features, d_model)
        ], dim=-1)  # (batch, n_features, d_model * 2)
        
        # Decode
        out = self.decoder(x_processed)  # (batch, n_features, output_len)
        
        # Residual connection
        res_out = self.residual_proj(residual.permute(0, 2, 1))  # (batch, n_features, output_len)
        out = out + res_out  # (batch, n_features, output_len)
        
        # Permute back
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

