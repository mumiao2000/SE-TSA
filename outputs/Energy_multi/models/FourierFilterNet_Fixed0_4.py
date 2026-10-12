import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FourierFilterNet_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_freq_keep': [4, 16, 64],
            'filter_hidden': [16, 64, 256]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.n_freq_keep = min(self.config.get('n_freq_keep', 16), self.input_len // 2 + 1)
        self.filter_hidden = self.config.get('filter_hidden', 64)
        
        # Learnable frequency filter: MLP operating on frequency magnitudes
        self.freq_filter = nn.Sequential(
            nn.Linear(self.n_freq_keep, self.filter_hidden),
            nn.GELU(),
            nn.Linear(self.filter_hidden, self.n_freq_keep),
            nn.Sigmoid()  # Soft gating
        )
        
        # Frequency to time mapping for prediction
        # We predict output_len frequencies from input_len frequencies
        self.freq_predictor = nn.Linear(self.input_len, self.output_len)
        
        # Direct time-domain residual
        self.time_residual = nn.Linear(self.input_len, self.output_len)
        
        self.revin = RevIN(self.n_features, affine=True)

    def _dct(self, x):
        # x: (batch, seq_len, n_features)
        # Use DCT-II via FFT for efficiency
        X = torch.fft.rfft(x, dim=1, norm='ortho')  # (batch, n_freq, n_features)
        return X.real  # Use real part for simplicity

    def _idct(self, X, n):
        # X: (batch, n_freq, n_features), real
        # Reconstruct using irfft
        X_complex = torch.complex(X, torch.zeros_like(X))  # (batch, n_freq, n_features)
        x = torch.fft.irfft(X_complex, n=n, dim=1, norm='ortho')  # (batch, n, n_features)
        return x

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Frequency domain transform
        x_freq = self._dct(x)  # (batch, input_freq, n_features)
        x_freq = x_freq[:, :self.n_freq_keep, :]  # (batch, n_freq_keep, n_features)
        
        # Apply per-feature frequency filtering
        x_freq_perm = x_freq.permute(0, 2, 1)  # (batch, n_features, n_freq_keep)
        filter_weights = self.freq_filter(x_freq_perm)  # (batch, n_features, n_freq_keep)
        x_freq_filtered = x_freq_perm * filter_weights  # (batch, n_features, n_freq_keep)
        
        # Predict output frequencies: pad to input_len then project
        x_freq_padded = F.pad(x_freq_filtered, (0, self.input_len - self.n_freq_keep))  # (batch, n_features, input_len)
        x_freq_pred = self.freq_predictor(x_freq_padded)  # (batch, n_features, output_len)
        
        # Convert back to time domain
        x_freq_pred = x_freq_pred.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Build proper frequency representation for irfft
        # irfft expects (batch, n_freq, n_features) where n_freq = n//2 + 1
        n_output_freq = self.output_len // 2 + 1
        freq_full = torch.zeros(x.size(0), n_output_freq, self.n_features, device=x.device)  # (batch, n_output_freq, n_features)
        n_pred = min(self.n_freq_keep, n_output_freq)
        # x_freq_pred has shape (batch, output_len, n_features), but we only need first n_pred frequencies
        # The predictor outputs output_len values, we interpret first n_pred as frequency coefficients
        freq_full[:, :n_pred, :] = x_freq_pred[:, :n_pred, :]  # (batch, n_pred, n_features)
        
        time_pred = self._idct(freq_full, self.output_len)  # (batch, output_len, n_features)
        
        # Time-domain residual
        x_time = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        time_res = self.time_residual(x_time).permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = time_pred + time_res  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

