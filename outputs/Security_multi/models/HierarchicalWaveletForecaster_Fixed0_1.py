import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HierarchicalWaveletForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 512],
            'n_levels': [1, 2, 4],
            'dropout': [0.0, 0.3],
            'kernel_size': [3, 5, 7]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_levels = self.config.get('n_levels', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.kernel_size = self.config.get('kernel_size', 3)
        
        # Instance normalization
        self.norm = nn.InstanceNorm1d(self.n_features, affine=True)
        
        # Ensure input_len is power of 2 for wavelet
        self.padded_len = 2 ** (self.input_len - 1).bit_length()
        self.pad_front = (self.padded_len - self.input_len) // 2
        self.pad_back = self.padded_len - self.input_len - self.pad_front
        
        # Wavelet decomposition filters (Haar)
        self.register_buffer('low_pass', torch.tensor([0.5, 0.5]))
        self.register_buffer('high_pass', torch.tensor([0.5, -0.5]))
        
        # Processors for each wavelet level
        self.approx_processors = nn.ModuleList()
        self.detail_processors = nn.ModuleList()
        
        current_len = self.padded_len
        for level in range(self.n_levels):
            current_len = current_len // 2
            self.approx_processors.append(
                nn.Sequential(
                    nn.Conv1d(self.n_features, self.hidden_dim, self.kernel_size, padding=self.kernel_size//2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Conv1d(self.hidden_dim, self.n_features, self.kernel_size, padding=self.kernel_size//2)
                )
            )
            self.detail_processors.append(
                nn.Sequential(
                    nn.Conv1d(self.n_features, self.hidden_dim, self.kernel_size, padding=self.kernel_size//2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Conv1d(self.hidden_dim, self.n_features, self.kernel_size, padding=self.kernel_size//2)
                )
            )
        
        # Final approximation processor
        self.final_approx_processor = nn.Sequential(
            nn.Conv1d(self.n_features, self.hidden_dim, self.kernel_size, padding=self.kernel_size//2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.hidden_dim, self.n_features, self.kernel_size, padding=self.kernel_size//2)
        )
        
        # Feature mixer: combine concatenated wavelet levels back to n_features
        self.feature_mixer = nn.Conv1d(self.n_features * (self.n_levels + 1), self.n_features, 1)
        
        # Output projection: from wavelet domain to time domain forecast
        self.output_proj = nn.Linear(self.padded_len, self.output_len)
    
    def haar_transform(self, x):
        # x: (batch, n_features, seq_len)
        batch_size, n_features, seq_len = x.shape
        
        # Pad if necessary
        if seq_len % 2 == 1:
            x = F.pad(x, (0, 1))
            seq_len += 1
        
        # Reshape for pairwise operations
        x = x.view(batch_size, n_features, seq_len // 2, 2)  # (batch, n_features, seq_len/2, 2)
        
        # Apply Haar filters
        approx = x[..., 0] * 0.5 + x[..., 1] * 0.5  # (batch, n_features, seq_len/2)
        detail = x[..., 0] * 0.5 - x[..., 1] * 0.5  # (batch, n_features, seq_len/2)
        
        return approx, detail
    
    def inverse_haar(self, approx, detail):
        # approx, detail: (batch, n_features, seq_len)
        batch_size, n_features, seq_len = approx.shape
        
        # Reconstruct
        x0 = approx + detail  # (batch, n_features, seq_len)
        x1 = approx - detail  # (batch, n_features, seq_len)
        
        # Interleave
        x = torch.stack([x0, x1], dim=-1)  # (batch, n_features, seq_len, 2)
        x = x.view(batch_size, n_features, seq_len * 2)  # (batch, n_features, seq_len*2)
        
        return x
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Instance normalization
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.norm(x)  # (batch, n_features, input_len)
        
        # Pad to power of 2
        x = F.pad(x, (self.pad_front, self.pad_back))  # (batch, n_features, padded_len)
        
        # Multi-level wavelet decomposition
        coeffs = []  # Store (approx, detail) at each level
        current = x
        
        for level in range(self.n_levels):
            approx, detail = self.haar_transform(current)  # both (batch, n_features, current_len/2)
            coeffs.append((approx, detail))
            current = approx
        
        # Process each level
        processed_details = []
        for level, (approx, detail) in enumerate(coeffs):
            # Process detail coefficients
            p_detail = self.detail_processors[level](detail)  # (batch, n_features, len_at_level)
            processed_details.append(p_detail)
        
        # Process final approximation
        final_approx = self.final_approx_processor(current)  # (batch, n_features, final_len)
        
        # Collect all processed coefficients
        all_coeffs = [final_approx] + processed_details  # list of (batch, n_features, various_lens)
        
        # Upsample all to same length and concatenate
        max_len = self.padded_len
        upsampled = []
        for coeff in all_coeffs:
            coeff_up = F.interpolate(coeff, size=max_len, mode='linear', align_corners=False)  # (batch, n_features, max_len)
            upsampled.append(coeff_up)
        
        # Concatenate along feature dimension
        combined = torch.cat(upsampled, dim=1)  # (batch, n_features*(n_levels+1), max_len)
        
        # Project back to n_features using proper 1x1 convolution
        combined = self.feature_mixer(combined)  # (batch, n_features, max_len)
        
        # Trim padding and project to output length
        result = combined[..., self.pad_front:self.pad_front + self.input_len]  # (batch, n_features, input_len)
        
        # Temporal projection to output
        result = self.output_proj(result)  # (batch, n_features, output_len)
        result = result.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        return result

