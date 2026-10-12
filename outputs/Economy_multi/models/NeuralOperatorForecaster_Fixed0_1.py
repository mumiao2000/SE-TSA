import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralOperatorForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_layers': [2, 4, 8],
            'n_modes': [8, 32, 128],
            'dropout': [0.0, 0.3],
            'lifting_ratio': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 256)
        self.n_layers = cfg.get('n_layers', 4)
        self.n_modes = cfg.get('n_modes', 32)
        self.dropout = cfg.get('dropout', 0.1)
        self.lifting_ratio = cfg.get('lifting_ratio', 4)
        
        self.d_lifting = self.d_model * self.lifting_ratio
        
        # Instance normalization for stability
        self.in_norm = nn.InstanceNorm1d(n_features, affine=True)
        
        # Lifting: map from n_features to d_lifting
        self.lifting = nn.Sequential(
            nn.Linear(n_features, self.d_lifting),
            nn.GELU(),
            nn.Linear(self.d_lifting, self.d_lifting)
        )
        
        # Positional encoding for temporal information
        self.pos_enc = self._create_positional_encoding(max(input_len, output_len), self.d_lifting)
        
        # FNO layers
        self.fno_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.fno_layers.append(self._make_fno_layer())
        
        # Projection: map from d_lifting to n_features
        self.projection = nn.Sequential(
            nn.Linear(self.d_lifting, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, n_features)
        )
        
        # Alternative: direct parameterization of future timesteps
        self.future_query = nn.Parameter(torch.randn(output_len, self.d_lifting))
        
    def _create_positional_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # (1, max_len, d_model)
    
    def _make_fno_layer(self):
        return nn.ModuleDict({
            'spectral_conv': nn.Linear(self.d_lifting, self.d_lifting),  # operates on d_lifting dimension
            'w': nn.Conv1d(self.d_lifting, self.d_lifting, 1),
            'norm': nn.LayerNorm(self.d_lifting),
            'dropout': nn.Dropout(self.dropout)
        })
    
    def _spectral_conv(self, x, layer):
        # x: (batch, seq_len, d_lifting)
        batch, seq_len, d = x.shape
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=1, n=seq_len)  # (batch, seq_len//2+1, d_lifting)
        
        # Get relevant modes
        n_ft = x_ft.shape[1]
        modes = min(self.n_modes, n_ft)
        
        # Truncate to modes
        x_ft_trunc = x_ft[:, :modes, :]  # (batch, modes, d_lifting)
        
        # Apply learned spectral weights (treat real/imag separately)
        x_ft_real = x_ft_trunc.real  # (batch, modes, d_lifting)
        x_ft_imag = x_ft_trunc.imag  # (batch, modes, d_lifting)
        
        # Stack real and imag along the modes dimension
        x_ft_cat = torch.cat([x_ft_real, x_ft_imag], dim=1)  # (batch, modes*2, d_lifting)
        
        # Apply linear transform across d_lifting features (last dimension)
        x_ft_out = layer['spectral_conv'](x_ft_cat)  # (batch, modes*2, d_lifting)
        
        # Split back into real and imag parts
        x_ft_real_out = x_ft_out[:, :modes, :]  # (batch, modes, d_lifting)
        x_ft_imag_out = x_ft_out[:, modes:, :]  # (batch, modes, d_lifting)
        
        # Reconstruct complex
        x_ft_out = torch.complex(x_ft_real_out, x_ft_imag_out)  # (batch, modes, d_lifting)
        
        # Pad back to full frequency representation
        out_ft = torch.zeros_like(x_ft)  # (batch, n_ft, d_lifting)
        out_ft[:, :modes, :] = x_ft_out
        
        # IFFT
        x_spec = torch.fft.irfft(out_ft, n=seq_len, dim=1)  # (batch, seq_len, d_lifting)
        
        # Skip connection in spatial domain
        x_w = layer['w'](x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, seq_len, d_lifting)
        
        return x_spec + x_w
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch, in_len, _ = x.shape
        
        # Normalize
        x = self.in_norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, input_len, n_features)
        
        # Lift to higher dimensional space
        x = self.lifting(x)  # (batch, input_len, d_lifting)
        
        # Add positional encoding
        x = x + self.pos_enc[:, :in_len, :]  # (batch, input_len, d_lifting)
        
        # Apply FNO layers
        for layer in self.fno_layers:
            x_in = x
            x = self._spectral_conv(x, layer)  # (batch, input_len, d_lifting)
            x = F.gelu(x)
            x = layer['norm'](x)
            x = layer['dropout'](x)
            x = x + x_in  # residual
        
        # Extend to output length using learned temporal queries
        # Use attention-based interpolation
        queries = self.future_query.unsqueeze(0).expand(batch, -1, -1)  # (batch, output_len, d_lifting)
        keys = x  # (batch, input_len, d_lifting)
        
        # Cross-attention from future queries to past keys
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.d_lifting ** 0.5)  # (batch, output_len, input_len)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, output_len, input_len)
        
        # Aggregate
        x_future = torch.bmm(attn_weights, keys)  # (batch, output_len, d_lifting)
        
        # Project to output features
        out = self.projection(x_future)  # (batch, output_len, n_features)
        
        return out
