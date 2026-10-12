import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralCDEFFrequency(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'n_modes': [16, 64, 256],
            'attention_heads': [2, 8, 32],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_modes = self.config.get('n_modes', 64)
        self.attention_heads = self.config.get('attention_heads', 8)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to hidden dimension
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Fourier frequency encoding for path representation
        self.freq_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.freq_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            ) for _ in range(self.n_layers)
        ])
        
        # CDE vector field: defines how hidden state evolves
        self.cde_fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim + self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            ) for _ in range(self.n_layers)
        ])
        
        # Fourier spectral mixing for continuous evolution
        self.spectral_mix = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ) for _ in range(self.n_layers)
        ])
        
        # Gated cross-attention for reading from hidden state
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, 
            num_heads=self.attention_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid(),
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.n_features),
        )
        
        # Learnable temporal embeddings for query positions
        self.temporal_embed = nn.Embedding(self.output_len, self.hidden_dim)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Project input to hidden dimension
        h = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Fourier frequency encoding of the path
        # Compute FFT-based features
        h_freq = torch.fft.rfft(h, dim=1, n=self.n_modes * 2)  # (batch, n_modes+1, hidden_dim)
        h_freq = h_freq[:, :self.n_modes, :]  # Truncate to n_modes  # (batch, n_modes, hidden_dim)
        h_freq_real = torch.view_as_real(h_freq).mean(dim=-1)  # (batch, n_modes, hidden_dim)
        
        # Interpolate frequency features back to sequence length
        h_freq = F.interpolate(
            h_freq_real.permute(0, 2, 1),  # (batch, hidden_dim, n_modes)
            size=self.input_len,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # (batch, input_len, hidden_dim)
        
        h = self.freq_proj(h) + h_freq  # (batch, input_len, hidden_dim)
        
        # Neural CDE evolution through layers
        for layer in range(self.n_layers):
            # Frequency encoding update
            h_freq_enc = self.freq_encoder[layer](h)  # (batch, input_len, hidden_dim)
            
            # CDE vector field: combine state and control
            h_combined = torch.cat([h, h_freq_enc], dim=-1)  # (batch, input_len, hidden_dim*2)
            dh = self.cde_fcs[layer](h_combined)  # (batch, input_len, hidden_dim)
            
            # Spectral mixing for smooth evolution
            h = h + dh  # Euler step for CDE  # (batch, input_len, hidden_dim)
            h = self.spectral_mix[layer](h)  # (batch, input_len, hidden_dim)
            h = F.layer_norm(h, h.shape[-1:])  # (batch, input_len, hidden_dim)
        
        # Gated cross-attention decoding
        # Create query positions for output horizon
        query_pos = torch.arange(self.output_len, device=x.device)  # (output_len,)
        queries = self.temporal_embed(query_pos)  # (output_len, hidden_dim)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, output_len, hidden_dim)
        
        # Project for attention
        Q = self.query_proj(queries)  # (batch, output_len, hidden_dim)
        K = self.key_proj(h)  # (batch, input_len, hidden_dim)
        V = self.value_proj(h)  # (batch, input_len, hidden_dim)
        
        # Cross-attention with gating
        attn_out, _ = self.attention(Q, K, V)  # (batch, output_len, hidden_dim)
        
        # Gating mechanism
        gate_input = torch.cat([queries, attn_out], dim=-1)  # (batch, output_len, hidden_dim*2)
        gate = self.gate_proj(gate_input)  # (batch, output_len, hidden_dim)
        out = gate * attn_out + (1 - gate) * queries  # (batch, output_len, hidden_dim)
        
        # Output projection
        out = self.output_proj(out)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
