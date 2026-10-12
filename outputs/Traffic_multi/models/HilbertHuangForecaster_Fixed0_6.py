import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HilbertHuangForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_imfs': [3, 5, 8],
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 2, 4],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.n_imfs = self.config.get('n_imfs', 5)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Learnable sifting: project to IMFs via attention-like mechanism
        self.imf_proj = nn.Linear(self.input_len, self.n_imfs * self.input_len)  # (batch, n_features, n_imfs * input_len)
        
        # Envelope extraction via max/mean pooling
        self.envelope_conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=7, padding=3, groups=self.n_features)
        
        # Hilbert-inspired feature extraction: real + imaginary (quadrature) features
        # Input is n_features * 2 (real and envelope concatenated along feature dim)
        self.hilbert_mlp = nn.Sequential(
            nn.Linear(self.n_features * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # LSTM for each IMF with residual connections
        self.imf_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=self.hidden_dim if i == 0 else self.hidden_dim // 2,
                hidden_size=self.hidden_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            ) for i in range(self.n_imfs)
        ])
        
        # Cross-IMF attention for recombination
        self.imf_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim // 2,
            num_heads=4,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_len)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for feature-wise operations: (batch, n_features, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Learnable IMF decomposition
        imf_flat = self.imf_proj(x_perm)  # (batch, n_features, n_imfs * input_len)
        imfs = imf_flat.view(x.size(0), self.n_features, self.n_imfs, self.input_len)  # (batch, n_features, n_imfs, input_len)
        
        # Envelope extraction for each IMF
        # Compute analytic signal approximation: original + quadrature (via diff)
        imfs_for_env = imfs.permute(0, 2, 1, 3).contiguous().view(x.size(0) * self.n_imfs, self.n_features, self.input_len)  # (batch*n_imfs, n_features, input_len)
        envelope = F.relu(self.envelope_conv(imfs_for_env))  # (batch*n_imfs, n_features, input_len)
        
        # Hilbert-inspired features: stack original and envelope as quadrature pair along feature dimension
        hilbert_input = torch.cat([
            imfs_for_env,  # real part: (batch*n_imfs, n_features, input_len)
            envelope  # envelope as imaginary approximation: (batch*n_imfs, n_features, input_len)
        ], dim=1)  # (batch*n_imfs, n_features * 2, input_len)
        
        # Permute to (batch*n_imfs, input_len, n_features * 2) for MLP on feature dimension
        hilbert_input = hilbert_input.permute(0, 2, 1)  # (batch*n_imfs, input_len, n_features * 2)
        
        hilbert_features = self.hilbert_mlp(hilbert_input)  # (batch*n_imfs, input_len, hidden_dim)
        hilbert_features = hilbert_features.permute(0, 2, 1)  # (batch*n_imfs, hidden_dim, input_len)
        
        # Reshape for LSTM processing per IMF
        lstm_input = hilbert_features.mean(dim=-1).view(x.size(0), self.n_imfs, self.hidden_dim)  # (batch, n_imfs, hidden_dim)
        lstm_input = lstm_input.permute(1, 0, 2)  # (n_imfs, batch, hidden_dim)
        
        # Process each IMF with LSTM
        imf_outputs = []
        for i, lstm in enumerate(self.imf_lstms):
            if i < lstm_input.size(0):
                inp = lstm_input[i]  # (batch, hidden_dim)
                # Expand to sequence for LSTM
                inp_seq = inp.unsqueeze(1).expand(-1, self.input_len, -1)  # (batch, input_len, hidden_dim)
                # Actually use the original sequence features
                if i == 0:
                    feat = hilbert_features.view(x.size(0), self.n_imfs, self.hidden_dim, -1)[:, 0, :, :].mean(dim=-1)  # (batch, hidden_dim)
                else:
                    feat = lstm_input[i]
                feat_seq = feat.unsqueeze(1).expand(-1, self.input_len, -1)  # (batch, input_len, hidden_dim//2 if i>0 else hidden_dim)
                if i > 0:
                    feat_seq = feat_seq[:, :, :self.hidden_dim // 2]
                out, _ = lstm(feat_seq)  # (batch, input_len, hidden_dim//2)
                imf_outputs.append(out[:, -1, :])  # (batch, hidden_dim//2)
        
        # Stack IMF representations
        imf_stack = torch.stack(imf_outputs, dim=1)  # (batch, n_imfs, hidden_dim//2)
        
        # Cross-IMF attention for adaptive recombination
        attn_out, _ = self.imf_attention(imf_stack, imf_stack, imf_stack)  # (batch, n_imfs, hidden_dim//2)
        
        # Aggregate across IMFs
        combined = attn_out.mean(dim=1)  # (batch, hidden_dim//2)
        
        # Final prediction
        out = self.output_proj(combined)  # (batch, output_len)
        out = out.unsqueeze(-1).expand(-1, -1, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

