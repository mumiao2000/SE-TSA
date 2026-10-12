import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralCDESpectralAttention(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'n_spectral_modes': [8, 32, 128],
            'cde_width': [32, 128, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.n_spectral_modes = self.config.get('n_spectral_modes', 32)
        self.cde_width = self.config.get('cde_width', 128)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to CDE width
        self.input_proj = nn.Linear(self.n_features, self.cde_width)  # (batch, input_len, cde_width)
        
        # CDE vector field: defines dynamics dz/dt = f(z, t)
        # We use a simple MLP as the vector field
        cde_vf_layers = []
        cde_vf_layers.append(nn.Linear(self.cde_width + 1, self.hidden_dim))  # +1 for time
        for _ in range(max(0, self.n_layers - 2)):
            cde_vf_layers.append(nn.ReLU())
            cde_vf_layers.append(nn.Dropout(self.dropout_rate))
            cde_vf_layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        cde_vf_layers.append(nn.ReLU())
        cde_vf_layers.append(nn.Linear(self.hidden_dim, self.cde_width * self.cde_width))  # matrix output
        self.cde_vector_field = nn.Sequential(*cde_vf_layers)
        
        # Spectral attention mechanism
        # Project hidden state to query, use input as key/value
        self.query_proj = nn.Linear(self.cde_width, self.hidden_dim)  # (batch, seq, hidden_dim)
        self.key_proj = nn.Linear(self.cde_width, self.hidden_dim)    # (batch, seq, hidden_dim)
        self.value_proj = nn.Linear(self.cde_width, self.hidden_dim)  # (batch, seq, hidden_dim)
        
        # Spectral gating in Fourier space
        self.spectral_gate = nn.Parameter(torch.randn(self.hidden_dim, self.n_spectral_modes, 2) * 0.02)
        
        # Output projections
        self.attn_out_proj = nn.Linear(self.hidden_dim, self.cde_width)  # (batch, seq, cde_width)
        self.output_proj = nn.Linear(self.cde_width, self.n_features)    # (batch, seq, n_features)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(self.cde_width)
        
    def cde_solve(self, x, times):
        # x: (batch, input_len, cde_width)
        # times: (input_len,)
        batch_size = x.shape[0]
        
        # Initialize hidden state as first observation
        z = x[:, 0, :]  # (batch, cde_width)
        
        # Simple Euler integration for CDE
        dt = 1.0 / self.input_len  # normalized time step
        hidden_states = [z]
        
        for i in range(1, self.input_len):
            t = times[i-1].unsqueeze(0).expand(batch_size, 1)  # (batch, 1)
            z_t = torch.cat([z, t], dim=-1)  # (batch, cde_width + 1)
            
            # Compute vector field
            vf_flat = self.cde_vector_field(z_t)  # (batch, cde_width * cde_width)
            vf = vf_flat.view(batch_size, self.cde_width, self.cde_width)  # (batch, cde_width, cde_width)
            
            # Approximate control derivative from data
            dx = x[:, i, :] - x[:, i-1, :]  # (batch, cde_width)
            dx = dx.unsqueeze(-1)  # (batch, cde_width, 1)
            
            # CDE update: dz = f(z, t) @ dx
            dz = torch.bmm(vf, dx).squeeze(-1)  # (batch, cde_width)
            z = z + dz  # (batch, cde_width)
            hidden_states.append(z)
        
        hidden_states = torch.stack(hidden_states, dim=1)  # (batch, input_len, cde_width)
        return hidden_states
    
    def spectral_attention(self, query, key, value):
        # query: (batch, output_len, hidden_dim) - from CDE hidden state
        # key: (batch, input_len, hidden_dim) - from input
        # value: (batch, input_len, hidden_dim) - from input
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        
        # Compute attention scores
        scores = torch.bmm(query, key.transpose(1, 2))  # (batch, output_len, input_len)
        scores = scores / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, output_len, input_len)
        
        # Apply attention
        out = torch.bmm(attn_weights, value)  # (batch, output_len, hidden_dim)
        
        # Spectral gating: FFT -> gate -> IFFT
        # Process each feature dimension independently
        out_fft = torch.fft.rfft(out, dim=1, n=max(seq_len_q, self.n_spectral_modes * 2))  # (batch, freq_len, hidden_dim)
        
        # Truncate/pad to n_spectral_modes
        freq_len = out_fft.shape[1]
        if freq_len > self.n_spectral_modes:
            out_fft = out_fft[:, :self.n_spectral_modes, :]  # (batch, n_spectral_modes, hidden_dim)
        
        # Apply learnable complex gate
        gate_real = self.spectral_gate[:, :, 0].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim, n_spectral_modes)
        gate_imag = self.spectral_gate[:, :, 1].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim, n_spectral_modes)
        
        # Transpose for broadcasting: out_fft is (batch, n_spectral_modes, hidden_dim)
        gate = torch.view_as_complex(torch.stack([
            gate_real.squeeze(0).squeeze(0).transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1),  # (batch, n_spectral_modes, hidden_dim)
            gate_imag.squeeze(0).squeeze(0).transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1)   # (batch, n_spectral_modes, hidden_dim)
        ], dim=-1))  # (batch, n_spectral_modes, hidden_dim)
        
        out_fft_gated = out_fft[:, :self.n_spectral_modes, :] * gate  # (batch, n_spectral_modes, hidden_dim)
        
        # Pad back and inverse FFT
        if freq_len > self.n_spectral_modes:
            padding = torch.zeros(batch_size, freq_len - self.n_spectral_modes, self.hidden_dim, 
                                 dtype=out_fft_gated.dtype, device=out_fft_gated.device)  # (batch, pad_len, hidden_dim)
            out_fft_full = torch.cat([out_fft_gated, padding], dim=1)  # (batch, freq_len, hidden_dim)
        else:
            out_fft_full = out_fft_gated
        
        out_spectral = torch.fft.irfft(out_fft_full, dim=1, n=seq_len_q)  # (batch, output_len, hidden_dim)
        
        # Residual connection
        out = out + 0.1 * out_spectral  # (batch, output_len, hidden_dim)
        
        return out, attn_weights
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project input
        x_proj = self.input_proj(x)  # (batch, input_len, cde_width)
        x_proj = self.dropout(x_proj)  # (batch, input_len, cde_width)
        
        # Create normalized time points
        times = torch.linspace(0, 1, self.input_len, device=x.device)  # (input_len,)
        
        # Solve CDE to get continuous hidden trajectory
        cde_hidden = self.cde_solve(x_proj, times)  # (batch, input_len, cde_width)
        cde_hidden = self.layer_norm(cde_hidden)  # (batch, input_len, cde_width)
        
        # Prepare for forecasting: use last hidden state to attend over full history
        # Query from final CDE state, repeated for each output step with learned positional variation
        query_base = cde_hidden[:, -1:, :]  # (batch, 1, cde_width)
        
        # Create output queries with slight perturbations for diversity across horizon
        output_queries = []
        for t in range(self.output_len):
            t_factor = t / max(1, self.output_len - 1)
            query_t = query_base * (1 + 0.1 * torch.sin(torch.tensor(t_factor * 3.14159)))  # (batch, 1, cde_width)
            output_queries.append(query_t)
        queries = torch.cat(output_queries, dim=1)  # (batch, output_len, cde_width)
        
        # Project to attention dimensions
        Q = self.query_proj(queries)  # (batch, output_len, hidden_dim)
        K = self.key_proj(cde_hidden)  # (batch, input_len, hidden_dim)
        V = self.value_proj(cde_hidden)  # (batch, input_len, hidden_dim)
        
        Q = self.dropout(Q)  # (batch, output_len, hidden_dim)
        K = self.dropout(K)  # (batch, input_len, hidden_dim)
        V = self.dropout(V)  # (batch, input_len, hidden_dim)
        
        # Spectral attention
        attn_out, _ = self.spectral_attention(Q, K, V)  # (batch, output_len, hidden_dim)
        attn_out = self.dropout(attn_out)  # (batch, output_len, hidden_dim)
        
        # Project back
        out = self.attn_out_proj(attn_out)  # (batch, output_len, cde_width)
        out = self.layer_norm(out)  # (batch, output_len, cde_width)
        
        # Final projection to features
        out = self.output_proj(out)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
