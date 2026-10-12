import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelKoopmanMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_koopman_blocks': [1, 4, 8],
            'latent_dim': [64, 256, 1024],
            'gate_dim': [16, 32, 64],
            'dropout': [0.0, 0.3],
            'n_temporal_layers': [1, 2, 4],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_koopman_blocks = self.config.get('n_koopman_blocks', 4)
        self.latent_dim = self.config.get('latent_dim', 256)
        self.gate_dim = self.config.get('gate_dim', 32)
        self.dropout = self.config.get('dropout', 0.0)
        self.n_temporal_layers = self.config.get('n_temporal_layers', 2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Hierarchical temporal decomposition: extract trend and seasonal components
        # Trend: moving average with kernel size matching dominant period (~12)
        self.trend_kernel = max(3, min(12, input_len // 4))
        self.trend_pad = self.trend_kernel // 2
        
        # Temporal encoding MLP for local patterns
        self.temporal_encoder = nn.ModuleList()
        for i in range(self.n_temporal_layers):
            in_d = self.n_features * 2 if i == 0 else self.hidden_dim  # *2 for trend+seasonal
            out_d = self.hidden_dim
            self.temporal_encoder.append(nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.LayerNorm(out_d),
            ))
        
        # Cross-channel attention with lightweight gating
        self.channel_query = nn.Linear(self.hidden_dim, self.gate_dim)
        self.channel_key = nn.Linear(self.hidden_dim, self.gate_dim)
        self.channel_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.channel_scale = self.gate_dim ** -0.5
        self.channel_out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.channel_norm = nn.LayerNorm(self.hidden_dim)
        
        # Koopman operator: encoder to latent space
        self.koopman_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        # Learnable Koopman operator matrix (linear dynamics in latent space)
        self.koopman_K = nn.Parameter(torch.randn(self.latent_dim, self.latent_dim) * 0.01)
        self.koopman_bias = nn.Parameter(torch.zeros(self.latent_dim))
        
        # Koopman decoder back to feature space
        self.koopman_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_dim, self.hidden_dim),
        )
        
        # Output projection: map from hidden space to output predictions
        # We need to predict output_len steps from input_len steps
        self.temporal_aggregator = nn.Linear(input_len, output_len)
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.n_features),
        )
        
        self.residual_proj = nn.Linear(self.n_features, self.n_features)
        
    def extract_trend_seasonal(self, x):
        # x: (batch, input_len, n_features)
        # Trend: simple moving average
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        trend = F.avg_pool1d(
            F.pad(x_perm, (self.trend_pad, self.trend_pad), mode='reflect'),
            kernel_size=self.trend_kernel,
            stride=1,
            padding=0
        )  # (batch, n_features, input_len)
        trend = trend.permute(0, 2, 1)  # (batch, input_len, n_features)
        seasonal = x - trend  # (batch, input_len, n_features)
        return trend, seasonal  # both (batch, input_len, n_features)
    
    def apply_koopman_dynamics(self, z, n_steps):
        # z: (batch, latent_dim) - initial latent state
        # n_steps: number of steps to predict
        # Returns: (batch, n_steps, latent_dim)
        batch_size = z.shape[0]
        states = []
        current_z = z
        K = self.koopman_K + torch.eye(self.latent_dim, device=z.device) * 0.9  # stable dynamics
        for _ in range(n_steps):
            current_z = torch.matmul(current_z, K.t()) + self.koopman_bias  # (batch, latent_dim)
            states.append(current_z)
        return torch.stack(states, dim=1)  # (batch, n_steps, latent_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Hierarchical decomposition
        trend, seasonal = self.extract_trend_seasonal(x)  # (batch, input_len, n_features) each
        x_decomp = torch.cat([trend, seasonal], dim=-1)  # (batch, input_len, n_features * 2)
        
        # Temporal encoding
        h = x_decomp  # (batch, input_len, n_features * 2)
        for layer in self.temporal_encoder:
            h = layer(h) + h if h.shape[-1] == layer[0].out_features else layer(h)  # (batch, input_len, hidden_dim)
        
        # Cross-channel attention with lightweight gating
        # Transpose for channel-wise attention: treat time as batch, channels as sequence
        h_t = h.permute(1, 0, 2)  # (input_len, batch, hidden_dim)
        
        # Channel attention: each time point attends across channels
        # We reshape to (batch * input_len, hidden_dim) for parallel processing
        h_flat = h.reshape(-1, self.hidden_dim)  # (batch * input_len, hidden_dim)
        
        q = self.channel_query(h_flat)  # (batch * input_len, gate_dim)
        k = self.channel_key(h_flat)    # (batch * input_len, gate_dim)
        v = self.channel_value(h_flat)  # (batch * input_len, hidden_dim)
        
        # Compute attention scores: reshape back for proper attention
        q = q.view(-1, self.input_len, self.gate_dim)  # (batch, input_len, gate_dim)
        k = k.view(-1, self.input_len, self.gate_dim)  # (batch, input_len, gate_dim)
        v = v.view(-1, self.input_len, self.hidden_dim)  # (batch, input_len, hidden_dim)
        
        # Attention across time dimension (temporal self-attention with channel-gating)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.channel_scale  # (batch, input_len, input_len)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, input_len, input_len)
        attn_out = torch.matmul(attn_weights, v)  # (batch, input_len, hidden_dim)
        
        # Residual connection and projection
        attn_out = self.channel_out_proj(attn_out)  # (batch, input_len, hidden_dim)
        h = self.channel_norm(h + attn_out)  # (batch, input_len, hidden_dim)
        
        # Koopman operator dynamics: encode to latent space
        # Aggregate temporal information for initial state
        h_pooled = h.mean(dim=1)  # (batch, hidden_dim)
        z0 = self.koopman_encoder(h_pooled)  # (batch, latent_dim)
        
        # Apply Koopman dynamics for prediction steps
        z_trajectory = self.apply_koopman_dynamics(z0, self.output_len)  # (batch, output_len, latent_dim)
        
        # Decode Koopman predictions
        h_pred = self.koopman_decoder(z_trajectory)  # (batch, output_len, hidden_dim)
        
        # Also use direct temporal aggregation as complementary path
        h_temp = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        h_temp = self.temporal_aggregator(h_temp)  # (batch, hidden_dim, output_len)
        h_temp = h_temp.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Combine Koopman and direct temporal paths
        h_combined = h_pred + h_temp  # (batch, output_len, hidden_dim)
        
        # Output projection
        out = self.output_proj(h_combined)  # (batch, output_len, n_features)
        
        # Residual connection from input mean (simple baseline)
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        residual = self.residual_proj(x_mean).expand(-1, self.output_len, -1)  # (batch, output_len, n_features)
        out = out + residual  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)
