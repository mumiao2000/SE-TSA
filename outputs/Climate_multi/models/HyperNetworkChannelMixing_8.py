import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperNetworkChannelMixing(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hyper_dim': [32, 128, 512],
            'embed_dim': [16, 64, 256],
            'n_heads': [2, 8, 32],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hyper_dim = self.config.get('hyper_dim', 128)
        self.embed_dim = self.config.get('embed_dim', 64)
        self.n_heads = self.config.get('n_heads', 8)
        self.dropout_rate = self.config.get('dropout', 0.0)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Global statistics encoder for hypernetwork input
        self.stat_encoder = nn.Sequential(
            nn.Linear(self.n_features * 2, self.embed_dim),  # mean and std per feature
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Cross-channel attention for global context
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.n_heads, 
                                                 dropout=self.dropout_rate, batch_first=True)
        
        # Hypernetwork: generates weights for channel-specific network
        # For each feature, generate a small MLP: input_len -> output_len
        self.weight_gen = nn.Sequential(
            nn.Linear(self.embed_dim, self.hyper_dim),
            nn.LayerNorm(self.hyper_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hyper_dim, self.hyper_dim),
            nn.LayerNorm(self.hyper_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Output heads for weight generation
        # Each feature needs: W1 (input_len, hidden), b1 (hidden), W2 (hidden, output_len), b2 (output_len)
        self.hidden_per_channel = max(4, self.input_len // 2)
        self.weight_proj = nn.Linear(self.hyper_dim, 
                                      self.n_features * (self.input_len * self.hidden_per_channel + 
                                                        self.hidden_per_channel +
                                                        self.hidden_per_channel * self.output_len +
                                                        self.output_len))
        
        # Feature interaction: low-rank cross-channel mixing
        self.channel_mix = nn.Sequential(
            nn.Linear(self.n_features, self.n_features // 4),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.n_features // 4, self.n_features)
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Learnable channel embeddings for hypernetwork conditioning
        self.channel_embed = nn.Parameter(torch.randn(self.n_features, self.embed_dim) * 0.02)
    
    def generate_weights(self, stats):
        # stats: (batch, n_features, 2) - mean and std
        batch = stats.size(0)
        
        # Encode statistics
        stats_flat = stats.view(batch, self.n_features, 2)  # (batch, n_features, 2)
        # Pad to expected input
        stats_padded = F.pad(stats_flat, (0, self.n_features * 2 - 2))  # wrong, fix
        # Actually reshape properly
        stats_input = stats_flat.view(batch, self.n_features, -1)  # (batch, n_features, 2)
        # Need to match expected input dim
        
        # Simple approach: just use mean and std as features
        stats_proj = torch.cat([stats_input, stats_input], dim=-1)  # (batch, n_features, 4)
        # Pad or project to right size
        if stats_proj.size(-1) < self.n_features * 2:
            stats_proj = F.pad(stats_proj, (0, self.n_features * 2 - stats_proj.size(-1)))
        else:
            stats_proj = stats_proj[..., :self.n_features * 2]
        
        # Actually let's be cleaner: compute proper statistics
        # stats should be (batch, n_features, 2) from input
        # We need (batch, n_features, embed_dim) for attention
        
        # Encode per-feature stats
        h = self.stat_encoder(stats_proj.view(batch * self.n_features, -1))  # (batch * n_features, embed_dim)
        h = h.view(batch, self.n_features, self.embed_dim)  # (batch, n_features, embed_dim)
        
        # Add channel embeddings
        h = h + self.channel_embed.unsqueeze(0)  # (batch, n_features, embed_dim)
        
        # Cross-channel attention
        h_attn, _ = self.cross_attn(h, h, h)  # (batch, n_features, embed_dim)
        h = h + self.dropout(h_attn)  # residual
        h = F.layer_norm(h, h.shape[-1:])  # (batch, n_features, embed_dim)
        
        # Generate weights per channel
        w_hyper = self.weight_gen(h)  # (batch, n_features, hyper_dim)
        
        # Project to actual weight values
        weights_flat = self.weight_proj(w_hyper)  # (batch, n_features, total_weight_size)
        
        # Parse weights
        idx = 0
        w_size1 = self.input_len * self.hidden_per_channel
        b_size1 = self.hidden_per_channel
        w_size2 = self.hidden_per_channel * self.output_len
        b_size2 = self.output_len
        total = w_size1 + b_size1 + w_size2 + b_size2
        
        W1 = weights_flat[:, :, idx:idx+w_size1].view(batch, self.n_features, self.input_len, self.hidden_per_channel)
        idx += w_size1
        b1 = weights_flat[:, :, idx:idx+b_size1].view(batch, self.n_features, self.hidden_per_channel)
        idx += b_size1
        W2 = weights_flat[:, :, idx:idx+w_size2].view(batch, self.n_features, self.hidden_per_channel, self.output_len)
        idx += w_size2
        b2 = weights_flat[:, :, idx:idx+b_size2].view(batch, self.n_features, self.output_len)
        
        return W1, b1, W2, b2
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch = x.size(0)
        
        # Compute per-feature statistics for hypernetwork conditioning
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        x_std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        stats = torch.cat([x_mean, x_std], dim=1).permute(0, 2, 1)  # (batch, n_features, 2)
        
        # Generate channel-specific weights
        W1, b1, W2, b2 = self.generate_weights(stats)  # each has batch dimension
        
        # Apply channel-specific MLPs
        # x: (batch, input_len, n_features)
        x_permuted = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # For each feature, apply its generated MLP
        outputs = []
        for f in range(self.n_features):
            xf = x_permuted[:, f, :]  # (batch, input_len)
            # MLP: input_len -> hidden -> output_len
            h = torch.matmul(xf.unsqueeze(1), W1[:, f, :, :]).squeeze(1) + b1[:, f, :]  # (batch, hidden)
            h = F.gelu(h)
            h = self.dropout(h)
            out_f = torch.matmul(h.unsqueeze(1), W2[:, f, :, :]).squeeze(1) + b2[:, f, :]  # (batch, output_len)
            outputs.append(out_f)
        
        out = torch.stack(outputs, dim=2)  # (batch, output_len, n_features)
        
        # Cross-channel mixing
        out = out + self.channel_mix(out)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

