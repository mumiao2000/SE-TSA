import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperNetworkAdaptiveForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hyper_dim': [32, 128, 512],
            'n_hidden_layers': [1, 2, 4],
            'hidden_width': [16, 64, 256],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hyper_dim = self.config.get('hyper_dim', 128)
        self.n_hidden_layers = self.config.get('n_hidden_layers', 2)
        self.hidden_width = self.config.get('hidden_width', 64)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Hypernetwork: processes input to generate weights
        # Input statistics encoder
        self.stat_encoder = nn.Sequential(
            nn.Linear(self.input_len * 2, self.hyper_dim),  # mean and std per feature
            nn.GELU(),
            nn.Linear(self.hyper_dim, self.hyper_dim),
            nn.GELU()
        )
        
        # Compute weight counts for main network
        # Main network: input_len -> hidden_width -> ... -> hidden_width -> output_len
        self.weight_shapes = []
        self.bias_shapes = []
        
        layer_dims = [self.input_len] + [self.hidden_width] * self.n_hidden_layers + [self.output_len]
        for i in range(len(layer_dims) - 1):
            self.weight_shapes.append((layer_dims[i+1], layer_dims[i]))
            self.bias_shapes.append(layer_dims[i+1])
        
        total_weights = sum(w[0] * w[1] for w in self.weight_shapes)
        total_biases = sum(b for b in self.bias_shapes)
        
        # Hypernetwork outputs all weights and biases
        self.hyper_head = nn.Sequential(
            nn.Linear(self.hyper_dim, self.hyper_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hyper_dim * 2, total_weights + total_biases)
        )
        
        # Feature-specific hypernetwork modulation
        self.feature_hyper = nn.Linear(self.hyper_dim, self.n_features * self.hyper_dim // 4)
        
    def generate_weights(self, x_stats):
        # x_stats: (batch, input_len * 2) - concatenated mean and std
        hyper_feat = self.stat_encoder(x_stats)  # (batch, hyper_dim)
        
        # Generate all parameters
        params = self.hyper_head(hyper_feat)  # (batch, total_params)
        
        # Split into weights and biases
        weights = []
        biases = []
        idx = 0
        for w_shape, b_shape in zip(self.weight_shapes, self.bias_shapes):
            w_size = w_shape[0] * w_shape[1]
            w = params[:, idx:idx+w_size].view(-1, w_shape[0], w_shape[1])  # (batch, out, in)
            weights.append(w)
            idx += w_size
            
            b = params[:, idx:idx+b_shape]  # (batch, out)
            biases.append(b)
            idx += b_shape
            
        return weights, biases, hyper_feat
        
    def apply_main_network(self, x, weights, biases):
        # x: (batch, n_features, input_len)
        # Process each feature with its generated network
        
        out = x  # (batch, n_features, input_len)
        
        for i, (w, b) in enumerate(zip(weights, biases)):
            # w: (batch, out_dim, in_dim)
            # b: (batch, out_dim)
            # out: (batch, n_features, in_dim)
            
            # Batch matrix multiply: (batch, n_features, in) @ (batch, in, out) -> need to use w properly
            # Actually w is (batch, out, in), so we need (batch, n_features, in) -> (batch, in, n_features) not right
            
            # Correct approach: for each batch element, apply its specific weights
            # out: (batch, n_features, in_dim)
            # w: (batch, out_dim, in_dim)
            
            # Use einsum for batch-specific weights
            # We want: out[b, f, o] = sum_i out[b, f, i] * w[b, o, i] + b[b, o]
            # But this mixes features, we want channel-independent
            
            # Actually, let's make weights shared across features but adapted per batch
            # Reshape out to (batch * n_features, in_dim)
            batch_size = out.size(0)
            out_flat = out.reshape(batch_size * self.n_features, -1)  # (batch*n_features, in_dim)
            
            # Repeat weights for each feature
            w_rep = w.repeat_interleave(self.n_features, dim=0)  # (batch*n_features, out_dim, in_dim)
            b_rep = b.repeat_interleave(self.n_features, dim=0)  # (batch*n_features, out_dim)
            
            # Apply: (batch*n_features, 1, in_dim) @ (batch*n_features, in_dim, out_dim) -> no, w is (out, in)
            # We need: output = input @ w.T + b
            out_flat = out_flat.unsqueeze(1)  # (batch*n_features, 1, in_dim)
            # w_rep is (batch*n_features, out_dim, in_dim)
            # We want out_flat @ w_rep.transpose(-2, -1)
            out_new = torch.bmm(out_flat, w_rep.transpose(-2, -1)).squeeze(1) + b_rep  # (batch*n_features, out_dim)
            
            out = out_new.view(batch_size, self.n_features, -1)  # (batch, n_features, out_dim)
            
            if i < len(weights) - 1:  # Activation on hidden layers
                out = F.gelu(out)
                out = F.dropout(out, p=self.dropout, training=self.training)
        
        return out  # (batch, n_features, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Compute statistics for hypernetwork
        x_mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        x_std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        
        # Per-feature statistics
        stats = torch.cat([x_mean.squeeze(1), x_std.squeeze(1)], dim=-1)  # (batch, n_features * 2)
        
        # Also compute temporal statistics
        temporal_mean = x.mean(dim=-1)  # (batch, input_len)
        temporal_std = x.std(dim=-1) + 1e-5  # (batch, input_len)
        temporal_stats = torch.cat([temporal_mean, temporal_std], dim=-1)  # (batch, input_len * 2)
        
        # Combine statistics
        combined_stats = torch.cat([stats, temporal_stats], dim=-1)  # (batch, ...)
        
        # Pad or project to expected size
        if combined_stats.size(-1) < self.input_len * 2:
            padding = self.input_len * 2 - combined_stats.size(-1)
            combined_stats = F.pad(combined_stats, (0, padding))
        else:
            combined_stats = combined_stats[:, :self.input_len * 2]
        
        # Generate weights
        weights, biases, hyper_feat = self.generate_weights(combined_stats)
        
        # Prepare input for main network: channel-independent
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Apply generated network
        out = self.apply_main_network(x_perm, weights, biases)  # (batch, n_features, output_len)
        
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

