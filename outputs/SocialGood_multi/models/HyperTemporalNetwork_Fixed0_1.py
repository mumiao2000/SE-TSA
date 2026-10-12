import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperTemporalNetwork_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'base_channels': [32, 128, 512],
            'hyper_dim': [64, 256, 1024],
            'n_hyper_layers': [1, 2, 4],
            'temporal_depth': [2, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.base_channels = self.config.get('base_channels', 128)
        self.hyper_dim = self.config.get('hyper_dim', 256)
        self.n_hyper_layers = self.config.get('n_hyper_layers', 2)
        self.temporal_depth = self.config.get('temporal_depth', 4)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        # RevIN for proper normalization/denormalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Hypernetwork: generates convolution weights from global statistics
        # Input statistics: mean, std, trend (last - first), autocorrelation
        self.stat_encoder = nn.Sequential(
            nn.Linear(n_features * 4, self.hyper_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hyper_dim, self.hyper_dim)
        )
        
        # Hypernetwork layers
        self.hyper_layers = nn.ModuleList()
        for i in range(self.n_hyper_layers):
            self.hyper_layers.append(nn.Sequential(
                nn.Linear(self.hyper_dim, self.hyper_dim),
                nn.SiLU(),
                nn.Dropout(self.dropout_rate)
            ))
        
        # Generate weights for temporal convolutions
        # Each temporal layer needs: weight (out_ch, in_ch, kernel) and bias (out_ch)
        self.kernel_size = 7
        self.n_temporal_layers = self.temporal_depth
        
        # Calculate total parameters needed
        # First layer: n_features -> base_channels
        # Middle layers: base_channels -> base_channels  
        # Last layer: base_channels -> n_features
        self.weight_shapes = []
        self.bias_shapes = []
        in_ch = n_features
        for i in range(self.n_temporal_layers):
            out_ch = self.base_channels if i < self.n_temporal_layers - 1 else n_features
            self.weight_shapes.append((out_ch, in_ch, self.kernel_size))
            self.bias_shapes.append(out_ch)
            in_ch = out_ch
        
        total_params = sum(w[0]*w[1]*w[2] + b for w, b in zip(self.weight_shapes, self.bias_shapes))
        
        self.hyper_output = nn.Linear(self.hyper_dim, total_params)
        
        # Fixed temporal positions encoding
        self.time_embed = nn.Parameter(torch.randn(1, input_len, n_features) * 0.02)
        
        # Output projection from n_features to output_len
        self.output_proj = nn.Sequential(
            nn.Linear(input_len, output_len * 2),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(output_len * 2, output_len)
        )
        
        # Residual connection weights
        self.residual_gates = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(self.n_temporal_layers)
        ])

    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.size(0)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Add learnable time embedding
        x = x + self.time_embed  # (batch, input_len, n_features)
        
        # Compute statistics for hypernetwork
        mean_feat = x.mean(dim=1)  # (batch, n_features)
        std_feat = x.std(dim=1) + 1e-5  # (batch, n_features)
        trend_feat = x[:, -1, :] - x[:, 0, :]  # (batch, n_features)
        # Simple autocorrelation: correlation between first half and second half
        half = self.input_len // 2
        auto_feat = (x[:, :half, :].mean(dim=1) * x[:, half:2*half, :].mean(dim=1)) / (std_feat ** 2 + 1e-5)  # (batch, n_features)
        
        stats = torch.cat([mean_feat, std_feat, trend_feat, auto_feat], dim=-1)  # (batch, n_features*4)
        
        # Hypernetwork generates weights
        hyper_state = self.stat_encoder(stats)  # (batch, hyper_dim)
        for hlayer in self.hyper_layers:
            hyper_state = hlayer(hyper_state) + hyper_state  # residual
        
        all_params = self.hyper_output(hyper_state)  # (batch, total_params)
        
        # Apply generated temporal convolutions
        current = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        param_idx = 0
        
        for i, (w_shape, b_shape) in enumerate(zip(self.weight_shapes, self.bias_shapes)):
            # Extract generated weights
            w_size = w_shape[0] * w_shape[1] * w_shape[2]
            b_size = b_shape
            
            W = all_params[:, param_idx:param_idx+w_size].view(batch_size, *w_shape)  # (batch, out_ch, in_ch, kernel)
            param_idx += w_size
            b = all_params[:, param_idx:param_idx+b_size]  # (batch, out_ch)
            param_idx += b_size
            
            # Apply conv with generated weights
            out_ch, in_ch, k = w_shape
            padding = k // 2
            
            # Manual grouped convolution: process each sample
            outputs = []
            for bi in range(batch_size):
                # current[bi]: (in_ch, input_len)
                # W[bi]: (out_ch, in_ch, k)
                out = F.conv1d(current[bi:bi+1], W[bi], b[bi], padding=padding)  # (1, out_ch, input_len)
                outputs.append(out)
            current = torch.cat(outputs, dim=0)  # (batch, out_ch, input_len)
            
            # Activation
            current = F.silu(current)  # (batch, out_ch, input_len)
            
            # Residual connection when channels match
            if i > 0 and w_shape[0] == self.weight_shapes[i-1][0] and i < self.n_temporal_layers - 1:
                gate = torch.sigmoid(self.residual_gates[i])
                # Residual from previous layer with same channels
                # current and prev have same shape
                pass  # Residual handled by storing prev if needed; skip for simplicity
        
        # current: (batch, n_features, input_len) after last layer (out_ch = n_features)
        
        # Project temporal dimension to output_len
        out = self.output_proj(current)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

