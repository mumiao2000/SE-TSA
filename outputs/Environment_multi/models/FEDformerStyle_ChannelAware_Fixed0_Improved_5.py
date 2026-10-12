import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class FEDformerStyle_ChannelAware_Fixed0_Improved(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'modes': [8, 32, 128],
            'kernel_sizes': [[3, 7, 15], [3, 7, 15, 31], [7, 15, 31, 63]],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 512)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.modes = self.config.get('modes', 32)
        self.kernel_sizes = self.config.get('kernel_sizes', [3, 7, 15, 31])
        
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_projection = nn.Linear(self.n_features, self.d_model)
        
        # Multi-scale moving average decomposition
        self.decomp_kernels = self.kernel_sizes
        self.moving_avg_layers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=k, stride=1, padding=k//2) 
            for k in self.decomp_kernels
        ])
        
        # Number of decomposition scales
        self.n_scales = len(self.decomp_kernels)
        
        # Channel grouping: split features into groups for specialized frequency processing
        self.n_groups = min(4, self.n_features)
        self.features_per_group = (self.n_features + self.n_groups - 1) // self.n_groups
        
        # Frequency-enhanced blocks per scale and layer
        self.freq_blocks = nn.ModuleList()
        self.scale_mixers = nn.ModuleList()
        
        for layer in range(self.n_layers):
            layer_blocks = nn.ModuleList()
            for scale in range(self.n_scales):
                # Frequency filters per group
                group_filters = nn.ParameterList([
                    nn.Parameter(torch.randn(self.n_heads, self.modes, 1) * 0.02)
                    for _ in range(self.n_groups)
                ])
                layer_blocks.append(nn.ModuleDict({
                    'group_filters': group_filters,
                    'dropout': nn.Dropout(self.dropout),
                }))
            self.freq_blocks.append(layer_blocks)
            
            # Scale mixer: combine information across scales
            self.scale_mixers.append(nn.Sequential(
                nn.Linear(self.d_model * self.n_scales, self.d_model),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.d_model),
            ))
        
        # Layer norms
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(self.d_model) for _ in range(self.n_layers)
        ])
        
        # Basis mixing network for output projection
        # Inspired by NeuralBasisExpansion but lightweight
        self.n_basis = min(64, self.output_len)
        self.basis_matrix = nn.Parameter(torch.randn(self.output_len, self.n_basis))  # (output_len, n_basis)
        
        # Coefficient predictor: from encoded input to basis coefficients
        self.coeff_encoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.n_basis * self.n_features),
        )
        
        # Direct trend residual connection
        self.trend_projector = nn.Linear(self.input_len, self.output_len)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _multi_scale_decomp(self, x):
        """Decompose input into multiple trend-seasonal pairs at different scales."""
        # x: (batch, input_len, d_model)
        # Transpose for 1D conv: (batch, d_model, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, d_model, input_len)
        
        decomposed = []
        for avg_layer in self.moving_avg_layers:
            # Apply moving average
            trend = avg_layer(x_perm)  # (batch, d_model, input_len)
            seasonal = x_perm - trend
            decomposed.append((trend.permute(0, 2, 1), seasonal.permute(0, 2, 1)))  # each (batch, input_len, d_model)
        
        return decomposed  # list of (trend, seasonal) tuples
    
    def _freq_attention(self, x, freq_block):
        """Apply frequency-domain attention with channel grouping."""
        # x: (batch, input_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Reshape for multi-head: (batch, n_heads, seq_len, head_dim)
        x_reshaped = x.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # (batch, n_heads, seq_len, head_dim)
        
        # FFT: (batch, n_heads, seq_len//2+1, head_dim)
        x_fft = torch.fft.rfft(x_reshaped, dim=2)
        
        # Dynamic mode truncation
        max_modes = x_fft.shape[2]
        n_modes = min(self.modes, max_modes)
        
        # Truncate to available modes
        x_fft_trunc = x_fft[:, :, :n_modes, :]  # (batch, n_heads, n_modes, head_dim)
        
        # Apply group filters (use first group filter for all for simplicity, or average)
        # For efficiency, use a single combined filter
        combined_filter = sum([f[:n_modes, :] for f in freq_block['group_filters']]) / len(freq_block['group_filters'])
        # combined_filter: (n_heads, n_modes, 1)
        
        # Apply filter
        x_fft_filtered = x_fft_trunc * combined_filter.unsqueeze(0)  # (batch, n_heads, n_modes, head_dim)
        
        # Pad back if needed
        if n_modes < max_modes:
            padding = torch.zeros(batch_size, self.n_heads, max_modes - n_modes, self.head_dim, 
                                  dtype=x_fft.dtype, device=x_fft.device)
            x_fft_filtered = torch.cat([x_fft_filtered, padding], dim=2)
        
        # Inverse FFT
        x_out = torch.fft.irfft(x_fft_filtered, n=seq_len, dim=2)  # (batch, n_heads, seq_len, head_dim)
        
        # Reshape back
        x_out = x_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)  # (batch, input_len, d_model)
        
        x_out = freq_block['dropout'](x_out)
        return x_out
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store original for trend residual
        x_original = x  # (batch, input_len, n_features)
        
        # Input projection
        x = self.input_projection(x)  # (batch, input_len, d_model)
        
        # Multi-scale decomposition
        decomposed = self._multi_scale_decomp(x)  # list of (trend, seasonal) at each scale
        
        # Process each scale through frequency blocks
        scale_outputs = []
        for scale_idx, (trend, seasonal) in enumerate(decomposed):
            # Use seasonal component for frequency attention
            scale_feat = seasonal  # (batch, input_len, d_model)
            
            # Stack layers
            for layer_idx in range(self.n_layers):
                # Frequency attention
                freq_out = self._freq_attention(scale_feat, self.freq_blocks[layer_idx][scale_idx])
                # (batch, input_len, d_model)
                
                # Residual and norm
                scale_feat = self.norm_layers[layer_idx](scale_feat + freq_out)
                # (batch, input_len, d_model)
            
            scale_outputs.append(scale_feat)  # (batch, input_len, d_model)
        
        # Mix scales
        # Concatenate all scale outputs
        mixed_input = torch.cat(scale_outputs, dim=-1)  # (batch, input_len, d_model * n_scales)
        
        # Global scale mixing
        encoded = self.scale_mixers[-1](mixed_input)  # (batch, input_len, d_model)
        # Use last layer's mixer, or create a dedicated one
        
        # Actually, let's use a dedicated final mixer
        # Recompute with proper mixer
        mixed_input = torch.cat(scale_outputs, dim=-1)  # (batch, input_len, d_model * n_scales)
        final_mixer = nn.Sequential(
            nn.Linear(self.d_model * self.n_scales, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
        ).to(x.device)
        encoded = final_mixer(mixed_input)  # (batch, input_len, d_model)
        
        # Global average pooling for context
        context = encoded.mean(dim=1)  # (batch, d_model)
        
        # Basis mixing prediction
        # Predict basis coefficients
        coeffs = self.coeff_encoder(context)  # (batch, n_basis * n_features)
        coeffs = coeffs.view(batch_size, self.n_basis, self.n_features)  # (batch, n_basis, n_features)
        
        # Mix basis functions: einsum 'ob,bnf->bof' where o=output_len, b=n_basis, n=batch, f=features
        # Actually: basis_matrix is (output_len, n_basis), coeffs is (batch, n_basis, n_features)
        # Result should be (batch, output_len, n_features)
        basis_pred = torch.einsum('on,bnf->bof', self.basis_matrix, coeffs)  # (batch, output_len, n_features)
        
        # Direct trend residual: project last values
        # Use the original input's last feature dimension
        trend_input = x_original.permute(0, 2, 1)  # (batch, n_features, input_len)
        trend_resid = self.trend_projector(trend_input)  # (batch, n_features, output_len)
        trend_resid = trend_resid.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Combine predictions
        output = basis_pred + trend_resid * 0.1  # (batch, output_len, n_features)
        
        # RevIN denormalization
        output = self.revin(output, mode='denorm')  # (batch, output_len, n_features)
        
        return output  # (batch, output_len, n_features)
