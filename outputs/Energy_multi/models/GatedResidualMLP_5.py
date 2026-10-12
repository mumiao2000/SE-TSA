import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class GatedResidualMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_layers': [2, 6, 12],
            'dropout': [0.0, 0.2],
            'n_scales': [2, 4, 8]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 6)
        self.dropout = self.config.get('dropout', 0.1)
        self.n_scales = self.config.get('n_scales', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale temporal convolutions
        self.scale_convs = nn.ModuleList()
        for scale in range(1, self.n_scales + 1):
            # Dilated or strided convolution for different scales
            if scale == 1:
                self.scale_convs.append(nn.Conv1d(n_features, n_features, kernel_size=3, padding=1))
            else:
                self.scale_convs.append(nn.Conv1d(n_features, n_features, kernel_size=scale*2+1, 
                                                   stride=scale, padding=scale))
        
        # Scale projection to common dimension
        scale_out_len = input_len  # For scale=1; others need handling
        self.scale_projs = nn.ModuleList()
        for scale in range(1, self.n_scales + 1):
            if scale == 1:
                self.scale_projs.append(nn.Identity())
            else:
                # Interpolate or project to common length
                self.scale_projs.append(nn.Linear((input_len + scale - 1) // scale, input_len))
        
        # Gated MLP layers
        self.gated_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        in_dim = input_len * n_features * self.n_scales  # Concatenated multi-scale features
        
        # Actually, let's do per-feature processing then combine
        # Simpler: flatten and use MLP with gating
        
        self.input_proj = nn.Linear(input_len, self.hidden_dim)
        
        for i in range(self.n_layers):
            # Gated residual block: two parallel branches
            self.gated_layers.append(nn.ModuleDict({
                'transform': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.GELU()
                ),
                'gate': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.Sigmoid()
                ),
                'proj': nn.Linear(self.hidden_dim, self.hidden_dim)
            }))
            self.layer_norms.append(nn.LayerNorm(self.hidden_dim))
        
        # Multi-scale feature extraction
        self.scale_weights = nn.Parameter(torch.ones(self.n_scales) / self.n_scales)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, output_len)
        )
        
        # Direct feature mixing
        self.feature_gate = nn.Sequential(
            nn.Linear(n_features, n_features * 2),
            nn.GELU(),
            nn.Linear(n_features * 2, n_features),
            nn.Sigmoid()
        )
        
    def _extract_multiscale(self, x):
        # x: (batch, input_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        scale_features = []
        for i, (conv, proj) in enumerate(zip(self.scale_convs, self.scale_projs)):
            scale = i + 1
            xf = conv(x)  # (batch, n_features, conv_out_len)
            if scale > 1:
                # Interpolate back to input_len
                xf = F.interpolate(xf, size=self.input_len, mode='linear', align_corners=False)
            scale_features.append(xf)
        
        # Weighted combination
        weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * f for w, f in zip(weights, scale_features))  # (batch, n_features, input_len)
        
        return combined.permute(0, 2, 1)  # (batch, input_len, n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Multi-scale feature extraction
        x_multi = self._extract_multiscale(x)  # (batch, input_len, n_features)
        
        # Combine with original
        x = x + x_multi * 0.5  # (batch, input_len, n_features)
        
        # Feature gating
        feature_gate = self.feature_gate(x.mean(dim=1))  # (batch, n_features)
        x = x * feature_gate.unsqueeze(1)  # (batch, input_len, n_features)
        
        # Per-feature MLP with gating
        # Process each feature independently through deep network
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.input_proj(x)  # (batch, n_features, hidden_dim)
        
        # Deep gated residual blocks
        for layer, norm in zip(self.gated_layers, self.layer_norms):
            residual = x  # (batch, n_features, hidden_dim)
            
            # Transform branch
            h = layer['transform'](x)  # (batch, n_features, hidden_dim * 2)
            h1, h2 = h.chunk(2, dim=-1)  # Each: (batch, n_features, hidden_dim)
            h = h1 * h2  # Gated multiplication: (batch, n_features, hidden_dim)
            
            # Gate branch
            g = layer['gate'](x)  # (batch, n_features, hidden_dim)
            
            # Apply gate and project
            h = layer['proj'](h * g)  # (batch, n_features, hidden_dim)
            
            # Residual connection with layer norm
            x = norm(residual + h)  # (batch, n_features, hidden_dim)
        
        # Output projection per feature
        x = self.output_proj(x)  # (batch, n_features, output_len)
        
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        return x  # (batch, output_len, n_features)
