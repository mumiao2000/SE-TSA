import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveMLP_Fixed0_Exp_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'gate_dim': [32, 256, 512],
            'n_scales': [2, 3, 4],
            'freq_dim': [16, 64, 256]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout = self.config.get('dropout', 0.3)
        self.gate_dim = self.config.get('gate_dim', 256)
        self.n_scales = self.config.get('n_scales', 3)
        self.freq_dim = self.config.get('freq_dim', 64)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale temporal decomposition: different pooling scales
        self.scale_poolers = nn.ModuleList()
        self.scale_projectors = nn.ModuleList()
        for i in range(self.n_scales):
            pool_size = 2 ** i
            # Adaptive pooling to handle variable input lengths
            self.scale_poolers.append(nn.AdaptiveAvgPool1d(max(1, self.input_len // pool_size)))
            self.scale_projectors.append(nn.Linear(max(1, self.input_len // pool_size), self.input_len))
        
        # Channel-wise feature extraction at each scale
        self.scale_encoders = nn.ModuleList()
        for _ in range(self.n_scales):
            layers = []
            in_dim = self.n_features
            for _ in range(self.n_layers):
                layers.append(nn.Linear(in_dim, self.hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(self.dropout))
                in_dim = self.hidden_dim
            layers.append(nn.Linear(in_dim, self.n_features))
            self.scale_encoders.append(nn.Sequential(*layers))
        
        # Hierarchical cross-channel gating
        self.gate_networks = nn.ModuleList()
        for i in range(self.n_scales):
            gate = nn.Sequential(
                nn.Linear(self.n_features * (i + 1), self.gate_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.gate_dim, self.n_features),
                nn.Sigmoid()
            )
            self.gate_networks.append(gate)
        
        # Frequency-aware branch: learnable sinusoidal embeddings
        self.freq_omega = nn.Parameter(torch.randn(self.freq_dim, self.input_len) * 0.01)
        self.freq_phi = nn.Parameter(torch.zeros(self.freq_dim, self.input_len))
        self.freq_proj = nn.Sequential(
            nn.Linear(self.freq_dim * 2, self.hidden_dim),  # *2 for sin and cos
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
        # Fusion and output projection
        total_features = self.n_features * (self.n_scales + 1)  # +1 for frequency branch
        self.fusion = nn.Sequential(
            nn.Linear(total_features, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_features)
        )
        
        # Final temporal projection to output horizon
        self.temporal_proj = nn.Linear(self.input_len, self.output_len)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Multi-scale processing
        scale_features = []
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        for i in range(self.n_scales):
            # Pool and project back to original length
            x_pooled = self.scale_poolers[i](x_perm)  # (batch, n_features, pooled_len)
            x_projected = self.scale_projectors[i](x_pooled)  # (batch, n_features, input_len)
            x_projected = x_projected.permute(0, 2, 1)  # (batch, input_len, n_features)
            
            # Channel-wise encoding
            encoded = self.scale_encoders[i](x_projected)  # (batch, input_len, n_features)
            
            # Hierarchical gating: accumulate previous scale features
            if i == 0:
                gate_input = encoded  # (batch, input_len, n_features)
            else:
                prev_concat = torch.cat(scale_features + [encoded], dim=-1)  # (batch, input_len, n_features * (i+1))
                gate = self.gate_networks[i](prev_concat)  # (batch, input_len, n_features)
                encoded = encoded * gate + sum(scale_features) * (1 - gate) / len(scale_features)
            
            scale_features.append(encoded)
        
        # Frequency-aware branch
        # Create learnable sinusoidal features
        t = torch.arange(self.input_len, device=x.device).float().unsqueeze(0)  # (1, input_len)
        # (freq_dim, input_len) * (1, input_len) -> element-wise then sum? No, we want outer product style
        # Actually: omega has shape (freq_dim, input_len), t has shape (1, input_len)
        # We want omega_t[b, f, t] = omega[f, t] * t[0, t] for each batch
        omega_t = self.freq_omega.unsqueeze(0) * t.unsqueeze(1)  # (1, freq_dim, input_len) * (1, 1, input_len) -> (1, freq_dim, input_len)
        omega_t = omega_t.expand(batch_size, -1, -1)  # (batch, freq_dim, input_len)
        
        # freq_phi shape: (freq_dim, input_len) -> (1, freq_dim, input_len) for broadcasting
        freq_phi_expanded = self.freq_phi.unsqueeze(0)  # (1, freq_dim, input_len)
        
        sin_feat = torch.sin(omega_t + freq_phi_expanded)  # (batch, freq_dim, input_len)
        cos_feat = torch.cos(omega_t + freq_phi_expanded)  # (batch, freq_dim, input_len)
        freq_feat = torch.cat([sin_feat, cos_feat], dim=1)  # (batch, freq_dim*2, input_len)
        freq_feat = freq_feat.permute(0, 2, 1)  # (batch, input_len, freq_dim*2)
        
        freq_out = self.freq_proj(freq_feat)  # (batch, input_len, n_features)
        
        # Concatenate all features
        all_features = torch.cat(scale_features + [freq_out], dim=-1)  # (batch, input_len, n_features * (n_scales + 1))
        
        # Fusion
        fused = self.fusion(all_features)  # (batch, input_len, n_features)
        
        # Residual connection with original input
        fused = fused + x  # (batch, input_len, n_features)
        
        # Temporal projection to output horizon
        fused = fused.permute(0, 2, 1)  # (batch, n_features, input_len)
        out = self.temporal_proj(fused)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

