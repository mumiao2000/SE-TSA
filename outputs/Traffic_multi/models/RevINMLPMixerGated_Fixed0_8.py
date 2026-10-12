import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class RevINMLPMixerGated_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'patch_size': [2, 8, 24],
            'hidden_dim': [64, 256, 1024],
            'expansion_factor': [2, 4, 8],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config parameters with defaults
        self.patch_size = self.config.get('patch_size', 8)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.expansion_factor = self.config.get('expansion_factor', 4)
        self.n_layers = self.config.get('n_layers', 4)
        self.dropout_rate = self.config.get('dropout', 0.0)
        
        # Ensure patch size divides input length
        if self.input_len % self.patch_size != 0:
            self.patch_size = self._find_best_patch_size(self.input_len)
        
        self.n_patches = self.input_len // self.patch_size
        self.patch_dim = self.patch_size
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding: linear projection from patch_dim to hidden_dim
        self.patch_embed = nn.Linear(self.patch_dim, self.hidden_dim)
        
        # MLP-Mixer blocks with GELU-GLU gating
        self.mixer_blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.mixer_blocks.append(self._create_mixer_block())
        
        # Output head: project from (n_patches * hidden_dim) to output_len
        self.head = nn.Sequential(
            nn.Linear(self.n_patches * self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.output_len)
        )
        
        self._init_weights()
    
    def _find_best_patch_size(self, input_len):
        # Find largest divisor of input_len from common patch sizes
        candidates = [24, 16, 12, 8, 6, 4, 3, 2, 1]
        for p in candidates:
            if input_len % p == 0:
                return p
        return 1
    
    def _create_mixer_block(self):
        return nn.ModuleDict({
            'norm1': nn.LayerNorm(self.hidden_dim),
            'temporal_mlp': self._create_gated_mlp(self.n_patches, self.expansion_factor),
            'norm2': nn.LayerNorm(self.hidden_dim),
            'channel_mlp': self._create_gated_mlp(self.hidden_dim, self.expansion_factor),
        })
    
    def _create_gated_mlp(self, dim, expansion_factor):
        hidden = dim * expansion_factor
        return nn.ModuleDict({
            'fc1': nn.Linear(dim, hidden * 2),  # 2x for gating: gate and value
            'fc2': nn.Linear(hidden, dim),  # project back to original dim
        })
    
    def _gated_mlp_forward(self, mlp, x):
        # x: (batch, n_features, seq_len, dim)
        x_proj = mlp['fc1'](x)  # (batch, n_features, seq_len, hidden*2)
        x_gate, x_val = torch.chunk(x_proj, 2, dim=-1)  # each (batch, n_features, seq_len, hidden)
        x_gate = F.gelu(x_gate)  # apply GELU to gate (proper GLU)
        x_gated = x_gate * x_val  # element-wise product
        return mlp['fc2'](x_gated)  # (batch, n_features, seq_len, dim)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose to channel-first: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Patch: (batch, n_features, n_patches, patch_size)
        x = x.reshape(batch_size, self.n_features, self.n_patches, self.patch_dim)  # (batch, n_features, n_patches, patch_dim)
        
        # Patch embedding: (batch, n_features, n_patches, hidden_dim)
        x = self.patch_embed(x)  # (batch, n_features, n_patches, hidden_dim)
        
        # MLP-Mixer blocks
        for block in self.mixer_blocks:
            # Temporal mixing: mix across patches (dim=2)
            residual = x  # (batch, n_features, n_patches, hidden_dim)
            x = block['norm1'](x)  # (batch, n_features, n_patches, hidden_dim)
            x = x.permute(0, 1, 3, 2)  # (batch, n_features, hidden_dim, n_patches)
            x = self._gated_mlp_forward(block['temporal_mlp'], x)  # (batch, n_features, hidden_dim, n_patches)
            x = x.permute(0, 1, 3, 2)  # (batch, n_features, n_patches, hidden_dim)
            x = x + residual  # (batch, n_features, n_patches, hidden_dim)
            
            # Channel mixing: mix across hidden_dim (dim=3)
            residual = x  # (batch, n_features, n_patches, hidden_dim)
            x = block['norm2'](x)  # (batch, n_features, n_patches, hidden_dim)
            x = self._gated_mlp_forward(block['channel_mlp'], x)  # (batch, n_features, n_patches, hidden_dim)
            x = x + residual  # (batch, n_features, n_patches, hidden_dim)
        
        # Flatten patches: (batch, n_features, n_patches * hidden_dim)
        x = x.reshape(batch_size, self.n_features, -1)  # (batch, n_features, n_patches * hidden_dim)
        
        # Output head: (batch, n_features, output_len)
        x = self.head(x)  # (batch, n_features, output_len)
        
        # Transpose back: (batch, output_len, n_features)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x  # (batch, output_len, n_features)
