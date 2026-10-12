import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class PatchMixerGatedForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'patch_size': [2, 8, 24],
            'hidden_dim': [64, 256, 1024],
            'n_mixer_layers': [1, 4, 8],
            'expansion_factor': [2, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.patch_size = self.config.get('patch_size', 8)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_mixer_layers = self.config.get('n_mixer_layers', 4)
        self.expansion_factor = self.config.get('expansion_factor', 4)
        self.dropout = self.config.get('dropout', 0.3)
        
        # Ensure patch size divides input reasonably
        self.n_patches = max(1, self.input_len // self.patch_size)
        self.effective_patch_size = self.input_len // self.n_patches
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Patch embedding
        self.patch_embed = nn.Linear(self.effective_patch_size * self.n_features, self.hidden_dim)
        
        # Positional encoding for patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, self.hidden_dim) * 0.02)
        
        # Mixer layers
        self.mixer_layers = nn.ModuleList()
        for _ in range(self.n_mixer_layers):
            self.mixer_layers.append(nn.ModuleDict({
                # Temporal mixing (across patches)
                'temporal_norm': nn.LayerNorm(self.hidden_dim),
                'temporal_mlp': nn.Sequential(
                    nn.Linear(self.n_patches, self.n_patches * self.expansion_factor),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.n_patches * self.expansion_factor, self.n_patches),
                    nn.Dropout(self.dropout),
                ),
                # Channel mixing (within patch embedding)
                'channel_norm': nn.LayerNorm(self.hidden_dim),
                'channel_mlp': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * self.expansion_factor),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim * self.expansion_factor, self.hidden_dim),
                    nn.Dropout(self.dropout),
                ),
            }))
        
        # Channel-adaptive gating from patch representation
        self.gate_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.n_features),
            nn.Sigmoid()
        )
        
        # Output head: predict future patches
        self.future_patches = (self.output_len + self.effective_patch_size - 1) // self.effective_patch_size
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.effective_patch_size * self.n_features)
        )
        
        # Direct linear prediction for residual
        self.direct = nn.Linear(self.input_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Direct residual
        x_direct = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_direct = self.direct(x_direct)  # (batch, n_features, output_len)
        x_direct = x_direct.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Create patches
        batch_size = x.size(0)
        # Pad if needed
        pad_len = self.n_patches * self.effective_patch_size - self.input_len
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # (batch, padded_len, n_features)
        
        x_patches = x.view(batch_size, self.n_patches, self.effective_patch_size, self.n_features)  # (batch, n_patches, patch_size, n_features)
        x_patches = x_patches.view(batch_size, self.n_patches, -1)  # (batch, n_patches, patch_size * n_features)
        
        # Embed patches
        h = self.patch_embed(x_patches)  # (batch, n_patches, hidden_dim)
        h = h + self.pos_embed  # (batch, n_patches, hidden_dim)
        
        # Apply mixer layers
        for mixer in self.mixer_layers:
            # Temporal mixing: transpose to (batch, hidden_dim, n_patches)
            h_temp = mixer['temporal_norm'](h)  # (batch, n_patches, hidden_dim)
            h_temp = h_temp.permute(0, 2, 1)  # (batch, hidden_dim, n_patches)
            h_temp = mixer['temporal_mlp'](h_temp)  # (batch, hidden_dim, n_patches)
            h_temp = h_temp.permute(0, 2, 1)  # (batch, n_patches, hidden_dim)
            h = h + h_temp  # (batch, n_patches, hidden_dim)
            
            # Channel mixing
            h_chan = mixer['channel_norm'](h)  # (batch, n_patches, hidden_dim)
            h_chan = mixer['channel_mlp'](h_chan)  # (batch, n_patches, hidden_dim)
            h = h + h_chan  # (batch, n_patches, hidden_dim)
        
        # Predict future: use last patch state and autoregressively predict
        # Simple approach: predict all future patches from pooled representation
        h_pooled = h.mean(dim=1, keepdim=True)  # (batch, 1, hidden_dim)
        h_pooled = h_pooled.expand(-1, self.future_patches, -1)  # (batch, future_patches, hidden_dim)
        
        # Generate channel gates
        gate = self.gate_generator(h_pooled)  # (batch, future_patches, n_features)
        
        # Predict patch contents
        patches_pred = self.head(h_pooled)  # (batch, future_patches, patch_size * n_features)
        patches_pred = patches_pred.view(batch_size, self.future_patches, self.effective_patch_size, self.n_features)  # (batch, future_patches, patch_size, n_features)
        
        # Flatten to sequence
        out = patches_pred.view(batch_size, -1, self.n_features)  # (batch, future_patches * patch_size, n_features)
        out = out[:, :self.output_len, :]  # (batch, output_len, n_features)
        
        # Apply channel gating (tile gate for each timestep in patch)
        gate = gate.unsqueeze(2).expand(-1, -1, self.effective_patch_size, -1)  # (batch, future_patches, patch_size, n_features)
        gate = gate.reshape(batch_size, -1, self.n_features)[:, :self.output_len, :]  # (batch, output_len, n_features)
        out = out * gate  # (batch, output_len, n_features)
        
        # Add residual
        out = out + x_direct * 0.1  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)

