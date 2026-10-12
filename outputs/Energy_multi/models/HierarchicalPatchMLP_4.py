import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HierarchicalPatchMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_scales': [2, 3],
            'mix_dim': [32, 128]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_scales = self.config.get('n_scales', 3)
        self.mix_dim = self.config.get('mix_dim', 128)
        
        # Define patch sizes: must divide input_len reasonably
        self.patch_sizes = [1, 4, 16][:self.n_scales]
        self.patch_sizes = [p for p in self.patch_sizes if p <= self.input_len]
        
        # Per-scale encoders: patch -> MLP -> hidden
        self.scale_encoders = nn.ModuleList()
        for p in self.patch_sizes:
            n_patches = self.input_len // p
            encoder = nn.Sequential(
                nn.Linear(n_patches, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
            self.scale_encoders.append(encoder)
        
        # Cross-scale mixing
        total_hidden = self.hidden_dim * len(self.patch_sizes)
        self.cross_mix = nn.Sequential(
            nn.Linear(total_hidden, self.mix_dim),
            nn.GELU(),
            nn.Linear(self.mix_dim, self.hidden_dim)
        )
        
        # Decoder: hidden -> output_len
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_len)
        )
        
        self.revin = RevIN(self.n_features, affine=True)

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Process each scale
        scale_reprs = []
        for p, encoder in zip(self.patch_sizes, self.scale_encoders):
            # Patch averaging: (batch, n_patches, n_features)
            n_patches = self.input_len // p
            x_patched = x[:, :n_patches * p, :].reshape(x.size(0), n_patches, p, x.size(-1))
            x_patched = x_patched.mean(dim=2)  # (batch, n_patches, n_features)
            
            # Encode per feature
            x_enc = x_patched.permute(0, 2, 1)  # (batch, n_features, n_patches)
            x_enc = encoder(x_enc)  # (batch, n_features, hidden_dim)
            scale_reprs.append(x_enc)
        
        # Concatenate and cross-mix
        combined = torch.cat(scale_reprs, dim=-1)  # (batch, n_features, total_hidden)
        mixed = self.cross_mix(combined)  # (batch, n_features, hidden_dim)
        
        # Decode to output
        out = self.decoder(mixed)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

