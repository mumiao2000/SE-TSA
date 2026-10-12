import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class MultiResolutionWaveletNet(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_scales': [2, 4, 8],
            'base_hidden': [32, 128, 512],
            'n_layers_per_scale': [1, 3, 6],
            'fusion_dim': [64, 256, 1024],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_scales = self.config.get('n_scales', 4)
        self.base_hidden = self.config.get('base_hidden', 128)
        self.n_layers_per_scale = self.config.get('n_layers_per_scale', 3)
        self.fusion_dim = self.config.get('fusion_dim', 256)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Learnable normalization
        self.scale = nn.Parameter(torch.ones(n_features))
        self.shift = nn.Parameter(torch.zeros(n_features))
        
        # Multi-resolution branches: each scale uses different dilation
        self.branches = nn.ModuleList()
        for s in range(self.n_scales):
            dilation = 2 ** s
            hidden = self.base_hidden // (2 ** (s // 2))  # smaller hidden for higher frequencies
            hidden = max(hidden, 16)
            self.branches.append(
                ResolutionBranch(n_features, hidden, self.n_layers_per_scale, dilation, self.dropout, input_len)
            )
        
        # Fusion: combine all scales with gating
        total_branch_dim = sum(b.out_dim for b in self.branches)
        self.fusion = nn.Sequential(
            nn.Linear(total_branch_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.fusion_dim, self.fusion_dim)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(total_branch_dim, self.n_scales),
            nn.Softmax(dim=-1)
        )
        
        # Output heads: one per scale + fused
        self.output_heads = nn.ModuleList()
        for s in range(self.n_scales):
            self.output_heads.append(
                nn.Linear(self.branches[s].out_dim, output_len * n_features)
            )
        self.fused_head = nn.Linear(self.fusion_dim, output_len * n_features)
        
        # Final mixing
        self.final_mix = nn.Linear((self.n_scales + 1) * n_features, n_features)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        B = x.shape[0]
        
        # Normalize
        mean = x.mean(dim=1, keepdim=True)  # (batch, 1, n_features)
        std = x.std(dim=1, keepdim=True) + 1e-5  # (batch, 1, n_features)
        x_norm = (x - mean) / std  # (batch, input_len, n_features)
        x_norm = x_norm * self.scale + self.shift  # (batch, input_len, n_features)
        
        # Process through each branch
        branch_outputs = []
        for branch in self.branches:
            out = branch(x_norm)  # (batch, branch.out_dim)
            branch_outputs.append(out)
        
        # Concatenate all branch outputs
        combined = torch.cat(branch_outputs, dim=-1)  # (batch, total_branch_dim)
        
        # Compute gates
        gates = self.gate(combined)  # (batch, n_scales)
        
        # Fusion
        fused = self.fusion(combined)  # (batch, fusion_dim)
        
        # Generate predictions from each branch and fused representation
        predictions = []
        for i, head in enumerate(self.output_heads):
            pred = head(branch_outputs[i])  # (batch, output_len * n_features)
            pred = pred.reshape(B, self.output_len, self.n_features)  # (batch, output_len, n_features)
            # Weight by gate
            pred = pred * gates[:, i:i+1, None]  # (batch, output_len, n_features)
            predictions.append(pred)
        
        # Fused prediction
        fused_pred = self.fused_head(fused)  # (batch, output_len * n_features)
        fused_pred = fused_pred.reshape(B, self.output_len, self.n_features)  # (batch, output_len, n_features)
        predictions.append(fused_pred)
        
        # Stack and mix
        all_preds = torch.stack(predictions, dim=-1)  # (batch, output_len, n_features, n_scales+1)
        all_preds = all_preds.reshape(B, self.output_len, -1)  # (batch, output_len, (n_scales+1)*n_features)
        
        out = self.final_mix(all_preds)  # (batch, output_len, n_features)
        
        # Denormalize
        out = out * std + mean  # (batch, output_len, n_features)
        
        return out


class ResolutionBranch(nn.Module):
    def __init__(self, n_features, hidden_dim, n_layers, dilation, dropout, input_len):
        super().__init__()
        self.out_dim = hidden_dim
        
        layers = []
        in_ch = n_features
        for i in range(n_layers):
            out_ch = hidden_dim
            # Use 1D conv with dilation for this resolution
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation))
            layers.append(nn.LayerNorm([out_ch]))  # normalize feature dim
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.drops = nn.ModuleList()
        
        in_ch = n_features
        for i in range(n_layers):
            out_ch = hidden_dim
            self.convs.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation))
            self.norms.append(nn.LayerNorm(out_ch))
            self.acts.append(nn.GELU())
            self.drops.append(nn.Dropout(dropout))
            in_ch = out_ch
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = x.transpose(1, 2)  # (batch, n_features, input_len)
        
        for conv, norm, act, drop in zip(self.convs, self.norms, self.acts, self.drops):
            x = conv(x)  # (batch, hidden_dim, input_len)
            # Trim padding to maintain length
            if x.shape[2] > 1000:  # safety check
                x = x[:, :, :-1]  # remove last padding
            x = x.transpose(1, 2)  # (batch, input_len, hidden_dim)
            x = norm(x)
            x = x.transpose(1, 2)  # (batch, hidden_dim, input_len)
            x = act(x)
            x = drop(x)
        
        # Global average pooling
        x = self.pool(x).squeeze(-1)  # (batch, hidden_dim)
        return x
