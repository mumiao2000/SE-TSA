import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttention_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_groups': [2, 8, 16],
            'd_model': [32, 128, 512],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.n_groups = self.config.get('n_groups', 8)
        self.d_model = self.config.get('d_model', 128)
        self.n_heads = self.config.get('n_heads', 8)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Feature grouping: soft assignment matrix
        self.group_logits = nn.Parameter(torch.randn(self.n_features, self.n_groups) * 0.1)
        
        # Input projection per group
        self.input_projs = nn.ModuleList([
            nn.Linear(self.input_len, self.d_model) for _ in range(self.n_groups)
        ])
        
        # Intra-group attention layers
        self.intra_attn = nn.ModuleList()
        for _ in range(self.n_layers):
            self.intra_attn.append(nn.ModuleDict({
                'qkv': nn.Linear(self.d_model, self.d_model * 3),
                'out': nn.Linear(self.d_model, self.d_model),
                'ln1': nn.LayerNorm(self.d_model),
                'ff': nn.Sequential(
                    nn.Linear(self.d_model, self.d_model * 4),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.d_model * 4, self.d_model)
                ),
                'ln2': nn.LayerNorm(self.d_model),
                'dropout': nn.Dropout(self.dropout_rate)
            }))
        
        # Inter-group attention: attend between group representatives
        self.inter_attn = nn.MultiheadAttention(self.d_model, self.n_heads, batch_first=True, dropout=self.dropout_rate)
        self.inter_ln = nn.LayerNorm(self.d_model)
        
        # Output projection: from concatenated group reps + global to output
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model * self.n_groups * 2, self.d_model * 4),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model * 4, self.output_len * self.n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.shape[0]
        
        # Soft group assignment
        group_weights = F.softmax(self.group_logits, dim=-1)  # (n_features, n_groups)
        
        # Transpose for feature-wise operations
        x_feat = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project features to group spaces
        group_inputs = []
        for g in range(self.n_groups):
            # Weighted combination of features for this group
            w = group_weights[:, g]  # (n_features,)
            # Each feature contributes to group g with weight w[f]
            x_g = x_feat * w.unsqueeze(0).unsqueeze(-1)  # (batch, n_features, input_len)
            # Sum across features to get group representation
            x_g = x_g.sum(dim=1)  # (batch, input_len)
            # Project
            x_g = self.input_projs[g](x_g)  # (batch, d_model)
            group_inputs.append(x_g)
        
        # Stack group representations: (batch, n_groups, d_model)
        group_seq = torch.stack(group_inputs, dim=1)  # (batch, n_groups, d_model)
        
        # Intra-group processing (self-attention on group sequence)
        h = group_seq  # (batch, n_groups, d_model)
        
        for layer in self.intra_attn:
            # Self-attention on groups
            h_ln = layer['ln1'](h)  # (batch, n_groups, d_model)
            qkv = layer['qkv'](h_ln)  # (batch, n_groups, d_model * 3)
            q, k, v = qkv.chunk(3, dim=-1)  # each (batch, n_groups, d_model)
            
            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)  # (batch, n_groups, n_groups)
            attn = F.softmax(scores, dim=-1)  # (batch, n_groups, n_groups)
            attn = layer['dropout'](attn)
            out = torch.matmul(attn, v)  # (batch, n_groups, d_model)
            out = layer['out'](out)  # (batch, n_groups, d_model)
            
            h = h + layer['dropout'](out)  # (batch, n_groups, d_model)
            
            # FF
            h = h + layer['dropout'](layer['ff'](layer['ln2'](h)))  # (batch, n_groups, d_model)
        
        # Inter-group attention with global context
        global_rep = h.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        global_enhanced, _ = self.inter_attn(global_rep, h, h)  # (batch, 1, d_model)
        global_enhanced = self.inter_ln(global_enhanced + global_rep)  # (batch, 1, d_model)
        
        # Broadcast and concatenate
        global_broadcast = global_enhanced.expand(-1, self.n_groups, -1)  # (batch, n_groups, d_model)
        combined = torch.cat([h, global_broadcast], dim=-1)  # (batch, n_groups, d_model * 2)
        
        # Flatten and project to output
        combined_flat = combined.view(batch_size, -1)  # (batch, n_groups * d_model * 2)
        
        # Project directly to (output_len * n_features)
        out = self.output_proj(combined_flat)  # (batch, output_len * n_features)
        out = out.view(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

