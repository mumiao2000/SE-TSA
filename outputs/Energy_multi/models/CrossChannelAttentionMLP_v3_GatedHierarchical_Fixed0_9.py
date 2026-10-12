import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttentionMLP_v3_GatedHierarchical_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [256, 1024, 2048],
            'n_layers': [2, 4, 8],
            'temporal_kernel_sizes': [[3, 5, 7], [5, 7, 11], [7, 11, 15]],
            'dropout': [0.0, 0.1, 0.3],
            'gate_dim': [64, 256, 512],
            'n_heads': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 1024)
        self.n_layers = self.config.get('n_layers', 4)
        temporal_kernels = self.config.get('temporal_kernel_sizes', [3, 5, 7])
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.gate_dim = self.config.get('gate_dim', 256)
        self.n_heads = self.config.get('n_heads', 4)
        
        # Condition-aware depth: deeper for short horizons, wider for long
        self.effective_layers = max(2, self.n_layers - self.output_len // 24)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection to hidden dimension
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Hierarchical temporal convolutions: multi-scale feature extraction
        # Compute output channels for each conv so they sum to hidden_dim
        n_kernels = len(temporal_kernels)
        base_channels = self.hidden_dim // n_kernels
        remainder = self.hidden_dim % n_kernels
        conv_out_channels = []
        for i in range(n_kernels):
            # Distribute remainder across first few convs
            ch = base_channels + (1 if i < remainder else 0)
            conv_out_channels.append(ch)
        
        self.temporal_convs = nn.ModuleList()
        self.temporal_gates = nn.ModuleList()
        for i, k in enumerate(temporal_kernels):
            out_ch = conv_out_channels[i]
            # Causal padding to maintain sequence length
            pad = (k - 1) // 2
            self.temporal_convs.append(
                nn.Conv1d(self.hidden_dim, out_ch, 
                         kernel_size=k, padding=pad, bias=False)
            )
            # Gating mechanism per scale
            self.temporal_gates.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(out_ch, self.gate_dim),
                    nn.SiLU(),
                    nn.Linear(self.gate_dim, out_ch),
                    nn.Sigmoid()
                )
            )
        
        # Actual fused dimension after concatenation
        self.fused_dim = sum(conv_out_channels)
        
        # Fusion of multi-scale temporal features
        self.temporal_fusion = nn.Sequential(
            nn.Linear(self.fused_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Cross-channel attention: operate in n_features space via projection
        # Project hidden_dim -> n_features, apply attention, project back
        self.channel_proj_down = nn.Linear(self.hidden_dim, self.n_features)
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=self.n_features,
            num_heads=self.n_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.channel_proj_up = nn.Linear(self.n_features, self.hidden_dim)
        self.channel_norm = nn.LayerNorm(self.hidden_dim)
        
        # Residual MLP blocks with depth modulation
        self.mlp_blocks = nn.ModuleList()
        for i in range(self.effective_layers):
            self.mlp_blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(self.hidden_dim),
                'fc1': nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                'act': nn.SiLU(),
                'drop1': nn.Dropout(self.dropout_rate),
                'fc2': nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                'drop2': nn.Dropout(self.dropout_rate),
                'gate': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.gate_dim),
                    nn.SiLU(),
                    nn.Linear(self.gate_dim, self.hidden_dim),
                    nn.Sigmoid()
                )
            }))
        
        # Horizon-aware temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output MLP with horizon-specific scaling
        output_hidden = self.hidden_dim // 2
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, output_hidden),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(output_hidden, self.output_len * self.n_features)
        )
        
        # Learnable horizon scale factor
        self.horizon_scale = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Project to hidden dimension
        x = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Hierarchical temporal convolutions with gating
        x_conv = x.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        
        conv_outputs = []
        for conv, gate in zip(self.temporal_convs, self.temporal_gates):
            conv_out = conv(x_conv)  # (batch, out_ch, input_len)
            # Compute scale gate from pooled features
            gate_val = gate(conv_out)  # (batch, out_ch)
            gate_val = gate_val.unsqueeze(-1)  # (batch, out_ch, 1)
            conv_out = conv_out * gate_val  # (batch, out_ch, input_len)
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features
        x_fused = torch.cat(conv_outputs, dim=1)  # (batch, fused_dim, input_len)
        x_fused = x_fused.permute(0, 2, 1)  # (batch, input_len, fused_dim)
        x_fused = self.temporal_fusion(x_fused)  # (batch, input_len, hidden_dim)
        
        # Residual connection around temporal processing
        x = x + x_fused  # (batch, input_len, hidden_dim)
        
        # Cross-channel attention with proper dimension matching
        # Project down to n_features space for attention
        x_channel = self.channel_proj_down(x)  # (batch, input_len, n_features)
        # Apply attention: (batch, seq, n_features) -> self-attention over seq
        attn_out, _ = self.channel_attn(x_channel, x_channel, x_channel)  # (batch, input_len, n_features)
        # Project back to hidden_dim
        attn_out = self.channel_proj_up(attn_out)  # (batch, input_len, hidden_dim)
        # Residual with normalization
        x = self.channel_norm(x + attn_out)  # (batch, input_len, hidden_dim)
        
        # Depth-modulated MLP blocks with gating
        for block in self.mlp_blocks:
            residual = x  # (batch, input_len, hidden_dim)
            x = block['norm1'](x)  # (batch, input_len, hidden_dim)
            x = block['fc1'](x)  # (batch, input_len, hidden_dim*2)
            x = block['act'](x)  # (batch, input_len, hidden_dim*2)
            x = block['drop1'](x)  # (batch, input_len, hidden_dim*2)
            x = block['fc2'](x)  # (batch, input_len, hidden_dim)
            # Adaptive gating
            gate = block['gate'](x)  # (batch, input_len, hidden_dim)
            x = x * gate  # (batch, input_len, hidden_dim)
            x = block['drop2'](x)  # (batch, input_len, hidden_dim)
            x = residual + x  # (batch, input_len, hidden_dim)
        
        # Temporal aggregation: pool over input sequence
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        x = self.temporal_pool(x).squeeze(-1)  # (batch, hidden_dim)
        
        # Output projection with horizon scaling
        x = self.output_mlp(x)  # (batch, output_len * n_features)
        x = x.view(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
        x = x * self.horizon_scale  # (batch, output_len, n_features)
        
        # RevIN denormalization
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x  # (batch, output_len, n_features)

