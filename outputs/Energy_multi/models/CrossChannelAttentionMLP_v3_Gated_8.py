import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossChannelAttentionMLP_v3_Gated(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_heads': [2, 8, 32],
            'n_layers': [1, 4, 8],
            'dropout': [0.0, 0.3],
            'dilation_base': [2, 4, 8],
            'n_scales': [2, 3, 4],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.hidden_dim = cfg.get('hidden_dim', 512)
        self.n_heads = cfg.get('n_heads', 8)
        self.n_layers = cfg.get('n_layers', 4)
        self.dropout = cfg.get('dropout', 0.1)
        self.dilation_base = cfg.get('dilation_base', 4)
        self.n_scales = cfg.get('n_scales', 3)
        
        # Ensure hidden_dim divisible by n_heads
        self.head_dim = max(1, self.hidden_dim // self.n_heads)
        self.hidden_dim = self.head_dim * self.n_heads
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection
        self.input_proj = nn.Linear(self.n_features, self.hidden_dim)
        
        # Multi-scale dilated temporal convolutions
        self.scale_convs = nn.ModuleList()
        self.scale_gates = nn.ModuleList()
        for i in range(self.n_scales):
            dilation = self.dilation_base ** i
            kernel_size = 3
            # Ensure kernel with dilation fits in input_len
            effective_kernel = (kernel_size - 1) * dilation + 1
            if effective_kernel <= self.input_len:
                conv = nn.Conv1d(
                    self.hidden_dim, 
                    self.hidden_dim // self.n_scales,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation // 2  # causal-ish padding
                )
                gate = nn.Sequential(
                    nn.Linear(self.hidden_dim // self.n_scales, self.hidden_dim // self.n_scales),
                    nn.Sigmoid()
                )
                self.scale_convs.append(conv)
                self.scale_gates.append(gate)
        
        self.actual_n_scales = len(self.scale_convs)
        self.scale_fusion = nn.Linear(self.hidden_dim // self.n_scales * self.actual_n_scales, self.hidden_dim)
        
        # Channel importance allocator (lightweight)
        self.channel_allocator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # global temporal pooling
            nn.Flatten(),
            nn.Linear(self.hidden_dim, self.n_features),
            nn.Softmax(dim=-1)
        )
        
        # Cross-channel attention blocks
        self.attn_layers = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.attn_mlps = nn.ModuleList()
        self.attn_mlp_norms = nn.ModuleList()
        
        for _ in range(self.n_layers):
            # Pre-norm for attention
            self.attn_norms.append(nn.LayerNorm(self.hidden_dim))
            # Cross-channel attention: attend over channels with n_features as embed_dim
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=self.hidden_dim,
                    num_heads=self.n_heads,
                    dropout=self.dropout,
                    batch_first=True
                )
            )
            # Post-attention MLP mixer
            self.attn_mlp_norms.append(nn.LayerNorm(self.hidden_dim))
            mlp_hidden = self.hidden_dim * 4
            self.attn_mlps.append(nn.Sequential(
                nn.Linear(self.hidden_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(mlp_hidden, self.hidden_dim),
                nn.Dropout(self.dropout)
            ))
        
        # Temporal aggregation for forecasting
        self.temporal_pool = nn.AdaptiveAvgPool1d(self.output_len)
        
        # Horizon-aware MLP decoder with residual connections
        decoder_layers = []
        decoder_dim = self.hidden_dim
        for i in range(2):
            decoder_layers.append(nn.Linear(decoder_dim, decoder_dim * 2))
            decoder_layers.append(nn.LayerNorm(decoder_dim * 2))
            decoder_layers.append(nn.GELU())
            decoder_layers.append(nn.Dropout(self.dropout))
            decoder_layers.append(nn.Linear(decoder_dim * 2, decoder_dim))
            decoder_layers.append(nn.LayerNorm(decoder_dim))
            if i < 1:
                decoder_layers.append(nn.GELU())
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_features)
        
        # Learnable horizon scaling
        self.horizon_scale = nn.Parameter(torch.ones(1, self.output_len, 1))
        self.horizon_shift = nn.Parameter(torch.zeros(1, self.output_len, 1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection
        h = self.input_proj(x)  # (batch, input_len, hidden_dim)
        
        # Multi-scale dilated convolutions
        h_perm = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        scale_outputs = []
        for i, (conv, gate) in enumerate(zip(self.scale_convs, self.scale_gates)):
            conv_out = conv(h_perm)  # (batch, hidden_dim//n_scales, input_len)
            conv_out = conv_out.permute(0, 2, 1)  # (batch, input_len, hidden_dim//n_scales)
            # Apply gating
            gate_val = gate(conv_out)  # (batch, input_len, hidden_dim//n_scales)
            conv_out = conv_out * gate_val  # (batch, input_len, hidden_dim//n_scales)
            scale_outputs.append(conv_out)
        
        # Fuse multi-scale features
        if len(scale_outputs) > 0:
            fused = torch.cat(scale_outputs, dim=-1)  # (batch, input_len, hidden_dim//n_scales * actual_n_scales)
            h = self.scale_fusion(fused)  # (batch, input_len, hidden_dim)
        
        # Channel importance allocation
        h_for_alloc = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        channel_weights = self.channel_allocator(h_for_alloc)  # (batch, n_features)
        channel_weights = channel_weights.unsqueeze(1)  # (batch, 1, n_features)
        
        # Cross-channel attention layers
        for norm, attn, mlp_norm, mlp in zip(
            self.attn_norms, self.attn_layers, 
            self.attn_mlp_norms, self.attn_mlps
        ):
            # Pre-norm residual attention
            h_norm = norm(h)  # (batch, input_len, hidden_dim)
            # For cross-channel: we want to attend across feature-like dimensions
            # Project to query space that allows channel interaction
            attn_out, _ = attn(h_norm, h_norm, h_norm)  # (batch, input_len, hidden_dim)
            h = h + attn_out  # residual, (batch, input_len, hidden_dim)
            
            # Pre-norm residual MLP
            h_norm = mlp_norm(h)  # (batch, input_len, hidden_dim)
            mlp_out = mlp(h_norm)  # (batch, input_len, hidden_dim)
            h = h + mlp_out  # residual, (batch, input_len, hidden_dim)
        
        # Temporal pooling to output length
        h = h.permute(0, 2, 1)  # (batch, hidden_dim, input_len)
        h = self.temporal_pool(h)  # (batch, hidden_dim, output_len)
        h = h.permute(0, 2, 1)  # (batch, output_len, hidden_dim)
        
        # Decoder with residual
        h_decoded = self.decoder(h)  # (batch, output_len, hidden_dim)
        h = h + h_decoded  # residual, (batch, output_len, hidden_dim)
        
        # Output projection
        out = self.output_proj(h)  # (batch, output_len, n_features)
        
        # Horizon-aware scaling
        out = out * self.horizon_scale + self.horizon_shift  # (batch, output_len, n_features)
        
        # Apply channel importance weighting (broadcast across time)
        out = out * channel_weights  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out  # (batch, output_len, n_features)
