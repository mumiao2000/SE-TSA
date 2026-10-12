import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class DilatedConvGatedMLP_Improved1_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_layers': [2, 4, 8],
            'dropout': [0.0, 0.1, 0.3],
            'd_ff': [128, 512, 2048],
            'n_freq_bands': [4, 8, 16],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.d_model = cfg.get('d_model', 256)
        self.n_layers = cfg.get('n_layers', 4)
        self.dropout = cfg.get('dropout', 0.1)
        self.d_ff = cfg.get('d_ff', 512)
        self.n_freq_bands = cfg.get('n_freq_bands', 8)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Input projection: per-feature independent linear + shared transformation
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, self.d_model // 4),
                nn.GELU(),
                nn.Linear(self.d_model // 4, self.d_model)
            ) for _ in range(self.n_features)
        ])
        
        # Channel fusion: mix information across features
        self.channel_fusion = nn.Sequential(
            nn.Linear(self.n_features * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.n_features * self.d_model)
        )
        
        # Dilated causal convolution blocks with channel-dependent gates
        self.conv_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for i in range(self.n_layers):
            dilation = 2 ** i
            # Causal padding: (kernel_size - 1) * dilation
            kernel_size = 3
            causal_padding = (kernel_size - 1) * dilation
            
            # Depthwise separable conv with proper causal padding
            conv = nn.Sequential(
                nn.ConstantPad1d((causal_padding, 0), 0),  # left pad only for causality
                nn.Conv1d(self.d_model, self.d_model, kernel_size, dilation=dilation, groups=self.d_model),
                nn.Conv1d(self.d_model, self.d_model, 1),  # pointwise
            )
            self.conv_layers.append(conv)
            
            # Channel-dependent gating: independent gate per feature
            self.gate_layers.append(nn.Sequential(
                nn.Linear(self.d_model, self.d_ff),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_ff, self.d_model),
                nn.Sigmoid()
            ))
            
            self.norm_layers.append(nn.LayerNorm(self.d_model))
        
        # Frequency-aware routing for periodic patterns (exploit 370-day seasonality)
        self.freq_proj = nn.Linear(self.d_model, self.n_freq_bands)
        self.freq_weights = nn.Parameter(torch.randn(self.n_freq_bands, self.d_model, self.d_model) * 0.02)
        self.freq_bias = nn.Parameter(torch.zeros(self.n_freq_bands, self.d_model))
        
        # Period embeddings for known seasonality
        max_period = 512  # covers 370-day period
        self.period_embed = nn.Embedding(max_period, self.d_model // 4)
        
        # Output projection
        self.output_norm = nn.LayerNorm(self.d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_ff, 1)
        )
        
        # Direct residual connection from input to output
        self.residual_proj = nn.Linear(input_len, output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Store for residual
        x_raw = x  # (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Per-feature input projection
        x_proj = []
        for f in range(self.n_features):
            xf = x[:, :, f:f+1]  # (batch, input_len, 1)
            xf = self.input_proj[f](xf)  # (batch, input_len, d_model)
            x_proj.append(xf)
        
        # Stack: (batch, input_len, n_features, d_model)
        x = torch.stack(x_proj, dim=2)  # (batch, input_len, n_features, d_model)
        
        # Reshape for processing: (batch * n_features, input_len, d_model)
        x = x.reshape(batch_size * self.n_features, self.input_len, self.d_model)  # (batch * n_features, input_len, d_model)
        
        # Dilated causal convolutions with channel-dependent gating
        for i in range(self.n_layers):
            residual = x  # (batch * n_features, input_len, d_model)
            
            # Conv on temporal dimension
            x_conv = x.permute(0, 2, 1)  # (batch * n_features, d_model, input_len)
            x_conv = self.conv_layers[i](x_conv)  # (batch * n_features, d_model, input_len)
            x_conv = x_conv[:, :, :self.input_len]  # trim to original length  # (batch * n_features, d_model, input_len)
            x_conv = x_conv.permute(0, 2, 1)  # (batch * n_features, input_len, d_model)
            
            # Channel-dependent gating
            gate = self.gate_layers[i](x_conv)  # (batch * n_features, input_len, d_model)
            x = x_conv * gate  # (batch * n_features, input_len, d_model)
            x = self.norm_layers[i](x + residual)  # (batch * n_features, input_len, d_model)
            x = F.dropout(x, p=self.dropout, training=self.training)  # (batch * n_features, input_len, d_model)
        
        # Reshape back: (batch, input_len, n_features, d_model)
        x = x.reshape(batch_size, self.input_len, self.n_features, self.d_model)  # (batch, input_len, n_features, d_model)
        
        # Channel fusion across features
        x_fusion = x.reshape(batch_size, self.input_len, self.n_features * self.d_model)  # (batch, input_len, n_features * d_model)
        x_fusion = self.channel_fusion(x_fusion)  # (batch, input_len, n_features * d_model)
        x_fusion = x_fusion.reshape(batch_size, self.input_len, self.n_features, self.d_model)  # (batch, input_len, n_features, d_model)
        x = x + x_fusion  # (batch, input_len, n_features, d_model)
        
        # Frequency-aware routing
        # Compute frequency band weights from temporal average
        x_temp = x.mean(dim=1)  # (batch, n_features, d_model)
        freq_logits = self.freq_proj(x_temp)  # (batch, n_features, n_freq_bands)
        freq_attn = F.softmax(freq_logits, dim=-1)  # (batch, n_features, n_freq_bands)
        
        # Apply weighted frequency transformations
        x_freq = torch.einsum('bnf,fdm->bndm', freq_attn, self.freq_weights)  # (batch, n_features, d_model, d_model)
        x = torch.einsum('btnd,bndm->btnm', x, x_freq)  # (batch, input_len, n_features, d_model)
        
        # FIX: Compute per-feature bias by weighting freq_bias with freq_attn
        # freq_attn: (batch, n_features, n_freq_bands), freq_bias: (n_freq_bands, d_model)
        # Result: (batch, n_features, d_model), then broadcast across time
        freq_bias_per_feature = torch.einsum('bnf,fd->bnd', freq_attn, self.freq_bias)  # (batch, n_features, d_model)
        x = x + freq_bias_per_feature.unsqueeze(1)  # (batch, input_len, n_features, d_model)
        
        # Add period-aware positional encoding
        positions = torch.arange(self.input_len, device=x.device) % 370  # exploit known period
        period_emb = self.period_embed(positions)  # (input_len, d_model // 4)
        period_emb = F.pad(period_emb, (0, self.d_model - self.d_model // 4))  # (input_len, d_model)
        x = x + period_emb.unsqueeze(0).unsqueeze(2)  # (batch, input_len, n_features, d_model)
        
        # Output projection
        x = self.output_norm(x)  # (batch, input_len, n_features, d_model)
        x = self.output_proj(x).squeeze(-1)  # (batch, input_len, n_features)
        
        # Project to output length
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x = self.residual_proj(x)  # (batch, n_features, output_len)
        x = x.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Add direct residual from raw input (trend preservation)
        x_resid = x_raw.mean(dim=1, keepdim=True).expand(-1, self.output_len, -1)  # (batch, output_len, n_features)
        x = x + 0.1 * x_resid  # (batch, output_len, n_features)
        
        # RevIN denormalization
        x = self.revin(x, mode='denorm')  # (batch, output_len, n_features)
        
        return x  # (batch, output_len, n_features)

