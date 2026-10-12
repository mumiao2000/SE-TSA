import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossVariableAttentionMLP(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 512],
            'n_heads': [2, 4, 8],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 4)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Project input to d_model per feature (treat time as feature dim)
        self.input_proj = nn.Linear(input_len, self.d_model)
        
        # Cross-variable attention: features attend to each other
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Layer norm on feature dimension
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        
        # Feedforward per feature
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        
        # Output MLP: d_model -> output_len per feature
        self.output_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, output_len)
        )
        
    def cross_variable_attention(self, x):
        # x: (batch, n_features, d_model)
        batch_size, n_features, d_model = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x)  # (batch, n_features, d_model)
        k = self.k_proj(x)  # (batch, n_features, d_model)
        v = self.v_proj(x)  # (batch, n_features, d_model)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, n_features, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_features, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_features, self.n_heads, self.head_dim).transpose(1, 2)
        # Now: (batch, n_heads, n_features, head_dim)
        
        # Attention: features attend to features
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # (batch, n_heads, n_features, n_features)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # (batch, n_heads, n_features, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_features, d_model)
        
        out = self.out_proj(out)  # (batch, n_features, d_model)
        return out
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Project to d_model
        h = self.input_proj(x)  # (batch, n_features, d_model)
        
        # Cross-variable attention with residual
        attn_out = self.cross_variable_attention(h)  # (batch, n_features, d_model)
        h = self.norm1(h + self.dropout(attn_out))  # (batch, n_features, d_model)
        
        # FFN with residual
        ffn_out = self.ffn(h)  # (batch, n_features, d_model)
        h = self.norm2(h + self.dropout(ffn_out))  # (batch, n_features, d_model)
        
        # Output projection to time domain
        out = self.output_mlp(h)  # (batch, n_features, output_len)
        
        # Transpose back
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
