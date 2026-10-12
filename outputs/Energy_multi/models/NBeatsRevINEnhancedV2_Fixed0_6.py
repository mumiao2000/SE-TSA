import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NBeatsRevINEnhancedV2_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_stacks': [2, 4, 8],
            'n_blocks_per_stack': [1, 2, 4],
            'trend_degrees': [2, 3, 5],
            'n_harmonics': [4, 8, 32],
            'dropout': [0.0, 0.1, 0.3],
            'use_multi_scale': [True, False],
            'use_hierarchical': [True, False],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Config extraction with defaults
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_stacks = self.config.get('n_stacks', 4)
        self.n_blocks_per_stack = self.config.get('n_blocks_per_stack', 2)
        self.trend_degrees = self.config.get('trend_degrees', 3)
        self.n_harmonics = self.config.get('n_harmonics', 8)
        self.dropout = self.config.get('dropout', 0.1)
        self.use_multi_scale = self.config.get('use_multi_scale', True)
        self.use_hierarchical = self.config.get('use_hierarchical', True)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale temporal convolution preprocessing
        if self.use_multi_scale:
            self.scale_kernels = [3, 5, 7]
            self.multi_scale_convs = nn.ModuleList([
                nn.Conv1d(self.n_features, self.hidden_dim // len(self.scale_kernels), 
                         kernel_size=k, padding=k//2, groups=1)
                for k in self.scale_kernels
            ])
            self.multi_scale_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.multi_scale_gate = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid()
            )
        
        # Hierarchical basis configurations
        if self.use_hierarchical:
            self.trend_pool = [1, 2, 3]  # polynomial degrees to choose from
            self.harmonic_pool = [4, 8, 16]  # harmonic counts to choose from
            self.hierarchical_router = nn.Sequential(
                nn.Linear(self.hidden_dim, len(self.trend_pool) + len(self.harmonic_pool)),
                nn.Softmax(dim=-1)
            )
        
        # Build N-BEATS stacks
        self.stacks = nn.ModuleList()
        self.stack_residual_gates = nn.ModuleList()
        
        for s in range(self.n_stacks):
            is_trend = (s % 2 == 0)
            stack_blocks = nn.ModuleList()
            
            for b in range(self.n_blocks_per_stack):
                if is_trend:
                    block = self._create_trend_block()
                else:
                    block = self._create_seasonality_block()
                stack_blocks.append(block)
            
            self.stacks.append(stack_blocks)
            
            # Learnable residual gate between stacks
            if s > 0:
                self.stack_residual_gates.append(nn.Sequential(
                    nn.Linear(self.n_features * self.output_len, self.n_features * self.output_len),
                    nn.Sigmoid()
                ))
        
        # Learnable basis mixing gates per feature
        self.basis_mix_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.n_features * self.output_len, self.n_features),
                nn.Sigmoid()
            )
            for _ in range(self.n_stacks)
        ])
        
        # Cross-feature attention for residual refinement
        self.cross_feature_attn = nn.MultiheadAttention(
            embed_dim=self.output_len, 
            num_heads=4, 
            batch_first=True,
            dropout=self.dropout
        )
        self.attn_norm = nn.LayerNorm(self.output_len)
        self.attn_proj = nn.Linear(self.output_len, self.output_len)
        
        # Feature mixing MLP after forecast generation
        self.feature_mixer = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.n_features)
        )
        
        # Final output projection
        self.output_proj = nn.Linear(self.n_features, self.n_features)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def _create_trend_block(self):
        return NBeatsTrendBlock(
            input_len=self.input_len,
            output_len=self.output_len,
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            degree=self.trend_degrees,
            dropout=self.dropout
        )
    
    def _create_seasonality_block(self):
        return NBeatsSeasonalityBlock(
            input_len=self.input_len,
            output_len=self.output_len,
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            n_harmonics=self.n_harmonics,
            dropout=self.dropout
        )
    
    def _get_hierarchical_basis(self, x_encoded):
        # x_encoded: (batch, hidden_dim)
        if not self.use_hierarchical:
            return self.trend_degrees, self.n_harmonics
        
        routing_weights = self.hierarchical_router(x_encoded)  # (batch, num_options)
        # Use mean routing for simplicity in forward pass
        trend_idx = routing_weights[:, :len(self.trend_pool)].mean(0).argmax()
        harm_idx = routing_weights[:, len(self.trend_pool):].mean(0).argmax()
        return self.trend_pool[trend_idx.item()], self.harmonic_pool[harm_idx.item()]
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]  # (batch, input_len, n_features)
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Multi-scale preprocessing
        if self.use_multi_scale:
            x_conv = x.permute(0, 2, 1)  # (batch, n_features, input_len)
            multi_scale_feats = []
            for conv in self.multi_scale_convs:
                ms_out = conv(x_conv)  # (batch, hidden_dim//3, input_len)
                multi_scale_feats.append(ms_out)
            multi_scale_feats = torch.cat(multi_scale_feats, dim=1)  # (batch, hidden_dim, input_len)
            multi_scale_feats = multi_scale_feats.permute(0, 2, 1)  # (batch, input_len, hidden_dim)
            multi_scale_feats = multi_scale_feats.mean(dim=1)  # (batch, hidden_dim)
            
            gate = self.multi_scale_gate(multi_scale_feats)  # (batch, hidden_dim)
            multi_scale_feats = self.multi_scale_proj(multi_scale_feats) * gate  # (batch, hidden_dim)
        else:
            multi_scale_feats = None
        
        # Initialize forecast and backcast
        forecast = torch.zeros(batch_size, self.output_len, self.n_features, device=x.device)  # (batch, output_len, n_features)
        backcast = x  # (batch, input_len, n_features)
        
        # Process through stacks
        for s, stack_blocks in enumerate(self.stacks):
            stack_forecast = torch.zeros(batch_size, self.output_len, self.n_features, device=x.device)  # (batch, output_len, n_features)
            
            for block in stack_blocks:
                block_forecast, block_backcast = block(backcast)  # (batch, output_len, n_features), (batch, input_len, n_features)
                backcast = backcast - block_backcast  # (batch, input_len, n_features)
                stack_forecast = stack_forecast + block_forecast  # (batch, output_len, n_features)
            
            # Learnable basis mixing gate
            gate_input = stack_forecast.reshape(batch_size, -1)  # (batch, output_len * n_features)
            mix_gate = self.basis_mix_gates[s](gate_input)  # (batch, n_features)
            mix_gate = mix_gate.unsqueeze(1)  # (batch, 1, n_features)
            stack_forecast = stack_forecast * mix_gate  # (batch, output_len, n_features)
            
            # Residual connection between stacks with learnable weight
            if s > 0:
                residual_input = forecast.reshape(batch_size, -1)  # (batch, output_len * n_features)
                residual_gate = self.stack_residual_gates[s-1](residual_input)  # (batch, output_len * n_features)
                residual_gate = residual_gate.reshape(batch_size, self.output_len, self.n_features)  # (batch, output_len, n_features)
                forecast = forecast * residual_gate + stack_forecast  # (batch, output_len, n_features)
            else:
                forecast = stack_forecast  # (batch, output_len, n_features)
        
        # Cross-feature attention on forecast residuals
        # Treat features as sequence, time steps as features for attention
        attn_input = forecast.permute(0, 2, 1)  # (batch, n_features, output_len)
        attn_out, _ = self.cross_feature_attn(attn_input, attn_input, attn_input)  # (batch, n_features, output_len)
        attn_out = self.attn_norm(attn_input + attn_out)  # (batch, n_features, output_len)
        attn_out = self.attn_proj(attn_out)  # (batch, n_features, output_len)
        forecast = forecast + 0.1 * attn_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Feature mixing MLP
        forecast = self.feature_mixer(forecast)  # (batch, output_len, n_features)
        
        # Final projection
        forecast = self.output_proj(forecast)  # (batch, output_len, n_features)
        
        # RevIN denormalization
        forecast = self.revin(forecast, mode='denorm')  # (batch, output_len, n_features)
        
        return forecast  # (batch, output_len, n_features)


class NBeatsTrendBlock(nn.Module):
    def __init__(self, input_len, output_len, n_features, hidden_dim, degree, dropout):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.degree = degree
        
        # FC stack
        self.fc = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # FIX: Output degree+1, not (degree+1) * n_features, since we process flattened features
        self.theta_f = nn.Linear(hidden_dim, degree + 1)
        self.theta_b = nn.Linear(hidden_dim, degree + 1)
        
        # Trend basis
        self.register_buffer('trend_t', torch.linspace(0, 1, output_len))
        self.register_buffer('backcast_t', torch.linspace(0, 1, input_len))
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Process each feature independently (channel-independent)
        x_flat = x.permute(0, 2, 1).reshape(batch_size * self.n_features, self.input_len)  # (batch*n_features, input_len)
        
        h = self.fc(x_flat)  # (batch*n_features, hidden_dim)
        
        theta_f = self.theta_f(h)  # (batch*n_features, degree+1)
        theta_b = self.theta_b(h)  # (batch*n_features, degree+1)
        
        # Generate trend basis
        trend_basis = torch.stack([self.trend_t ** i for i in range(self.degree + 1)], dim=1)  # (output_len, degree+1)
        backcast_basis = torch.stack([self.backcast_t ** i for i in range(self.degree + 1)], dim=1)  # (input_len, degree+1)
        
        forecast = torch.matmul(theta_f, trend_basis.T)  # (batch*n_features, output_len)
        backcast = torch.matmul(theta_b, backcast_basis.T)  # (batch*n_features, input_len)
        
        forecast = forecast.reshape(batch_size, self.n_features, self.output_len).permute(0, 2, 1)  # (batch, output_len, n_features)
        backcast = backcast.reshape(batch_size, self.n_features, self.input_len).permute(0, 2, 1)  # (batch, input_len, n_features)
        
        return forecast, backcast  # (batch, output_len, n_features), (batch, input_len, n_features)


class NBeatsSeasonalityBlock(nn.Module):
    def __init__(self, input_len, output_len, n_features, hidden_dim, n_harmonics, dropout):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.n_harmonics = n_harmonics
        
        # FC stack
        self.fc = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # FIX: Output 2*n_harmonics, not 2*n_harmonics*n_features, since we process flattened features
        self.theta_f = nn.Linear(hidden_dim, 2 * n_harmonics)
        self.theta_b = nn.Linear(hidden_dim, 2 * n_harmonics)
        
        # Seasonality basis (Fourier)
        self.register_buffer('cos_basis_f', torch.zeros(output_len, n_harmonics))
        self.register_buffer('sin_basis_f', torch.zeros(output_len, n_harmonics))
        self.register_buffer('cos_basis_b', torch.zeros(input_len, n_harmonics))
        self.register_buffer('sin_basis_b', torch.zeros(input_len, n_harmonics))
        
        for i in range(1, n_harmonics + 1):
            self.cos_basis_f[:, i-1] = torch.cos(2 * torch.pi * i * torch.linspace(0, 1, output_len))
            self.sin_basis_f[:, i-1] = torch.sin(2 * torch.pi * i * torch.linspace(0, 1, output_len))
            self.cos_basis_b[:, i-1] = torch.cos(2 * torch.pi * i * torch.linspace(0, 1, input_len))
            self.sin_basis_b[:, i-1] = torch.sin(2 * torch.pi * i * torch.linspace(0, 1, input_len))
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # Process each feature independently
        x_flat = x.permute(0, 2, 1).reshape(batch_size * self.n_features, self.input_len)  # (batch*n_features, input_len)
        
        h = self.fc(x_flat)  # (batch*n_features, hidden_dim)
        
        theta_f = self.theta_f(h)  # (batch*n_features, 2*n_harmonics)
        theta_b = self.theta_b(h)  # (batch*n_features, 2*n_harmonics)
        
        # Split into cosine and sine coefficients
        cos_coef_f = theta_f[:, :self.n_harmonics]  # (batch*n_features, n_harmonics)
        sin_coef_f = theta_f[:, self.n_harmonics:]  # (batch*n_features, n_harmonics)
        cos_coef_b = theta_b[:, :self.n_harmonics]  # (batch*n_features, n_harmonics)
        sin_coef_b = theta_b[:, self.n_harmonics:]  # (batch*n_features, n_harmonics)
        
        # Generate seasonality via Fourier basis
        forecast = torch.matmul(cos_coef_f, self.cos_basis_f.T) + torch.matmul(sin_coef_f, self.sin_basis_f.T)  # (batch*n_features, output_len)
        backcast = torch.matmul(cos_coef_b, self.cos_basis_b.T) + torch.matmul(sin_coef_b, self.sin_basis_b.T)  # (batch*n_features, input_len)
        
        forecast = forecast.reshape(batch_size, self.n_features, self.output_len).permute(0, 2, 1)  # (batch, output_len, n_features)
        backcast = backcast.reshape(batch_size, self.n_features, self.input_len).permute(0, 2, 1)  # (batch, input_len, n_features)
        
        return forecast, backcast  # (batch, output_len, n_features), (batch, input_len, n_features)
