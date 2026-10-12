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
            'dropout': [0.0, 0.3],
            'trend_degree': [2, 3, 5],
            'n_harmonics': [4, 8, 32],
            'use_temporal_attention': [True, False],
            'feature_interaction_rank': [4, 16, 64],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.hidden_dim = cfg.get('hidden_dim', 256)
        self.n_stacks = cfg.get('n_stacks', 4)
        self.n_blocks_per_stack = cfg.get('n_blocks_per_stack', 2)
        self.dropout = cfg.get('dropout', 0.1)
        self.trend_degree = cfg.get('trend_degree', 3)
        self.n_harmonics = cfg.get('n_harmonics', 8)
        self.use_temporal_attention = cfg.get('use_temporal_attention', True)
        self.feature_interaction_rank = cfg.get('feature_interaction_rank', 16)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Horizon-adaptive basis sizing
        self.adaptive_trend_degree = min(self.trend_degree, max(2, self.output_len // 6))
        self.adaptive_n_harmonics = min(self.n_harmonics, max(4, self.output_len // 2))
        
        # Stack-wise temporal attention (optional, lightweight)
        if self.use_temporal_attention:
            self.temporal_attn = nn.ModuleList()
            for _ in range(self.n_stacks):
                self.temporal_attn.append(
                    nn.Sequential(
                        nn.Linear(self.input_len, self.input_len // 4),
                        nn.ReLU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.input_len // 4, self.input_len),
                        nn.Sigmoid()
                    )
                )
        
        # N-BEATS stacks
        self.stacks = nn.ModuleList()
        for s in range(self.n_stacks):
            # Alternate between trend and seasonality stacks
            is_trend = (s % 2 == 0)
            stack_type = 'trend' if is_trend else 'seasonality'
            n_blocks = self.n_blocks_per_stack
            
            blocks = nn.ModuleList()
            for b in range(n_blocks):
                blocks.append(
                    NBeatsBlockV2(
                        input_len=self.input_len,
                        output_len=self.output_len,
                        hidden_dim=self.hidden_dim,
                        stack_type=stack_type,
                        trend_degree=self.adaptive_trend_degree,
                        n_harmonics=self.adaptive_n_harmonics,
                        dropout=self.dropout,
                        use_temporal_attention=self.use_temporal_attention
                    )
                )
            self.stacks.append(blocks)
        
        # Stack output gating (learnable mixing)
        self.stack_gates = nn.Parameter(torch.ones(self.n_stacks, self.n_features, self.output_len))
        
        # Residual connection weights between stacks
        self.residual_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(self.n_stacks - 1)
        ])
        
        # Learnable stack combination weights
        self.stack_combine = nn.Parameter(torch.ones(self.n_stacks) / self.n_stacks)
        
        # Dynamic low-rank feature interaction
        self.feature_interaction = DynamicFeatureInteraction(
            n_features=self.n_features,
            output_len=self.output_len,
            rank=self.feature_interaction_rank,
            dropout=self.dropout
        )
        
        # Final output projection (optional refinement)
        self.output_refine = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_features, self.n_features)
        )
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalize
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for channel-independent processing: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Process each feature independently through stacks
        stack_outputs = []
        current = x  # (batch, n_features, input_len)
        
        for s, blocks in enumerate(self.stacks):
            stack_input = current  # (batch, n_features, input_len)
            
            # Apply temporal attention gating if enabled
            if self.use_temporal_attention:
                attn_gate = self.temporal_attn[s](stack_input.mean(dim=1, keepdim=True))  # (batch, 1, input_len)
                stack_input = stack_input * attn_gate  # (batch, n_features, input_len)
            
            # Process through blocks in stack
            stack_forecast = None
            stack_backcast = None
            
            for b, block in enumerate(blocks):
                if b == 0:
                    forecast, backcast = block(stack_input)  # (batch, n_features, output_len), (batch, n_features, input_len)
                else:
                    forecast, backcast = block(backcast)  # (batch, n_features, output_len), (batch, n_features, input_len)
                
                if stack_forecast is None:
                    stack_forecast = forecast  # (batch, n_features, output_len)
                else:
                    stack_forecast = stack_forecast + forecast  # (batch, n_features, output_len)
            
            # Apply stack gate
            gate = torch.sigmoid(self.stack_gates[s:s+1])  # (1, n_features, output_len)
            stack_forecast = stack_forecast * gate  # (batch, n_features, output_len)
            
            stack_outputs.append(stack_forecast)  # (batch, n_features, output_len)
            
            # Residual update for next stack
            if s < len(self.stacks) - 1:
                residual_weight = torch.sigmoid(self.residual_weights[s])  # scalar
                current = stack_input - residual_weight * backcast  # (batch, n_features, input_len)
        
        # Combine stack outputs with learned weights
        combined = torch.stack(stack_outputs, dim=0)  # (n_stacks, batch, n_features, output_len)
        stack_weights = F.softmax(self.stack_combine, dim=0).view(self.n_stacks, 1, 1, 1)  # (n_stacks, 1, 1, 1)
        forecast = (combined * stack_weights).sum(dim=0)  # (batch, n_features, output_len)
        
        # Transpose back: (batch, output_len, n_features)
        forecast = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Dynamic feature interaction
        forecast = self.feature_interaction(forecast)  # (batch, output_len, n_features)
        
        # Output refinement
        forecast = self.output_refine(forecast)  # (batch, output_len, n_features)
        
        # RevIN denormalize
        forecast = self.revin(forecast, mode='denorm')  # (batch, output_len, n_features)
        
        return forecast  # (batch, output_len, n_features)


class NBeatsBlockV2(nn.Module):
    def __init__(self, input_len, output_len, hidden_dim, stack_type, trend_degree, n_harmonics, dropout, use_temporal_attention):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.stack_type = stack_type
        self.trend_degree = trend_degree
        self.n_harmonics = n_harmonics
        self.use_temporal_attention = use_temporal_attention
        
        # FC stack for theta generation
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
        
        # Theta generation
        if stack_type == 'trend':
            self.theta_size = trend_degree + 1
            self.theta_f = nn.Linear(hidden_dim, self.theta_size)
            self.theta_b = nn.Linear(hidden_dim, self.theta_size)
            # Trend basis
            t = torch.linspace(0, 1, output_len)
            self.register_buffer('trend_basis', torch.stack([t ** i for i in range(self.theta_size)], dim=0))  # (theta_size, output_len)
            t_back = torch.linspace(0, 1, input_len)
            self.register_buffer('trend_basis_back', torch.stack([t_back ** i for i in range(self.theta_size)], dim=0))  # (theta_size, input_len)
        else:
            self.theta_size = 2 * n_harmonics
            self.theta_f = nn.Linear(hidden_dim, self.theta_size)
            self.theta_b = nn.Linear(hidden_dim, self.theta_size)
            # Seasonality basis (Fourier)
            t = torch.arange(output_len).float()
            harmonics = []
            for i in range(1, n_harmonics + 1):
                harmonics.append(torch.cos(2 * torch.pi * i * t / output_len))
                harmonics.append(torch.sin(2 * torch.pi * i * t / output_len))
            self.register_buffer('season_basis', torch.stack(harmonics, dim=0))  # (2*n_harmonics, output_len)
            
            t_back = torch.arange(input_len).float()
            harmonics_back = []
            for i in range(1, n_harmonics + 1):
                harmonics_back.append(torch.cos(2 * torch.pi * i * t_back / input_len))
                harmonics_back.append(torch.sin(2 * torch.pi * i * t_back / input_len))
            self.register_buffer('season_basis_back', torch.stack(harmonics_back, dim=0))  # (2*n_harmonics, input_len)
    
    def forward(self, x):
        # x: (batch, n_features, input_len)
        batch_size, n_features, _ = x.shape
        
        # Flatten features for shared processing
        x_flat = x.reshape(batch_size * n_features, self.input_len)  # (batch*n_features, input_len)
        
        # FC stack
        h = self.fc(x_flat)  # (batch*n_features, hidden_dim)
        
        # Theta
        theta_f = self.theta_f(h)  # (batch*n_features, theta_size)
        theta_b = self.theta_b(h)  # (batch*n_features, theta_size)
        
        # Generate forecast and backcast
        if self.stack_type == 'trend':
            forecast = torch.matmul(theta_f, self.trend_basis)  # (batch*n_features, output_len)
            backcast = torch.matmul(theta_b, self.trend_basis_back)  # (batch*n_features, input_len)
        else:
            forecast = torch.matmul(theta_f, self.season_basis)  # (batch*n_features, output_len)
            backcast = torch.matmul(theta_b, self.season_basis_back)  # (batch*n_features, input_len)
        
        # Reshape back
        forecast = forecast.view(batch_size, n_features, self.output_len)  # (batch, n_features, output_len)
        backcast = backcast.view(batch_size, n_features, self.input_len)  # (batch, n_features, input_len)
        
        return forecast, backcast


class DynamicFeatureInteraction(nn.Module):
    def __init__(self, n_features, output_len, rank, dropout):
        super().__init__()
        self.n_features = n_features
        self.output_len = output_len
        self.rank = rank
        
        # Dynamic low-rank factorization: U @ diag(gate) @ V^T
        self.U = nn.Linear(n_features, rank, bias=False)
        self.V = nn.Linear(rank, n_features, bias=False)
        
        # Gate generation from input statistics
        self.gate_gen = nn.Sequential(
            nn.Linear(n_features * 2, rank),  # mean and std across time
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rank, rank),
            nn.Sigmoid()
        )
        
        # Bypass connection
        self.bypass_weight = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # x: (batch, output_len, n_features)
        batch_size = x.shape[0]
        
        # Compute feature statistics for dynamic gating
        x_mean = x.mean(dim=1)  # (batch, n_features)
        x_std = x.std(dim=1)  # (batch, n_features)
        stats = torch.cat([x_mean, x_std], dim=-1)  # (batch, n_features*2)
        
        # Generate dynamic gate
        gate = self.gate_gen(stats)  # (batch, rank)
        
        # Apply low-rank interaction per timestep
        out = []
        for t in range(self.output_len):
            xt = x[:, t, :]  # (batch, n_features)
            # Low-rank transform with dynamic gating
            h = self.U(xt)  # (batch, rank)
            h = h * gate  # (batch, rank) - dynamic modulation
            yt = self.V(h)  # (batch, n_features)
            out.append(yt)
        
        out = torch.stack(out, dim=1)  # (batch, output_len, n_features)
        
        # Adaptive bypass: start with near-zero interaction, learn to use it
        bypass = torch.sigmoid(self.bypass_weight)
        return bypass * out + (1 - bypass) * x  # (batch, output_len, n_features)
