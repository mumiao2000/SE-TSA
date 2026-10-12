import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NBeatsRevINEnhancedV2(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [64, 256, 1024],
            'n_stacks': [2, 4, 8],
            'n_blocks_per_stack': [1, 2, 4],
            'n_harmonics': [4, 16, 64],
            'n_polynomials': [2, 4, 8],
            'dropout': [0.0, 0.1, 0.3],
            'bottleneck_dim': [8, 32, 128],
            'n_frequency_bands': [2, 4, 8],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.n_stacks = self.config.get('n_stacks', 4)
        self.n_blocks_per_stack = self.config.get('n_blocks_per_stack', 2)
        self.n_harmonics = self.config.get('n_harmonics', 16)
        self.n_polynomials = self.config.get('n_polynomials', 4)
        self.dropout = self.config.get('dropout', 0.1)
        self.bottleneck_dim = self.config.get('bottleneck_dim', 32)
        self.n_frequency_bands = self.config.get('n_frequency_bands', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale frequency-adaptive harmonics
        self.frequency_bands = nn.Parameter(
            torch.linspace(0.5, self.n_harmonics, self.n_frequency_bands), requires_grad=True
        )  # (n_frequency_bands,)
        self.band_widths = nn.Parameter(
            torch.ones(self.n_frequency_bands) * 0.5, requires_grad=True
        )  # (n_frequency_bands,)
        self.band_weights = nn.Parameter(
            torch.ones(self.n_frequency_bands) / self.n_frequency_bands, requires_grad=True
        )  # (n_frequency_bands,)
        
        # Stack-wise cross-attention gating
        self.stack_attention = nn.ModuleList()
        for _ in range(self.n_stacks):
            self.stack_attention.append(nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid()
            ))
        
        # Build N-BEATS stacks
        self.stacks = nn.ModuleList()
        self.stack_types = []
        
        for s in range(self.n_stacks):
            if s % 2 == 0:
                stack_type = 'trend'
            else:
                stack_type = 'seasonality'
            self.stack_types.append(stack_type)
            
            blocks = nn.ModuleList()
            for b in range(self.n_blocks_per_stack):
                blocks.append(self._create_block(stack_type))
            self.stacks.append(blocks)
        
        # Residual gating weights
        self.residual_gates = nn.ParameterList([
            nn.Parameter(torch.ones(self.n_features, self.output_len) * 0.5)
            for _ in range(self.n_stacks)
        ])
        
        # Low-rank bilinear feature interaction
        self.feature_interaction_u = nn.Linear(self.n_features, self.bottleneck_dim, bias=False)
        self.feature_interaction_v = nn.Linear(self.n_features, self.bottleneck_dim, bias=False)
        self.feature_interaction_out = nn.Linear(self.bottleneck_dim, self.n_features, bias=False)
        
        # Learnable basis mixing per feature
        self.trend_weight = nn.Parameter(torch.ones(self.n_features, self.output_len) * 0.5)
        self.season_weight = nn.Parameter(torch.ones(self.n_features, self.output_len) * 0.5)
        
        # Output projection with uncertainty
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_shift = nn.Parameter(torch.zeros(1))
        
    def _create_block(self, stack_type):
        return NBeatsBlockV2(
            input_len=self.input_len,
            output_len=self.output_len,
            n_features=self.n_features,
            hidden_dim=self.hidden_dim,
            n_harmonics=self.n_harmonics,
            n_polynomials=self.n_polynomials,
            stack_type=stack_type,
            dropout=self.dropout,
            n_frequency_bands=self.n_frequency_bands
        )
    
    def _get_multi_scale_harmonics(self, t):
        # t: (output_len,)
        # frequency_bands: (n_frequency_bands,)
        # band_widths: (n_frequency_bands,)
        
        t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, output_len)
        bands = self.frequency_bands.view(-1, 1, 1)  # (n_frequency_bands, 1, 1)
        widths = F.softplus(self.band_widths).view(-1, 1, 1)  # (n_frequency_bands, 1, 1)
        
        # Gaussian frequency bands around center frequencies
        freq_grid = torch.arange(1, self.n_harmonics + 1, device=t.device).float()  # (n_harmonics,)
        freq_grid = freq_grid.view(1, -1, 1)  # (1, n_harmonics, 1)
        
        # Soft assignment to frequency bands
        band_logits = -((freq_grid - bands) ** 2) / (2 * widths ** 2 + 1e-6)  # (n_frequency_bands, n_harmonics, 1)
        band_weights = F.softmax(band_logits, dim=0)  # (n_frequency_bands, n_harmonics, 1)
        
        # Apply learned band importance
        global_band_weights = F.softmax(self.band_weights, dim=0)  # (n_frequency_bands,)
        band_weights = band_weights * global_band_weights.view(-1, 1, 1)  # (n_frequency_bands, n_harmonics, 1)
        
        # Generate harmonics with band-weighted frequencies
        harmonics = []
        for i in range(self.n_harmonics):
            # Effective frequency is weighted sum of band centers
            effective_freq = (band_weights[:, i, 0] * self.frequency_bands).sum()
            harmonics.append(torch.sin(2 * 3.14159265 * effective_freq * t))
            harmonics.append(torch.cos(2 * 3.14159265 * effective_freq * t))
        
        return torch.cat(harmonics, dim=1)  # (1, 2*n_harmonics, output_len)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-independent processing: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        batch_size = x.shape[0]
        
        # Initialize forecasts
        forecast = torch.zeros(batch_size, self.n_features, self.output_len, device=x.device)  # (batch, n_features, output_len)
        backcast = x.clone()  # (batch, n_features, input_len)
        
        # Stack outputs for cross-attention gating
        stack_outputs = []
        
        for s, (stack, stack_type) in enumerate(zip(self.stacks, self.stack_types)):
            stack_forecast = torch.zeros(batch_size, self.n_features, self.output_len, device=x.device)  # (batch, n_features, output_len)
            
            for block in stack:
                block_forecast, block_backcast = block(backcast, self._get_multi_scale_harmonics)  # (batch, n_features, output_len), (batch, n_features, input_len)
                stack_forecast = stack_forecast + block_forecast  # (batch, n_features, output_len)
                backcast = backcast - block_backcast  # (batch, n_features, input_len)
            
            # Stack-wise cross-attention gating
            if s > 0 and len(stack_outputs) > 0:
                # Aggregate previous stack outputs
                prev_stacks = torch.stack(stack_outputs, dim=-1).mean(dim=-1)  # (batch, n_features, output_len)
                
                # Create gating signal from current and previous
                gate_input = torch.cat([
                    stack_forecast.mean(dim=-1, keepdim=True).expand(-1, -1, self.output_len),  # (batch, n_features, output_len)
                    prev_stacks  # (batch, n_features, output_len)
                ], dim=1)  # (batch, 2*n_features, output_len)
                
                # Per-feature gating
                gate_input = gate_input.permute(0, 2, 1)  # (batch, output_len, 2*n_features)
                # Simplified: use mean across features for efficiency
                gate_input = gate_input.mean(dim=-1, keepdim=True).expand(-1, -1, self.hidden_dim * 2)  # (batch, output_len, hidden_dim*2)
                gate_input = gate_input.reshape(batch_size * self.output_len, self.hidden_dim * 2)  # (batch*output_len, hidden_dim*2)
                
                gate = self.stack_attention[s](gate_input)  # (batch*output_len, hidden_dim)
                gate = gate[:, 0].reshape(batch_size, self.output_len, 1).permute(0, 2, 1)  # (batch, 1, output_len)
                gate = gate.expand(-1, self.n_features, -1)  # (batch, n_features, output_len)
                
                # Apply gate to blend with residual
                residual = self.residual_gates[s].unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_features, output_len)
                stack_forecast = gate * stack_forecast + (1 - gate) * residual * stack_forecast  # (batch, n_features, output_len)
            
            stack_outputs.append(stack_forecast)
            
            # Learnable basis mixing
            if stack_type == 'trend':
                weight = torch.sigmoid(self.trend_weight).unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_features, output_len)
            else:
                weight = torch.sigmoid(self.season_weight).unsqueeze(0).expand(batch_size, -1, -1)  # (batch, n_features, output_len)
            
            forecast = forecast + weight * stack_forecast  # (batch, n_features, output_len)
        
        # Low-rank bilinear feature interaction
        # forecast: (batch, n_features, output_len)
        forecast_perm = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        u = self.feature_interaction_u(forecast_perm)  # (batch, output_len, bottleneck_dim)
        v = self.feature_interaction_v(forecast_perm)  # (batch, output_len, bottleneck_dim)
        
        # Bilinear interaction with low-rank constraint
        interaction = u * v  # (batch, output_len, bottleneck_dim)
        interaction = self.feature_interaction_out(interaction)  # (batch, output_len, n_features)
        
        # Residual connection with learned mixing
        mix_weight = torch.sigmoid(self.output_scale)  # scalar
        forecast_perm = forecast_perm + mix_weight * interaction  # (batch, output_len, n_features)
        
        # Final output
        forecast = forecast_perm.permute(0, 2, 1)  # (batch, n_features, output_len)
        forecast = forecast.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Apply output shift
        forecast = forecast + self.output_shift  # (batch, output_len, n_features)
        
        # Denormalize
        forecast = self.revin(forecast, mode='denorm')  # (batch, output_len, n_features)
        
        return forecast  # (batch, output_len, n_features)


class NBeatsBlockV2(nn.Module):
    def __init__(self, input_len, output_len, n_features, hidden_dim, n_harmonics, n_polynomials, 
                 stack_type, dropout, n_frequency_bands):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_harmonics = n_harmonics
        self.n_polynomials = n_polynomials
        self.stack_type = stack_type
        self.n_frequency_bands = n_frequency_bands
        
        # FC layers for backcast/forecast generation
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
        
        if stack_type == 'trend':
            self.basis_size = n_polynomials
            self.theta_f = nn.Linear(hidden_dim, n_polynomials)
            self.theta_b = nn.Linear(hidden_dim, n_polynomials)
            
            # Trend basis: polynomial functions
            t = torch.linspace(-1, 1, output_len)
            self.register_buffer('trend_t', t)
            
            t_back = torch.linspace(-1, 1, input_len)
            self.register_buffer('trend_t_back', t_back)
            
        else:  # seasonality
            self.basis_size = 2 * n_harmonics
            self.theta_f = nn.Linear(hidden_dim, 2 * n_harmonics)
            self.theta_b = nn.Linear(hidden_dim, 2 * n_harmonics)
            
            # Seasonality basis will be generated dynamically
            t = torch.arange(output_len, dtype=torch.float32)
            self.register_buffer('season_t', t)
            
            t_back = torch.arange(input_len, dtype=torch.float32)
            self.register_buffer('season_t_back', t_back)
    
    def forward(self, x, get_harmonics_fn):
        # x: (batch, n_features, input_len)
        batch_size = x.shape[0]
        
        # FC processing
        x_flat = x.reshape(batch_size * self.n_features, self.input_len)  # (batch*n_features, input_len)
        h = self.fc(x_flat)  # (batch*n_features, hidden_dim)
        
        # Generate basis coefficients
        theta_f = self.theta_f(h)  # (batch*n_features, basis_size)
        theta_b = self.theta_b(h)  # (batch*n_features, basis_size)
        
        if self.stack_type == 'trend':
            # Polynomial basis
            t = self.trend_t  # (output_len,)
            t_powers = torch.stack([t ** p for p in range(self.n_polynomials)], dim=0)  # (n_polynomials, output_len)
            forecast_basis = t_powers.T.unsqueeze(0)  # (1, output_len, n_polynomials)
            
            t_back = self.trend_t_back  # (input_len,)
            t_back_powers = torch.stack([t_back ** p for p in range(self.n_polynomials)], dim=0)  # (n_polynomials, input_len)
            backcast_basis = t_back_powers.T.unsqueeze(0)  # (1, input_len, n_polynomials)
            
        else:
            # Multi-scale harmonic basis
            harmonics = get_harmonics_fn(self.season_t)  # (1, 2*n_harmonics, output_len)
            forecast_basis = harmonics.permute(0, 2, 1)  # (1, output_len, 2*n_harmonics)
            
            # Use same frequencies for backcast
            harmonics_back = get_harmonics_fn(self.season_t_back)  # (1, 2*n_harmonics, input_len)
            backcast_basis = harmonics_back.permute(0, 2, 1)  # (1, input_len, 2*n_harmonics)
        
        # Apply basis
        forecast = torch.bmm(theta_f.unsqueeze(1), forecast_basis.expand(batch_size * self.n_features, -1, -1).permute(0, 2, 1))  # (batch*n_features, 1, output_len)
        forecast = forecast.squeeze(1).reshape(batch_size, self.n_features, self.output_len)  # (batch, n_features, output_len)
        
        backcast = torch.bmm(theta_b.unsqueeze(1), backcast_basis.expand(batch_size * self.n_features, -1, -1).permute(0, 2, 1))  # (batch*n_features, 1, input_len)
        backcast = backcast.squeeze(1).reshape(batch_size, self.n_features, self.input_len)  # (batch, n_features, input_len)
        
        return forecast, backcast
