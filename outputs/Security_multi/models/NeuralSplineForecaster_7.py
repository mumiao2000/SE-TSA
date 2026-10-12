import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class NeuralSplineForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_knots': [8, 32, 128],
            'hidden_dim': [64, 256, 1024],
            'n_basis_layers': [1, 3, 6],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        n_knots = self.config.get('n_knots', 32)
        hidden_dim = self.config.get('hidden_dim', 256)
        n_basis_layers = self.config.get('n_basis_layers', 3)
        dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        self.n_knots = n_knots
        self.hidden_dim = hidden_dim
        
        # Encoder: extract features from input
        encoder_layers = []
        in_dim = self.n_features
        for i in range(n_basis_layers):
            out_dim = hidden_dim if i < n_basis_layers - 1 else hidden_dim
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < n_basis_layers - 1:
                encoder_layers.append(nn.GELU())
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Temporal encoder (per-feature GRU-like structure using linear layers)
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_basis_layers > 1 else 0
        )
        
        # Spline coefficient predictor
        self.knot_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_knots * self.n_features)
        )
        
        # Spline position predictor (normalized to [0, 1])
        self.position_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_knots),
            nn.Sigmoid()
        )
        
        # Amplitude modulation
        self.amplitude_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.n_features),
            nn.Sigmoid()
        )
        
        # Output time positions (learned or fixed)
        self.register_buffer('output_positions', torch.linspace(0, 1, output_len))
        
        # Fine-tuning MLP
        self.refiner = nn.Sequential(
            nn.Linear(self.n_features, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.n_features)
        )

    def _compute_b_spline_basis(self, t, knots, degree=3):
        # t: (output_len,) positions in [0, 1]
        # knots: (batch, n_knots) sorted positions in [0, 1]
        # Returns basis matrix: (batch, output_len, n_knots)
        
        batch_size = knots.size(0)
        n_knots = knots.size(1)
        n_eval = t.size(0)
        
        # Extend knots for boundary conditions
        knots_extended = torch.cat([
            knots[:, :1].repeat(1, degree),
            knots,
            knots[:, -1:].repeat(1, degree)
        ], dim=1)  # (batch, n_knots + 2*degree)
        
        # Evaluate B-spline basis using Cox-de Boor recursion (simplified)
        # For efficiency, use a simpler basis: Gaussian RBF around knots
        t_expanded = t.unsqueeze(0).unsqueeze(-1)  # (1, output_len, 1)
        knots_expanded = knots.unsqueeze(1)  # (batch, 1, n_knots)
        
        # Compute bandwidth based on knot spacing
        knot_diff = knots[:, 1:] - knots[:, :-1]  # (batch, n_knots-1)
        bandwidth = knot_diff.mean(dim=1, keepdim=True) * 2  # (batch, 1)
        bandwidth = bandwidth.clamp(min=0.01).unsqueeze(1)  # (batch, 1, 1)
        
        # RBF basis
        dist = (t_expanded - knots_expanded) / bandwidth  # (batch, output_len, n_knots)
        basis = torch.exp(-dist ** 2)  # (batch, output_len, n_knots)
        
        # Normalize
        basis = basis / (basis.sum(dim=-1, keepdim=True) + 1e-8)  # (batch, output_len, n_knots)
        
        return basis

    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        batch_size = x.size(0)
        
        # Encode each time step
        encoded = self.encoder(x)  # (batch, input_len, hidden_dim)
        
        # Temporal encoding
        temporal_out, (h_n, c_n) = self.temporal_encoder(encoded)  # temporal_out: (batch, input_len, hidden_dim)
        
        # Global context from final hidden state
        # h_n: (num_layers * num_directions, batch, hidden_dim//2)
        context = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (batch, hidden_dim)
        
        # Predict knot values
        knot_values = self.knot_predictor(context)  # (batch, n_knots * n_features)
        knot_values = knot_values.reshape(batch_size, self.n_knots, self.n_features)  # (batch, n_knots, n_features)
        
        # Predict knot positions (sort them)
        knot_positions = self.position_predictor(context)  # (batch, n_knots)
        knot_positions, _ = torch.sort(knot_positions, dim=-1)  # (batch, n_knots)
        
        # Predict amplitude modulation
        amplitude = self.amplitude_net(context)  # (batch, n_features)
        amplitude = amplitude.unsqueeze(1)  # (batch, 1, n_features)
        
        # Compute B-spline basis at output positions
        basis = self._compute_b_spline_basis(self.output_positions, knot_positions)  # (batch, output_len, n_knots)
        
        # Evaluate spline: (batch, output_len, n_knots) @ (batch, n_knots, n_features)
        spline_out = torch.bmm(basis, knot_values)  # (batch, output_len, n_features)
        
        # Apply amplitude modulation
        spline_out = spline_out * amplitude  # (batch, output_len, n_features)
        
        # Refine with MLP
        refined = self.refiner(spline_out)  # (batch, output_len, n_features)
        out = spline_out + 0.1 * refined  # (batch, output_len, n_features)
        
        # Add trend from last input value
        last_val = x[:, -1:, :]  # (batch, 1, n_features)
        out = out + last_val  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out
