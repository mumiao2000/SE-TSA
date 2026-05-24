import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperTemporalMLP_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hyper_dim': [32, 128, 512],
            'pred_hidden': [64, 256, 1024],
            'n_pred_layers': [2, 3, 5],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hyper_dim = self.config.get('hyper_dim', 128)
        self.pred_hidden = self.config.get('pred_hidden', 256)
        self.n_pred_layers = self.config.get('n_pred_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Advanced preprocessing: feature-wise adaptive normalization
        self.feature_gates = nn.Parameter(torch.ones(self.n_features))
        self.feature_norms = nn.ModuleList([
            nn.LayerNorm(self.input_len) for _ in range(self.n_features)
        ])
        
        # Hypernetwork: encodes temporal context
        self.hyper_encoder = nn.Sequential(
            nn.Linear(self.input_len * self.n_features, self.hyper_dim * 4),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hyper_dim * 4, self.hyper_dim * 2),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hyper_dim * 2, self.hyper_dim)
        )
        
        # Hypernetwork generates weights for prediction MLP
        # Prediction MLP: n_features * input_len -> n_features * output_len
        pred_in = self.n_features * self.input_len
        pred_out = self.n_features * self.output_len
        
        # Calculate total parameters in prediction network
        self.pred_shapes = []
        in_dim = pred_in
        total_params = 0
        for i in range(self.n_pred_layers):
            out_dim = pred_out if i == self.n_pred_layers - 1 else self.pred_hidden
            # Weight matrix + bias
            total_params += in_dim * out_dim + out_dim
            self.pred_shapes.append((in_dim, out_dim))
            in_dim = out_dim
        
        # Hypernetwork outputs all prediction network parameters
        self.hyper_weights = nn.Linear(self.hyper_dim, total_params)
        
        # Optional: layer-specific scaling
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(self.n_pred_layers)
        ])
        
        # Output scaling for denormalization approximation
        self.output_scale = nn.Parameter(torch.ones(self.n_features))
        self.output_bias = nn.Parameter(torch.zeros(self.n_features))
        
    def apply_advanced_norm(self, x):
        # x: (batch, input_len, n_features)
        B, T, F = x.shape  # (batch, input_len, n_features)
        
        # Per-feature normalization
        x_norm = []
        for f in range(F):
            x_f = x[:, :, f]  # (batch, input_len)
            x_f = self.feature_norms[f](x_f)  # (batch, input_len)
            x_f = x_f * self.feature_gates[f]  # (batch, input_len)
            x_norm.append(x_f)
        x_norm = torch.stack(x_norm, dim=2)  # (batch, input_len, n_features)
        return x_norm
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.apply_advanced_norm(x)  # (batch, input_len, n_features)
        
        B, T, F = x.shape  # (batch, input_len, n_features)
        
        # Encode context with hypernetwork
        x_flat = x.reshape(B, -1)  # (batch, input_len * n_features)
        hyper_z = self.hyper_encoder(x_flat)  # (batch, hyper_dim)
        
        # Generate prediction network weights
        params = self.hyper_weights(hyper_z)  # (batch, total_params)
        
        # Apply generated weights
        current = x_flat  # (batch, input_len * n_features)
        
        param_idx = 0
        for i, (in_d, out_d) in enumerate(self.pred_shapes):
            # Extract weight and bias
            n_w = in_d * out_d
            n_b = out_d
            
            W = params[:, param_idx:param_idx + n_w].reshape(B, in_d, out_d)  # (batch, in_d, out_d)
            param_idx += n_w
            b = params[:, param_idx:param_idx + n_b].reshape(B, 1, out_d)  # (batch, 1, out_d)
            param_idx += n_b
            
            # Apply with layer-specific scaling
            current = torch.bmm(current.unsqueeze(1), W * self.layer_scales[i]).squeeze(1) + b.squeeze(1)  # (batch, out_d)
            
            if i < self.n_pred_layers - 1:
                # Use explicit SiLU definition to avoid F.silu import issues
                current = current * torch.sigmoid(current)  # (batch, out_d)
                current = torch.nn.functional.dropout(current, p=self.dropout, training=self.training)  # (batch, out_d)
        
        # Reshape to output
        out = current.reshape(B, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Apply learned output scaling per feature
        out = out * self.output_scale.unsqueeze(0).unsqueeze(0) + self.output_bias.unsqueeze(0).unsqueeze(0)  # (batch, output_len, n_features)
        
        return out
