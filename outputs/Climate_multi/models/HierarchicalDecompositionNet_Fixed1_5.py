import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HierarchicalDecompositionNet_Fixed1(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'n_scales': [2, 4, 8],
            'basis_dim': [16, 64, 256],
            'n_bases_per_scale': [2, 4, 8],
            'dropout': [0.0, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.n_scales = self.config.get('n_scales', 4)
        self.basis_dim = self.config.get('basis_dim', 64)
        self.n_bases_per_scale = self.config.get('n_bases_per_scale', 4)
        self.dropout_rate = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Multi-scale decomposition filters (learned wavelet-like)
        self.decomposition_filters = nn.ModuleList()
        for i in range(self.n_scales):
            kernel_size = 2 ** (i + 1) + 1
            padding = kernel_size // 2
            self.decomposition_filters.append(
                nn.Conv1d(self.n_features, self.n_features, kernel_size, padding=padding, groups=self.n_features)
            )
        
        # Residual (high-frequency) filter
        self.residual_filter = nn.Conv1d(self.n_features, self.n_features, 3, padding=1, groups=self.n_features)
        
        # Basis function generators for each scale
        self.basis_generators = nn.ModuleList()
        for i in range(self.n_scales):
            self.basis_generators.append(
                nn.Sequential(
                    nn.Linear(self.input_len, self.basis_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.basis_dim, self.output_len * self.n_bases_per_scale)
                )
            )
        
        # Coefficient predictors for each scale
        self.coeff_predictors = nn.ModuleList()
        for i in range(self.n_scales):
            self.coeff_predictors.append(
                nn.Sequential(
                    nn.Linear(self.input_len, self.basis_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.basis_dim, self.n_bases_per_scale * self.n_features)
                )
            )
        
        # Residual predictor - takes flattened input
        self.residual_predictor = nn.Sequential(
            nn.Linear(self.n_features * self.input_len, self.basis_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.basis_dim, self.output_len * self.n_features)
        )
        
        # Scale mixture gate (input-dependent weights for combining scales)
        self.mixture_gate = nn.Sequential(
            nn.Linear(self.n_features * self.input_len, self.n_scales + 1),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        batch = x.size(0)
        
        # Prepare for conv: (batch, n_features, input_len)
        x_conv = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Multi-scale decomposition
        scales = []
        current = x_conv
        for filt in self.decomposition_filters:
            scale = filt(current)  # (batch, n_features, input_len)
            scales.append(scale)
            current = current - scale  # residual for next scale
        
        # Final residual
        residual = self.residual_filter(current)  # (batch, n_features, input_len)
        
        # Flat input for coefficient prediction
        x_flat = x_conv.reshape(batch, -1)  # (batch, n_features * input_len)
        
        # Compute mixture weights
        mix_weights = self.mixture_gate(x_flat)  # (batch, n_scales + 1)
        
        # Predict for each scale using basis expansion
        predictions = []
        for i, (scale, basis_gen, coeff_pred) in enumerate(zip(scales, self.basis_generators, self.coeff_predictors)):
            # scale: (batch, n_features, input_len)
            scale_flat = scale.reshape(batch, self.n_features, self.input_len)  # (batch, n_features, input_len)
            
            # Pool across features for scale representation
            scale_repr = scale_flat.mean(dim=1)  # (batch, input_len)
            
            # Generate basis functions: (batch, output_len * n_bases)
            bases = basis_gen(scale_repr)  # (batch, output_len * n_bases)
            bases = bases.reshape(batch, self.n_bases_per_scale, self.output_len)  # (batch, n_bases, output_len)
            
            # Predict coefficients: (batch, n_bases * n_features)
            coeffs = coeff_pred(scale_repr)  # (batch, n_bases * n_features)
            coeffs = coeffs.reshape(batch, self.n_features, self.n_bases_per_scale)  # (batch, n_features, n_bases)
            
            # Combine: (batch, n_features, n_bases) @ (batch, n_bases, output_len)
            pred = torch.bmm(coeffs, bases)  # (batch, n_features, output_len)
            predictions.append(pred)
        
        # Residual prediction
        res_pred = self.residual_predictor(x_flat)  # (batch, output_len * n_features)
        res_pred = res_pred.reshape(batch, self.n_features, self.output_len)  # (batch, n_features, output_len)
        predictions.append(res_pred)
        
        # Weighted combination
        mix_weights = mix_weights.reshape(batch, 1, 1, self.n_scales + 1)  # (batch, 1, 1, n_scales+1)
        preds_stack = torch.stack(predictions, dim=-1)  # (batch, n_features, output_len, n_scales+1)
        out = (preds_stack * mix_weights).sum(dim=-1)  # (batch, n_features, output_len)
        
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Add direct trend continuation
        trend = x[:, -1:, :]  # (batch, 1, n_features)
        # Simple linear extrapolation
        steps = torch.arange(1, self.output_len + 1, device=x.device).float().view(1, self.output_len, 1)
        trend_ext = trend * (1 + steps * 0.01)  # (batch, output_len, n_features)
        out = out + trend_ext * 0.1  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

