import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperNetworkForecaster(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hyper_dim': [64, 256, 1024],
            'n_predict_layers': [1, 3, 6],
            'predict_hidden': [32, 128, 512],
            'dropout': [0.0, 0.3],
            'n_z_samples': [1, 4, 16]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        cfg = self.config or {}
        self.hyper_dim = cfg.get('hyper_dim', 256)
        self.n_predict_layers = cfg.get('n_predict_layers', 3)
        self.predict_hidden = cfg.get('predict_hidden', 128)
        self.dropout = cfg.get('dropout', 0.1)
        self.n_z_samples = cfg.get('n_z_samples', 4)
        
        self.revin = RevIN(n_features, affine=True)
        
        # Encoder: process input sequence to get hypernetwork input
        self.encoder_lstm = nn.LSTM(
            n_features, self.hyper_dim, 
            num_layers=2, 
            batch_first=True,
            dropout=self.dropout,
            bidirectional=True
        )
        
        # Compress bidirectional output
        self.encoder_proj = nn.Linear(self.hyper_dim * 2, self.hyper_dim)
        
        # Hypernetwork: generates weights for prediction network
        # Calculate total parameters needed
        self._calculate_weight_shapes()
        
        # Hypernetwork MLP
        self.hyper_net = nn.Sequential(
            nn.Linear(self.hyper_dim, self.hyper_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hyper_dim, self.hyper_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_dim, self.total_weights)
        )
        
        # Optional: variational hypernetwork for uncertainty
        self.hyper_mu = nn.Linear(self.hyper_dim, self.hyper_dim)
        self.hyper_logvar = nn.Linear(self.hyper_dim, self.hyper_dim)
        
        # Direct trend/cycle extraction
        self.trend_extractor = nn.Linear(input_len, 1)
        self.seasonal_extractor = nn.Conv1d(n_features, n_features, kernel_size=7, padding=3, groups=n_features)
        
    def _calculate_weight_shapes(self):
        # Prediction network: maps (input_len, n_features) -> (output_len, n_features)
        # We use a temporal MLP: flatten time, predict future time steps
        
        self.weight_shapes = []
        self.bias_shapes = []
        
        # Layer 1: input_len * n_features -> predict_hidden
        in_size = self.input_len * self.n_features
        self.weight_shapes.append((self.predict_hidden, in_size))
        self.bias_shapes.append((self.predict_hidden,))
        
        # Hidden layers
        for _ in range(self.n_predict_layers - 1):
            self.weight_shapes.append((self.predict_hidden, self.predict_hidden))
            self.bias_shapes.append((self.predict_hidden,))
        
        # Output layer: predict_hidden -> output_len * n_features
        self.weight_shapes.append((self.output_len * self.n_features, self.predict_hidden))
        self.bias_shapes.append((self.output_len * self.n_features,))
        
        # Calculate total parameters
        self.total_weights = sum(w[0] * w[1] + b[0] for w, b in zip(self.weight_shapes, self.bias_shapes))
    
    def _generate_weights(self, z):
        # z: (batch, hyper_dim)
        weights = self.hyper_net(z)  # (batch, total_weights)
        
        # Parse weights
        weight_list = []
        bias_list = []
        idx = 0
        for w_shape, b_shape in zip(self.weight_shapes, self.bias_shapes):
            w_size = w_shape[0] * w_shape[1]
            b_size = b_shape[0]
            
            w = weights[:, idx:idx+w_size].view(-1, *w_shape)  # (batch, out, in)
            idx += w_size
            b = weights[:, idx:idx+b_size]  # (batch, out)
            idx += b_size
            
            weight_list.append(w)
            bias_list.append(b)
        
        return weight_list, bias_list
    
    def _apply_predict_net(self, x, weights, biases):
        # x: (batch, input_len * n_features)
        for i, (w, b) in enumerate(zip(weights, biases)):
            # w: (batch, out, in), b: (batch, out)
            # x: (batch, in)
            x = torch.bmm(w, x.unsqueeze(-1)).squeeze(-1) + b  # (batch, out)
            if i < len(weights) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Extract trend and seasonal components
        x_trend = x.mean(dim=-1, keepdim=True)  # (batch, input_len, 1)
        x_trend = self.trend_extractor(x_trend.squeeze(-1)).unsqueeze(-1)  # (batch, 1, 1)
        
        x_seasonal = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_seasonal = self.seasonal_extractor(x_seasonal)  # (batch, n_features, input_len)
        x_seasonal = x_seasonal.permute(0, 2, 1)  # (batch, input_len, n_features)
        
        x_decomp = x - x_seasonal + x_trend  # (batch, input_len, n_features)
        
        # Encode with LSTM
        enc_out, (h_n, c_n) = self.encoder_lstm(x_decomp)  # (batch, input_len, hyper_dim*2)
        
        # Final hidden state
        h_forward = h_n[-2, :, :]  # (batch, hyper_dim)
        h_backward = h_n[-1, :, :]  # (batch, hyper_dim)
        h_cat = torch.cat([h_forward, h_backward], dim=-1)  # (batch, hyper_dim*2)
        z = self.encoder_proj(h_cat)  # (batch, hyper_dim)
        z = F.relu(z)
        
        # Generate prediction network weights
        weights, biases = self._generate_weights(z)  # lists of (batch, ...)
        
        # Prepare input for prediction network
        x_flat = x_decomp.view(batch, -1)  # (batch, input_len * n_features)
        
        # Apply generated network
        out = self._apply_predict_net(x_flat, weights, biases)  # (batch, output_len * n_features)
        out = out.view(batch, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        # Denormalize
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out
