import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperNetworkAdaptiveForecaster_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hyper_dim': [64, 256, 1024],
            'prediction_hidden': [32, 128, 512],
            'n_hyper_layers': [2, 3, 5],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hyper_dim = self.config.get('hyper_dim', 256)
        self.prediction_hidden = self.config.get('prediction_hidden', 128)
        self.n_hyper_layers = self.config.get('n_hyper_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Context encoder: extracts statistics from input
        self.context_encoder = nn.Sequential(
            nn.Linear(self.n_features, self.hyper_dim),
            nn.LayerNorm(self.hyper_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            # Temporal pooling via attention
            nn.Linear(self.hyper_dim, 1)
        )  # outputs attention weights
        
        # Project pooled context to hyper_dim
        self.context_proj = nn.Linear(self.n_features, self.hyper_dim)
        
        # FFT-based frequency context
        self.freq_encoder = nn.Sequential(
            nn.Linear(self.n_features * 2, self.hyper_dim),  # real and imag parts
            nn.LayerNorm(self.hyper_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Hypernetwork: generates prediction network weights
        # Prediction network: (n_features + time_idx) -> n_features
        pred_in = self.n_features + 1  # feature + normalized time position
        pred_out = self.n_features
        
        # Weight sizes for prediction network: [pred_in * prediction_hidden, prediction_hidden, prediction_hidden * pred_out, pred_out]
        self.weight_sizes = [
            pred_in * self.prediction_hidden,
            self.prediction_hidden,
            self.prediction_hidden * pred_out,
            pred_out
        ]
        self.total_weights = sum(self.weight_sizes)
        
        # Hypernetwork MLP
        hyper_layers = []
        in_dim = self.hyper_dim * 2  # combined context + freq
        for i in range(self.n_hyper_layers):
            out_dim = self.hyper_dim * 2 if i < self.n_hyper_layers - 1 else self.total_weights
            hyper_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < self.n_hyper_layers - 1 else nn.Identity(),
                nn.GELU() if i < self.n_hyper_layers - 1 else nn.Identity(),
                nn.Dropout(self.dropout) if i < self.n_hyper_layers - 1 else nn.Identity()
            ])
            in_dim = out_dim
        self.hypernetwork = nn.Sequential(*hyper_layers)
        
        # Time embedding for prediction
        self.time_embed = nn.Embedding(self.output_len, 1)
    
    def extract_context(self, x):
        # x: (batch, input_len, n_features)
        # Attention-based pooling
        attn = self.context_encoder[0](x)  # (batch, input_len, hyper_dim)
        for layer in self.context_encoder[1:]:
            attn = layer(attn)  # (batch, input_len, 1) at end
        attn = F.softmax(attn.squeeze(-1), dim=1)  # (batch, input_len)
        
        context = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # (batch, n_features)
        context = self.context_proj(context)  # (batch, hyper_dim)
        
        # Frequency context
        x_fft = torch.fft.rfft(x, dim=1)  # (batch, input_len//2+1, n_features), complex
        # Use dominant frequencies
        magnitudes = torch.abs(x_fft)  # (batch, input_len//2+1, n_features)
        _, top_indices = torch.topk(magnitudes.mean(dim=-1), k=min(5, magnitudes.shape[1]), dim=1)  # (batch, 5)
        
        # Gather top frequency components
        freq_features = torch.cat([
            x_fft.real[:, :min(10, x_fft.shape[1]), :].reshape(x.shape[0], -1),
            x_fft.imag[:, :min(10, x_fft.shape[1]), :].reshape(x.shape[0], -1)
        ], dim=1)  # (batch, freq_features)
        
        # Pad or truncate to expected size
        freq_dim = self.n_features * 2
        if freq_features.shape[1] > freq_dim:
            freq_features = freq_features[:, :freq_dim]
        elif freq_features.shape[1] < freq_dim:
            freq_features = F.pad(freq_features, (0, freq_dim - freq_features.shape[1]))
        
        freq_context = self.freq_encoder(freq_features)  # (batch, hyper_dim)
        
        return context, freq_context  # both (batch, hyper_dim)
    
    def generate_weights(self, context, freq_context):
        # Combine contexts
        combined = torch.cat([context, freq_context], dim=-1)  # (batch, hyper_dim * 2)
        weights = self.hypernetwork(combined)  # (batch, total_weights)
        return weights
    
    def apply_prediction_network(self, x, weights, time_idx):
        # x: (batch, n_features), weights: (batch, total_weights), time_idx: (batch,)
        batch_size = x.shape[0]
        
        # Parse weights
        w1 = weights[:, :self.weight_sizes[0]].reshape(batch_size, self.n_features + 1, self.prediction_hidden)
        b1 = weights[:, self.weight_sizes[0]:self.weight_sizes[0]+self.weight_sizes[1]].reshape(batch_size, 1, self.prediction_hidden)
        w2 = weights[:, self.weight_sizes[0]+self.weight_sizes[1]:-self.weight_sizes[3]].reshape(batch_size, self.prediction_hidden, self.n_features)
        b2 = weights[:, -self.weight_sizes[3]:].reshape(batch_size, 1, self.n_features)
        
        # Add time embedding
        t_embed = self.time_embed(time_idx)  # (batch, 1)
        x_t = torch.cat([x, t_embed], dim=-1).unsqueeze(1)  # (batch, 1, n_features + 1)
        
        # Prediction network with generated weights
        h = torch.bmm(x_t, w1) + b1  # (batch, 1, prediction_hidden)
        h = F.gelu(h)  # (batch, 1, prediction_hidden)
        h = F.dropout(h, p=self.dropout, training=self.training)  # (batch, 1, prediction_hidden)
        out = torch.bmm(h, w2) + b2  # (batch, 1, n_features)
        
        return out.squeeze(1)  # (batch, n_features)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Extract context
        context, freq_context = self.extract_context(x)  # (batch, hyper_dim), (batch, hyper_dim)
        
        # Generate prediction network weights
        weights = self.generate_weights(context, freq_context)  # (batch, total_weights)
        
        # Use last timestep as initial condition
        x_last = x[:, -1, :]  # (batch, n_features)
        
        # Auto-regressive prediction with generated weights
        predictions = []
        x_t = x_last  # (batch, n_features)
        for t in range(self.output_len):
            time_idx = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)  # (batch,)
            x_t = self.apply_prediction_network(x_t, weights, time_idx)  # (batch, n_features)
            predictions.append(x_t)  # (batch, n_features)
        
        out = torch.stack(predictions, dim=1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out  # (batch, output_len, n_features)
