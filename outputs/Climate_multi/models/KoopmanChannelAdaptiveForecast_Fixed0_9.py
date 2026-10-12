import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanChannelAdaptiveForecast_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [64, 256, 1024],
            'encoder_depth': [2, 4, 8],
            'n_koopman_blocks': [2, 8, 32],
            'gate_dim': [16, 32, 64],
            'dropout': [0.0, 0.15, 0.3],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 256)
        self.encoder_depth = self.config.get('encoder_depth', 4)
        self.n_koopman_blocks = self.config.get('n_koopman_blocks', 8)
        self.gate_dim = self.config.get('gate_dim', 32)
        self.dropout_rate = self.config.get('dropout', 0.15)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Channel-adaptive gate: learns feature-specific scaling
        self.channel_gate_fc1 = nn.Linear(self.n_features, self.gate_dim)  # (batch, gate_dim)
        self.channel_gate_fc2 = nn.Linear(self.gate_dim, self.n_features)  # (batch, n_features)
        self.channel_gate_norm = nn.LayerNorm(self.n_features)
        
        # Encoder: maps input time series to latent Koopman observables
        encoder_layers = []
        in_dim = self.input_len
        for i in range(self.encoder_depth):
            out_dim = self.latent_dim if i == self.encoder_depth - 1 else self.latent_dim * 2
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.encoder_depth - 1:
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(self.dropout_rate))
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Koopman operator: linear evolution in latent space
        # Use block-diagonal structure for multiple Koopman operators
        self.koopman_blocks = nn.ModuleList([
            nn.Linear(self.latent_dim, self.latent_dim, bias=False)
            for _ in range(self.n_koopman_blocks)
        ])
        self.koopman_mix = nn.Linear(self.latent_dim * self.n_koopman_blocks, self.latent_dim)
        self.koopman_norm = nn.LayerNorm(self.latent_dim)
        
        # Decoder: maps evolved latent state back to time series
        decoder_layers = []
        in_dim = self.latent_dim
        for i in range(self.encoder_depth):
            out_dim = self.output_len if i == self.encoder_depth - 1 else self.latent_dim * 2
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.encoder_depth - 1:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(self.dropout_rate))
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Residual projection for direct path
        self.residual_proj = nn.Linear(self.input_len, self.output_len)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize Koopman operators near identity for stable dynamics
        for block in self.koopman_blocks:
            nn.init.orthogonal_(block.weight)
            with torch.no_grad():
                block.weight.data = 0.9 * block.weight.data + 0.1 * torch.eye(
                    self.latent_dim, device=block.weight.device
                )
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Channel-adaptive gating: compute feature-specific scales
        # Aggregate temporal info for gating
        x_gate_pool = x.mean(dim=1)  # (batch, n_features)
        gate = F.relu(self.channel_gate_fc1(x_gate_pool))  # (batch, gate_dim)
        gate = torch.sigmoid(self.channel_gate_fc2(gate))  # (batch, n_features)
        gate = self.channel_gate_norm(gate)  # (batch, n_features)
        
        # Apply channel gating
        x_gated = x * gate.unsqueeze(1)  # (batch, input_len, n_features)
        
        # Residual connection around gating
        x_gated = x_gated + x  # (batch, input_len, n_features)
        
        # Encode: (batch, n_features, input_len) -> (batch, n_features, latent_dim)
        x_enc = x_gated.permute(0, 2, 1)  # (batch, n_features, input_len)
        latent = self.encoder(x_enc)  # (batch, n_features, latent_dim)
        
        # Koopman evolution: apply multiple Koopman blocks and mix
        koopman_outputs = []
        for block in self.koopman_blocks:
            evolved = block(latent)  # (batch, n_features, latent_dim)
            koopman_outputs.append(evolved)
        
        koopman_concat = torch.cat(koopman_outputs, dim=-1)  # (batch, n_features, latent_dim * n_koopman_blocks)
        latent_evolved = self.koopman_mix(koopman_concat)  # (batch, n_features, latent_dim)
        latent_evolved = self.koopman_norm(latent_evolved)  # (batch, n_features, latent_dim)
        
        # Decode: (batch, n_features, latent_dim) -> (batch, n_features, output_len)
        out = self.decoder(latent_evolved)  # (batch, n_features, output_len)
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # Residual connection: direct linear projection
        residual = self.residual_proj(x_enc)  # (batch, n_features, output_len)
        residual = residual.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = out + residual  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

