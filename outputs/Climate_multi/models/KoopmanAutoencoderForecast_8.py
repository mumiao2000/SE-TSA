import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanAutoencoderForecast(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [64, 256, 1024],
            'encoder_depth': [1, 3, 5],
            'n_koopman_blocks': [2, 8, 32],
            'dropout': [0.0, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 256)
        self.encoder_depth = self.config.get('encoder_depth', 3)
        self.n_koopman_blocks = self.config.get('n_koopman_blocks', 8)
        self.dropout_rate = self.config.get('dropout', 0.0)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: maps (batch, input_len, n_features) -> (batch, latent_dim)
        encoder_layers = []
        in_dim = self.input_len * self.n_features
        for i in range(self.encoder_depth):
            out_dim = self.latent_dim if i == self.encoder_depth - 1 else max(self.latent_dim, in_dim // 2)
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.encoder_depth - 1:
                encoder_layers.append(nn.LayerNorm(out_dim))
                encoder_layers.append(nn.GELU())
                encoder_layers.append(nn.Dropout(self.dropout_rate))
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Koopman operator: block-diagonal for stability
        self.koopman_blocks = nn.ParameterList([
            nn.Parameter(torch.randn(self.latent_dim // self.n_koopman_blocks, 
                                    self.latent_dim // self.n_koopman_blocks) * 0.01)
            for _ in range(self.n_koopman_blocks)
        ])
        
        # Ensure Koopman operator is near-identity for stability
        self.koopman_identity = nn.Parameter(torch.eye(self.latent_dim // self.n_koopman_blocks), requires_grad=False)
        
        # Decoder: maps (batch, latent_dim) -> (batch, output_len, n_features)
        decoder_layers = []
        in_dim = self.latent_dim
        for i in range(self.encoder_depth):
            out_dim = self.output_len * self.n_features if i == self.encoder_depth - 1 else max(self.latent_dim, in_dim // 2)
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < self.encoder_depth - 1:
                decoder_layers.append(nn.LayerNorm(out_dim))
                decoder_layers.append(nn.GELU())
                decoder_layers.append(nn.Dropout(self.dropout_rate))
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Temporal evolution: predict multiple steps
        self.step_weights = nn.Parameter(torch.ones(self.output_len) / self.output_len)
        
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def apply_koopman(self, z, n_steps):
        # z: (batch, latent_dim)
        # Apply Koopman operator n_steps times
        block_size = self.latent_dim // self.n_koopman_blocks
        z_blocks = torch.split(z, block_size, dim=-1)  # list of (batch, block_size)
        
        results = []
        for zb, K in zip(z_blocks, self.koopman_blocks):
            # Koopman is K + I for near-identity initialization
            K_eff = K + self.koopman_identity
            # Matrix power via repeated multiplication
            z_evolved = zb
            for _ in range(n_steps):
                z_evolved = torch.matmul(z_evolved, K_eff.T)  # (batch, block_size)
            results.append(z_evolved)
        
        return torch.cat(results, dim=-1)  # (batch, latent_dim)
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Encode
        x_flat = x.view(-1, self.input_len * self.n_features)  # (batch, input_len * n_features)
        z = self.encoder(x_flat)  # (batch, latent_dim)
        z = self.dropout(z)
        
        # Predict each output step with Koopman evolution
        # Weighted combination of different step predictions for smoothness
        outputs = []
        for t in range(1, self.output_len + 1):
            z_t = self.apply_koopman(z, t)  # (batch, latent_dim)
            out_t = self.decoder(z_t)  # (batch, output_len * n_features)
            # We only need the t-th step, but decoder outputs full sequence
            # Re-decode for each step
            out_t = out_t.view(-1, self.output_len, self.n_features)[:, :1, :]  # (batch, 1, n_features)
            outputs.append(out_t)
        
        # Actually, more efficient: decode once per step
        # Let's recompute properly
        out = torch.zeros(x.size(0), self.output_len, self.n_features, device=x.device, dtype=x.dtype)
        for t in range(self.output_len):
            z_t = self.apply_koopman(z, t + 1)  # (batch, latent_dim)
            step_out = self.decoder(z_t)  # (batch, output_len * n_features)
            # Take appropriate slice - actually we need per-step decoder
            # Simpler: use shared decoder, take first prediction as representative
            step_out = step_out.view(-1, self.output_len, self.n_features)
            out[:, t:t+1, :] = step_out[:, 0:1, :]  # (batch, 1, n_features)
        
        # Better approach: linear interpolation in latent space then decode once
        z_steps = []
        for t in range(1, self.output_len + 1):
            z_steps.append(self.apply_koopman(z, t))
        z_traj = torch.stack(z_steps, dim=1)  # (batch, output_len, latent_dim)
        
        # Decode all at once
        z_traj_flat = z_traj.view(-1, self.latent_dim)  # (batch * output_len, latent_dim)
        out_flat = self.decoder(z_traj_flat)  # (batch * output_len, output_len * n_features)
        # Take diagonal: each z_t predicts step t
        out = out_flat.view(-1, self.output_len, self.output_len, self.n_features)
        # Extract predictions: out[b, t, t, :] for each t
        out = torch.stack([out[:, i, i, :] for i in range(self.output_len)], dim=1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

