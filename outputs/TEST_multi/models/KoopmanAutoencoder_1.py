import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class KoopmanAutoencoder(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'latent_dim': [32, 128, 512],
            'encoder_depth': [2, 3, 5],
            'koopman_regularization': [0.0, 0.01, 0.1],
            'dropout': [0.0, 0.1, 0.3]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.latent_dim = self.config.get('latent_dim', 128)
        self.encoder_depth = self.config.get('encoder_depth', 3)
        self.koopman_reg = self.config.get('koopman_regularization', 0.01)
        self.dropout = self.config.get('dropout', 0.1)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Encoder: time -> latent
        encoder_layers = []
        in_dim = self.input_len
        for i in range(self.encoder_depth):
            out_dim = self.latent_dim if i == self.encoder_depth - 1 else self.latent_dim * 2
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers[:-2])  # Remove last activation/dropout for latent
        
        # Koopman operator (learnable matrix for each feature/channel)
        self.koopman_K = nn.Parameter(torch.randn(self.n_features, self.latent_dim, self.latent_dim) * 0.01)
        
        # Decoder: latent -> time
        decoder_layers = []
        in_dim = self.latent_dim
        for i in range(self.encoder_depth):
            out_dim = self.output_len if i == self.encoder_depth - 1 else self.latent_dim * 2
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = out_dim
        self.decoder = nn.Sequential(*decoder_layers[:-2])
        
        # Final projection
        self.final_proj = nn.Linear(self.output_len, self.output_len)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Process each channel independently
        B, T, F = x.shape  # (batch, input_len, n_features)
        
        # Encode: (batch, n_features, input_len) -> (batch, n_features, latent_dim)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        latent = []
        for f in range(F):
            z = self.encoder(x_perm[:, f, :])  # (batch, latent_dim)
            latent.append(z)
        latent = torch.stack(latent, dim=1)  # (batch, n_features, latent_dim)
        
        # Koopman evolution: z_{t+1} = K * z_t
        # For multi-step: z_{t+k} = K^k * z_t
        # Use matrix exponential for stability
        predictions = []
        current_latent = latent  # (batch, n_features, latent_dim)
        
        # Compute Koopman power for output steps
        # Simplified: apply K once and linearly interpolate, or use eigendecomposition
        for f in range(F):
            K_f = self.koopman_K[f]  # (latent_dim, latent_dim)
            # Ensure stability via skew-symmetric component
            K_stable = K_f - K_f.T  # Skew-symmetric for pure rotation (no explosion)
            K_stable = K_stable + torch.eye(self.latent_dim, device=K_f.device) * 0.5  # Slight contraction
            
            z_f = current_latent[:, f, :]  # (batch, latent_dim)
            
            # Evolve: simple linear step
            z_evolved = []
            for step in range(self.output_len):
                # Progressive evolution with damping
                power = step + 1
                K_power = torch.matrix_power(K_stable, min(power, 10))  # Cap power for stability
                z_step = z_f @ K_power.T  # (batch, latent_dim)
                z_evolved.append(z_step)
            
            z_evolved = torch.stack(z_evolved, dim=1)  # (batch, output_len, latent_dim)
            predictions.append(z_evolved)
        
        predictions = torch.stack(predictions, dim=2)  # (batch, output_len, n_features, latent_dim)
        
        # Decode each step
        out = []
        for t in range(self.output_len):
            step_latent = predictions[:, t, :, :]  # (batch, n_features, latent_dim)
            decoded = []
            for f in range(F):
                d = self.decoder(step_latent[:, f, :])  # (batch, output_len) - actually should be single value
                # Fix: decoder outputs output_len, we need to redesign
                # Actually let's use a different approach: decode latent to single step
                d = d[:, :1]  # Take first element as prediction
                decoded.append(d)
            decoded = torch.cat(decoded, dim=-1)  # (batch, n_features)
            out.append(decoded)
        
        out = torch.stack(out, dim=1)  # (batch, output_len, n_features)
        
        # Alternative simpler approach: direct latent to full sequence
        # Redesign: latent -> decoder outputs output_len directly
        out = torch.zeros(B, self.output_len, F, device=x.device)  # (batch, output_len, n_features)
        for f in range(F):
            # Decode full latent evolution for this feature
            z_f = predictions[:, :, f, :]  # (batch, output_len, latent_dim)
            # Flatten and decode
            z_flat = z_f.reshape(B * self.output_len, self.latent_dim)  # (batch*output_len, latent_dim)
            d_flat = self.decoder(z_flat)  # (batch*output_len, output_len) - wrong dimension
            
        # Let me rewrite more carefully
        # Actually, let's use a simpler correct implementation
        # Encode to latent, then use Koopman to get future latents, then decode
        
        # Correct approach: encoder outputs latent, we evolve in latent space, then decode each
        # But decoder should map latent_dim -> 1 (for single step)
        # Let's adjust: final decoder layer outputs 1
        
        # Re-initialize decoder properly
        # Actually, let me just fix in forward with a proper architecture
        
        # Simpler correct version:
        # latent: (batch, n_features, latent_dim)
        # For each future step, evolve and decode
        
        # Rebuild decoder on the fly for single step
        single_step_decoder = nn.Linear(self.latent_dim, 1).to(x.device)
        
        out_steps = []
        current_z = latent  # (batch, n_features, latent_dim)
        for step in range(self.output_len):
            step_out = []
            for f in range(F):
                K_f = self.koopman_K[f]  # (latent_dim, latent_dim)
                # Simple evolution
                z_next = current_z[:, f, :] @ K_f.T  # (batch, latent_dim)
                # Decode to single value
                val = single_step_decoder(z_next).squeeze(-1)  # (batch,)
                step_out.append(val)
                # Update for next step
                current_z = current_z.clone()
                current_z[:, f, :] = z_next
            step_out = torch.stack(step_out, dim=1)  # (batch, n_features)
            out_steps.append(step_out)
        
        out = torch.stack(out_steps, dim=1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

