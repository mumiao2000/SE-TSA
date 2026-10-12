import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class HyperNetworkKoopmanHybrid(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hyper_dim': [64, 256, 1024],
            'n_koopman_modes': [4, 16, 64],
            'n_resolutions': [2, 4, 8],
            'dropout': [0.0, 0.15, 0.3],
            'latent_rank': [8, 32, 128]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        # Hyperparameters
        self.hyper_dim = self.config.get('hyper_dim', 256)
        self.n_koopman_modes = self.config.get('n_koopman_modes', 16)
        self.n_resolutions = self.config.get('n_resolutions', 4)
        self.dropout_rate = self.config.get('dropout', 0.15)
        self.latent_rank = self.config.get('latent_rank', 32)
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Temporal embedding
        self.temporal_embed = nn.Linear(self.input_len, self.hyper_dim)
        
        # Multi-resolution period extraction
        self.period_list = self._compute_periods()
        self.resolution_embeds = nn.ModuleList([
            nn.Linear(p, self.hyper_dim // self.n_resolutions) 
            for p in self.period_list
        ])
        
        # HyperNetwork: generates Koopman transition weights
        self.hyper_encoder = nn.Sequential(
            nn.Linear(self.hyper_dim, self.hyper_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hyper_dim, self.hyper_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Generate low-rank Koopman operators: K = U @ diag(eigen) @ V^T
        self.koopman_U_generator = nn.Linear(self.hyper_dim // 2, self.n_features * self.latent_rank)
        self.koopman_V_generator = nn.Linear(self.hyper_dim // 2, self.n_features * self.latent_rank)
        self.koopman_eigen_generator = nn.Linear(self.hyper_dim // 2, self.latent_rank)
        
        # Channel-specific bias adaptation
        self.bias_generator = nn.Linear(self.hyper_dim // 2, self.n_features)
        
        # Feature projection for dynamics
        self.feature_proj_in = nn.Linear(self.n_features, self.n_features)
        self.feature_proj_out = nn.Linear(self.n_features, self.n_features)
        
        # Temporal evolution MLP for residual
        self.temporal_mlp = nn.Sequential(
            nn.Linear(self.input_len, self.hyper_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hyper_dim, self.output_len)
        )
        
        # Final mixing
        self.output_mixer = nn.Sequential(
            nn.Linear(self.n_features * 2, self.n_features),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.n_features, self.n_features)
        )
        
    def _compute_periods(self):
        # Use dominant period ~12 and divisors for multi-resolution
        base_period = max(2, int(round(12)))
        periods = []
        for i in range(1, self.n_resolutions + 1):
            p = max(2, base_period // i)
            if p not in periods:
                periods.append(p)
        # Pad if needed
        while len(periods) < self.n_resolutions:
            periods.append(periods[-1] // 2 if periods[-1] > 2 else 2)
        return periods[:self.n_resolutions]
    
    def forward(self, x):
        # x: (batch, input_len, n_features)
        batch_size = x.shape[0]
        
        # RevIN normalization
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Store for residual
        x_input = x  # (batch, input_len, n_features)
        
        # === Multi-resolution temporal encoding ===
        # Global temporal embedding
        global_temp = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        global_temp = global_temp.reshape(batch_size * self.n_features, self.input_len)
        global_embed = self.temporal_embed(global_temp)  # (batch*n_features, hyper_dim)
        global_embed = global_embed.reshape(batch_size, self.n_features, self.hyper_dim)
        global_embed = global_embed.mean(dim=1)  # (batch, hyper_dim)
        
        # Period-based resolution embeddings
        res_embeds = []
        for i, period in enumerate(self.period_list):
            # Reshape into periods
            if self.input_len >= period:
                n_periods = self.input_len // period
                x_period = x[:, :n_periods * period, :]  # (batch, n_periods*period, n_features)
                x_period = x_period.reshape(batch_size, n_periods, period, self.n_features)
                x_period = x_period.permute(0, 3, 1, 2)  # (batch, n_features, n_periods, period)
                x_period = x_period.reshape(batch_size * self.n_features * n_periods, period)
                
                period_embed = self.resolution_embeds[i](x_period)  # (batch*n_features*n_periods, hyper_dim//n_res)
                period_embed = period_embed.reshape(batch_size, self.n_features, n_periods, -1)
                period_embed = period_embed.mean(dim=2)  # (batch, n_features, hyper_dim//n_res)
                period_embed = period_embed.mean(dim=1)  # (batch, hyper_dim//n_res)
                res_embeds.append(period_embed)
            else:
                # Fallback: use global embedding slice
                res_embeds.append(global_embed[:, :self.hyper_dim // self.n_resolutions])
        
        # Combine resolution embeddings
        if len(res_embeds) > 0:
            res_combined = torch.cat(res_embeds, dim=-1)  # (batch, sum of dims)
            # Pad or project to hyper_dim
            if res_combined.shape[-1] < self.hyper_dim:
                padding = torch.zeros(batch_size, self.hyper_dim - res_combined.shape[-1], device=x.device)
                res_combined = torch.cat([res_combined, padding], dim=-1)
            elif res_combined.shape[-1] > self.hyper_dim:
                res_combined = res_combined[:, :self.hyper_dim]
        else:
            res_combined = global_embed
        
        # Fuse global and multi-resolution
        hyper_input = global_embed + res_combined  # (batch, hyper_dim)
        
        # === HyperNetwork generates Koopman operators ===
        hyper_state = self.hyper_encoder(hyper_input)  # (batch, hyper_dim//2)
        
        # Generate low-rank Koopman factors
        U_flat = self.koopman_U_generator(hyper_state)  # (batch, n_features * latent_rank)
        V_flat = self.koopman_V_generator(hyper_state)  # (batch, n_features * latent_rank)
        eigen_vals = self.koopman_eigen_generator(hyper_state)  # (batch, latent_rank)
        
        # Reshape to matrices
        U = U_flat.reshape(batch_size, self.n_features, self.latent_rank)  # (batch, n_features, latent_rank)
        V = V_flat.reshape(batch_size, self.n_features, self.latent_rank)  # (batch, n_features, latent_rank)
        
        # Ensure stable eigenvalues (magnitude < 1 for stability)
        eigen_vals = torch.tanh(eigen_vals)  # (batch, latent_rank)
        
        # Generate channel-specific bias
        channel_bias = self.bias_generator(hyper_state)  # (batch, n_features)
        
        # === Apply Koopman dynamics for prediction ===
        # Project input features
        x_proj = self.feature_proj_in(x)  # (batch, input_len, n_features)
        
        # Take last timestep as initial state
        x_last = x_proj[:, -1, :]  # (batch, n_features)
        
        # Koopman evolution: x_{t+1} = x_t + U @ diag(eigen^t) @ V^T @ x_t + bias
        outputs = []
        current_state = x_last  # (batch, n_features)
        
        for t in range(self.output_len):
            # Compute Koopman operator at time t
            eigen_t = eigen_vals ** (t + 1)  # (batch, latent_rank)
            eigen_diag = torch.diag_embed(eigen_t)  # (batch, latent_rank, latent_rank)
            
            # K = U @ diag(eigen) @ V^T
            K_temp = torch.bmm(U, eigen_diag)  # (batch, n_features, latent_rank)
            K = torch.bmm(K_temp, V.transpose(1, 2))  # (batch, n_features, n_features)
            
            # Apply dynamics
            state_evolved = torch.bmm(K, current_state.unsqueeze(-1)).squeeze(-1)  # (batch, n_features)
            state_evolved = state_evolved + channel_bias  # (batch, n_features)
            
            # Residual connection
            next_state = current_state + state_evolved  # (batch, n_features)
            outputs.append(next_state)
            current_state = next_state
        
        koopman_out = torch.stack(outputs, dim=1)  # (batch, output_len, n_features)
        koopman_out = self.feature_proj_out(koopman_out)  # (batch, output_len, n_features)
        
        # === Temporal MLP path for local patterns ===
        # Per-feature temporal evolution
        x_mlp = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        mlp_out = self.temporal_mlp(x_mlp)  # (batch, n_features, output_len)
        mlp_out = mlp_out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        # === Combine Koopman and MLP predictions ===
        combined = torch.cat([koopman_out, mlp_out], dim=-1)  # (batch, output_len, n_features*2)
        out = self.output_mixer(combined)  # (batch, output_len, n_features)
        
        # Final residual from last input
        last_input = x[:, -1:, :]  # (batch, 1, n_features)
        out = out + last_input  # (batch, output_len, n_features)
        
        # RevIN denormalization
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        
        return out

