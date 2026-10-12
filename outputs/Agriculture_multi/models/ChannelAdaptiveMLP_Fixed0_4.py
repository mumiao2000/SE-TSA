import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class ChannelAdaptiveMLP_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_dim': [128, 512, 2048],
            'n_layers': [1, 3, 6],
            'dropout': [0.0, 0.3],
            'gate_dim': [32, 128, 512]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.n_layers = self.config.get('n_layers', 3)
        self.dropout_rate = self.config.get('dropout', 0.1)
        self.gate_dim = self.config.get('gate_dim', 128)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Per-channel temporal encoders
        self.channel_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_len, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.gate_dim),
            ) for _ in range(self.n_features)
        ])  # (batch, gate_dim) per channel
        
        # Cross-channel attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.gate_dim,
            num_heads=4,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Feature interaction MLP
        layers = []
        in_dim = self.gate_dim * self.n_features
        for i in range(self.n_layers):
            out_dim = self.hidden_dim if i < self.n_layers - 1 else self.gate_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            ])
            in_dim = out_dim
        self.interaction_mlp = nn.Sequential(*layers)
        
        # Per-channel decoders
        self.channel_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.gate_dim * 2, self.hidden_dim),  # *2 for skip connection
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, output_len),
            ) for _ in range(self.n_features)
        ])
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Per-channel encoding
        channel_feats = []
        for i, encoder in enumerate(self.channel_encoders):
            ch = x[:, :, i]  # (batch, input_len)
            ch = encoder(ch)  # (batch, gate_dim)
            channel_feats.append(ch)
        
        # Stack to (batch, n_features, gate_dim)
        channel_stack = torch.stack(channel_feats, dim=1)  # (batch, n_features, gate_dim)
        
        # Cross-channel attention
        attn_out, _ = self.cross_attn(channel_stack, channel_stack, channel_stack)  # (batch, n_features, gate_dim)
        
        # Combine with original features
        combined = attn_out + channel_stack  # (batch, n_features, gate_dim)
        
        # Flatten and interact - use reshape instead of view to handle non-contiguous tensors
        flat = combined.reshape(combined.size(0), -1)  # (batch, n_features * gate_dim)
        interacted = self.interaction_mlp(flat)  # (batch, gate_dim)
        
        # Expand back for per-channel decoding
        interacted_expanded = interacted.unsqueeze(1).expand(-1, self.n_features, -1)  # (batch, n_features, gate_dim)
        
        # Concatenate with channel-specific features
        decoder_input = torch.cat([combined, interacted_expanded], dim=-1)  # (batch, n_features, gate_dim * 2)
        
        # Per-channel decoding
        outputs = []
        for i, decoder in enumerate(self.channel_decoders):
            ch_in = decoder_input[:, i, :]  # (batch, gate_dim * 2)
            ch_out = decoder(ch_in)  # (batch, output_len)
            outputs.append(ch_out)
        
        out = torch.stack(outputs, dim=-1)  # (batch, output_len, n_features)
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

