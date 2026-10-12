import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossformerHybrid_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'd_model': [64, 256, 1024],
            'n_heads': [2, 8, 32],
            'e_layers': [1, 3, 6],
            'd_layers': [1, 2, 4],
            'dropout': [0.0, 0.3],
            'segment_len': [4, 8, 16]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.d_model = self.config.get('d_model', 256)
        self.n_heads = self.config.get('n_heads', 8)
        self.e_layers = self.config.get('e_layers', 3)
        self.d_layers = self.config.get('d_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.segment_len = self.config.get('segment_len', 8)
        
        # Adjust segment length
        self.segment_len = min(self.segment_len, input_len // 4)
        self.segment_len = max(self.segment_len, 2)
        
        self.win_size = self.segment_len  # window size for segmentation
        
        # RevIN normalization
        self.revin = RevIN(self.n_features, affine=True)
        
        # Embedding: project each segment to d_model
        self.value_embedding = nn.Linear(self.win_size * self.n_features, self.d_model)
        
        # Encoder layers with hierarchical reduction
        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.encoder_projs = nn.ModuleList()
        
        d_in = self.d_model
        self.encoder_dims = [d_in]  # Track dimensions at each layer
        for i in range(self.e_layers):
            self.encoder_layers.append(
                TwoStageAttentionLayer(d_in, self.n_heads, self.win_size, self.dropout)
            )
            self.encoder_norms.append(nn.LayerNorm(d_in))
            # Downsample for next layer (except last)
            if i < self.e_layers - 1:
                down_ratio = 2
                d_out = max(d_in // down_ratio, 64)
                self.encoder_projs.append(nn.Linear(d_in, d_out))
                d_in = d_out
                self.encoder_dims.append(d_in)
            else:
                self.encoder_projs.append(None)
        
        self.final_d_model = d_in  # Final encoder output dimension
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        
        for i in range(self.d_layers):
            self.decoder_layers.append(
                DecoderLayer(self.final_d_model, self.n_heads, self.dropout)
            )
            self.decoder_norms.append(nn.LayerNorm(self.final_d_model))
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(self.final_d_model, self.final_d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.final_d_model // 2, self.win_size * self.n_features)
        )
        
        # For output: we need to generate output_len predictions
        self.output_len = output_len
        self.n_features = n_features
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        B, T, N = x.shape
        
        # RevIN
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Segment the input: (batch, num_seg, win_size, n_features)
        num_seg = T // self.win_size
        if num_seg == 0:
            # Pad if needed
            pad_len = self.win_size - T
            x = F.pad(x, (0, 0, 0, pad_len))  # (batch, win_size, n_features)
            num_seg = 1
        
        x = x[:, :num_seg * self.win_size, :]  # (batch, num_seg*win_size, n_features)
        x_segments = x.reshape(B, num_seg, self.win_size, N)  # (batch, num_seg, win_size, n_features)
        
        # Flatten each segment for embedding
        x_flat = x_segments.reshape(B, num_seg, -1)  # (batch, num_seg, win_size*n_features)
        
        # Embed
        enc_out = self.value_embedding(x_flat)  # (batch, num_seg, d_model)
        
        # Encoder
        for i in range(self.e_layers):
            enc_out = self.encoder_layers[i](enc_out)  # (batch, num_seg, encoder_dims[i])
            enc_out = self.encoder_norms[i](enc_out)  # (batch, num_seg, encoder_dims[i])
            
            # Downsample segments for next layer
            if self.encoder_projs[i] is not None:
                # Merge adjacent segments
                curr_seg = enc_out.shape[1]
                if curr_seg > 1:
                    # Handle odd number of segments by padding if needed
                    if curr_seg % 2 == 1:
                        pad_seg = torch.zeros(B, 1, enc_out.shape[2], device=enc_out.device)  # (batch, 1, encoder_dims[i])
                        enc_out = torch.cat([enc_out, pad_seg], dim=1)  # (batch, curr_seg+1, encoder_dims[i])
                        curr_seg += 1
                    enc_out = enc_out.reshape(B, curr_seg // 2, 2, -1).mean(dim=2)  # (batch, curr_seg//2, encoder_dims[i])
                enc_out = self.encoder_projs[i](enc_out)  # (batch, new_num_seg, encoder_dims[i+1])
        
        # Decoder: prepare query for forecasting
        # Simple approach: use last encoder state to initialize decoder
        # For direct forecasting, we'll use a learned query
        
        # Predict future segments
        num_pred_seg = (self.output_len + self.win_size - 1) // self.win_size
        
        # Use encoder output to generate predictions
        # Take mean of encoder outputs as context
        context = enc_out.mean(dim=1, keepdim=True)  # (batch, 1, final_d_model)
        
        # Generate predictions autoregressively or directly
        # Direct approach: replicate context and project
        pred = context.repeat(1, num_pred_seg, 1)  # (batch, num_pred_seg, final_d_model)
        
        # Apply decoder layers
        for i in range(self.d_layers):
            pred = self.decoder_layers[i](pred, enc_out)  # (batch, num_pred_seg, final_d_model)
            pred = self.decoder_norms[i](pred)  # (batch, num_pred_seg, final_d_model)
        
        # Project to segments
        pred = self.projection(pred)  # (batch, num_pred_seg, win_size*n_features)
        
        # Reshape to (batch, num_pred_seg*win_size, n_features)
        pred = pred.reshape(B, num_pred_seg, self.win_size, N)  # (batch, num_pred_seg, win_size, n_features)
        pred = pred.reshape(B, -1, N)  # (batch, num_pred_seg*win_size, n_features)
        
        # Trim to output_len
        pred = pred[:, :self.output_len, :]  # (batch, output_len, n_features)
        
        # RevIN denorm
        pred = self.revin(pred, mode='denorm')  # (batch, output_len, n_features)
        
        return pred


class TwoStageAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, win_size, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.win_size = win_size
        
        # Cross-time attention: attend within each segment's time steps
        self.time_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.time_norm = nn.LayerNorm(d_model)
        
        # Cross-dimension attention: attend across feature segments
        self.dim_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dim_norm = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ff_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (batch, num_seg, d_model)
        B, S, D = x.shape
        
        # Time attention (self-attention across segments)
        time_out, _ = self.time_attention(x, x, x)  # (batch, num_seg, d_model)
        x = self.time_norm(x + time_out)  # (batch, num_seg, d_model)
        
        # Dimension attention using router tokens
        # Create router tokens
        routers = x.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        
        # Router attends to all segments
        dim_out, _ = self.dim_attention(routers, x, x)  # (batch, 1, d_model)
        dim_out = dim_out.repeat(1, S, 1)  # (batch, num_seg, d_model)
        x = self.dim_norm(x + dim_out)  # (batch, num_seg, d_model)
        
        # Feed-forward
        ff_out = self.ff(x)  # (batch, num_seg, d_model)
        x = self.ff_norm(x + ff_out)  # (batch, num_seg, d_model)
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ff_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_out):
        # Self attention
        self_out, _ = self.self_attn(x, x, x)  # (batch, tgt_len, d_model)
        x = self.self_norm(x + self_out)  # (batch, tgt_len, d_model)
        
        # Cross attention to encoder
        cross_out, _ = self.cross_attn(x, enc_out, enc_out)  # (batch, tgt_len, d_model)
        x = self.cross_norm(x + cross_out)  # (batch, tgt_len, d_model)
        
        # FF
        ff_out = self.ff(x)  # (batch, tgt_len, d_model)
        x = self.ff_norm(x + ff_out)  # (batch, tgt_len, d_model)
        
        return x
