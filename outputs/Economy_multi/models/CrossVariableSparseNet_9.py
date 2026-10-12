import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class CrossVariableSparseNet(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'temporal_dim': [32, 128, 512],
            'n_neighbors': [3, 7, 15],
            'n_message_rounds': [1, 2, 4],
            'dropout': [0.0, 0.2],
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        self.temporal_dim = self.config.get('temporal_dim', 128)
        self.n_neighbors = self.config.get('n_neighbors', 7)
        self.n_message_rounds = self.config.get('n_message_rounds', 2)
        self.dropout = self.config.get('dropout', 0.2)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Feature-specific temporal encoders (depthwise separable)
        self.temporal_enc = nn.Conv1d(
            n_features, n_features * self.temporal_dim,
            kernel_size=3, padding=1, groups=n_features
        )
        self.temporal_gate = nn.Conv1d(
            n_features, n_features * self.temporal_dim,
            kernel_size=3, padding=1, groups=n_features
        )
        
        # Feature embeddings for sparse neighbor selection
        self.feat_emb = nn.Parameter(torch.randn(n_features, self.temporal_dim))
        nn.init.xavier_uniform_(self.feat_emb)
        
        # Message passing MLPs
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.temporal_dim * 2, self.temporal_dim),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.temporal_dim, self.temporal_dim)
            ) for _ in range(self.n_message_rounds)
        ])
        
        # Update MLPs
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.temporal_dim * 2, self.temporal_dim),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.temporal_dim, self.temporal_dim)
            ) for _ in range(self.n_message_rounds)
        ])
        
        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(self.temporal_dim, self.temporal_dim // 2),
            nn.ELU(),
            nn.Linear(self.temporal_dim // 2, output_len)
        )
        
    def get_sparse_neighbors(self):
        # Compute feature similarity
        sim = torch.mm(self.feat_emb, self.feat_emb.t())  # (n_features, n_features)
        # Exclude self
        mask = torch.eye(self.n_features, device=sim.device).bool()
        sim = sim.masked_fill(mask, float('-inf'))
        
        # Select top-k neighbors per feature
        _, topk_idx = torch.topk(sim, min(self.n_neighbors, self.n_features - 1), dim=-1)
        return topk_idx  # (n_features, n_neighbors)
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Transpose for conv: (batch, n_features, input_len)
        x = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        
        # Feature-specific temporal encoding with gating
        h = self.temporal_enc(x)  # (batch, n_features * temporal_dim, input_len)
        g = torch.sigmoid(self.temporal_gate(x))  # (batch, n_features * temporal_dim, input_len)
        h = h * g  # gated
        
        # Reshape: (batch, n_features, temporal_dim, input_len)
        h = h.view(h.size(0), self.n_features, self.temporal_dim, h.size(-1))
        # Pool over time
        h = h.mean(dim=-1)  # (batch, n_features, temporal_dim)
        
        # Get sparse neighbor structure
        neighbors = self.get_sparse_neighbors()  # (n_features, n_neighbors)
        
        # Message passing rounds
        for round_idx in range(self.n_message_rounds):
            # Gather neighbor features
            # neighbors[i] gives indices of neighbors for feature i
            batch_size = h.size(0)
            
            # Create messages
            messages = []
            for i in range(self.n_features):
                # Self feature
                h_i = h[:, i:i+1, :]  # (batch, 1, temporal_dim)
                # Neighbor features
                nbr_idx = neighbors[i]  # (n_neighbors,)
                h_nbr = h[:, nbr_idx, :]  # (batch, n_neighbors, temporal_dim)
                
                # Broadcast self to neighbors
                h_i_exp = h_i.expand(-1, h_nbr.size(1), -1)  # (batch, n_neighbors, temporal_dim)
                
                # Concatenate and compute message
                msg_in = torch.cat([h_i_exp, h_nbr], dim=-1)  # (batch, n_neighbors, temporal_dim * 2)
                msg = self.message_mlps[round_idx](msg_in)  # (batch, n_neighbors, temporal_dim)
                
                # Aggregate messages (mean)
                msg_agg = msg.mean(dim=1)  # (batch, temporal_dim)
                messages.append(msg_agg)
            
            # Stack messages
            messages = torch.stack(messages, dim=1)  # (batch, n_features, temporal_dim)
            
            # Update
            update_in = torch.cat([h, messages], dim=-1)  # (batch, n_features, temporal_dim * 2)
            h_new = self.update_mlps[round_idx](update_in)  # (batch, n_features, temporal_dim)
            h = h + h_new  # residual
        
        # Output projection per feature
        out = self.output_proj(h)  # (batch, n_features, output_len)
        
        # Transpose
        out = out.permute(0, 2, 1)  # (batch, output_len, n_features)
        
        out = self.revin(out, mode='denorm')  # (batch, output_len, n_features)
        return out

