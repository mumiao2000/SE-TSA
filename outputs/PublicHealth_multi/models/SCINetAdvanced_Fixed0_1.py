import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base import BaseForecastModel
from src.models.layers.revin import RevIN

class SCINetAdvanced_Fixed0(BaseForecastModel):
    @classmethod
    def get_all_configs(cls):
        return {
            'hidden_size': [64, 256, 1024],
            'levels': [2, 4, 6],
            'kernel_size': [3, 5, 9],
            'dropout': [0.0, 0.3],
            'groups': [1, 4, 16]
        }

    def __init__(self, input_len, output_len, n_features, config=None):
        super().__init__(input_len, output_len, n_features, config)
        
        self.hidden_size = self.config.get('hidden_size', 256)
        self.levels = self.config.get('levels', 4)
        self.kernel_size = self.config.get('kernel_size', 5)
        self.dropout = self.config.get('dropout', 0.1)
        self.groups = self.config.get('groups', 4)
        
        self.revin = RevIN(self.n_features, affine=True)
        
        # Ensure kernel is odd
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
        # Input projection
        self.input_projection = nn.Conv1d(self.n_features, self.hidden_size, kernel_size=1)
        
        # Build SCI-Tree
        self.tree = SCITree(self.levels, self.hidden_size, self.kernel_size, self.dropout, self.groups)
        
        # Bidirectional tree
        self.tree_reverse = SCITree(self.levels, self.hidden_size, self.kernel_size, self.dropout, self.groups)
        
        # Fusion of forward and backward
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.input_len * self.hidden_size, self.output_len * self.n_features)
        )
        
    def forward(self, x):
        # x: (batch, input_len, n_features)
        x = self.revin(x, mode='norm')  # (batch, input_len, n_features)
        
        # Input projection: (batch, n_features, input_len)
        x_perm = x.permute(0, 2, 1)  # (batch, n_features, input_len)
        x_proj = self.input_projection(x_perm)  # (batch, hidden_size, input_len)
        x_proj = x_proj.permute(0, 2, 1)  # (batch, input_len, hidden_size)
        
        # Forward tree
        x_forward = self.tree(x_proj)  # (batch, input_len, hidden_size)
        
        # Backward tree
        x_reverse = torch.flip(x_proj, dims=[1])  # (batch, input_len, hidden_size)
        x_backward = self.tree_reverse(x_reverse)  # (batch, input_len, hidden_size)
        x_backward = torch.flip(x_backward, dims=[1])  # (batch, input_len, hidden_size)
        
        # Fusion
        x_fused = torch.cat([x_forward, x_backward], dim=-1)  # (batch, input_len, hidden_size*2)
        x_fused = self.fusion(x_fused)  # (batch, input_len, hidden_size)
        
        # No residual - tree output is already processed; use fusion output directly
        x_out = x_fused  # (batch, input_len, hidden_size)
        
        # Projection
        x_out = self.projection(x_out)  # (batch, output_len * n_features)
        x_out = x_out.view(-1, self.output_len, self.n_features)  # (batch, output_len, n_features)
        
        x_out = self.revin(x_out, mode='denorm')  # (batch, output_len, n_features)
        return x_out


class SCITree(nn.Module):
    def __init__(self, levels, hidden_size, kernel_size, dropout, groups):
        super().__init__()
        self.levels = levels
        self.hidden_size = hidden_size
        
        # Build tree structure
        self.blocks = nn.ModuleList()
        for level in range(levels):
            level_blocks = nn.ModuleList()
            for node in range(2 ** level):
                level_blocks.append(
                    SCIBlock(hidden_size, kernel_size, dropout, groups)
                )
            self.blocks.append(level_blocks)
        
        # Skip connections for intermediate levels
        self.skip_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(levels)
        ])
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Store original length for reconstruction
        original_len = x.shape[1]
        current = [x]
        
        # Store lengths at each level for proper reconstruction
        level_lengths = [original_len]
        
        for level in range(self.levels):
            next_level = []
            next_lengths = []
            for node_idx, block in enumerate(self.blocks[level]):
                if node_idx < len(current):
                    even, odd = block(current[node_idx])
                    next_level.extend([even, odd])
                    next_lengths.extend([even.shape[1], odd.shape[1]])
            current = next_level
            level_lengths = next_lengths
        
        # Reconstruct: merge from bottom up
        for level in range(self.levels - 1, -1, -1):
            merged = []
            i = 0
            while i < len(current):
                if i + 1 < len(current):
                    # Interleave even and odd
                    even, odd = current[i], current[i + 1]
                    min_len = min(even.shape[1], odd.shape[1])
                    even = even[:, :min_len, :]
                    odd = odd[:, :min_len, :]
                    
                    # Interleave
                    merged_seq = torch.zeros(even.shape[0], min_len * 2, even.shape[2], device=even.device)
                    merged_seq[:, 0::2, :] = even
                    merged_seq[:, 1::2, :] = odd
                    merged.append(merged_seq)
                    i += 2
                else:
                    merged.append(current[i])
                    i += 1
            current = merged
        
        out = current[0]
        
        # Ensure output matches original input length
        if out.shape[1] != original_len:
            if out.shape[1] < original_len:
                # Pad to original length
                pad_len = original_len - out.shape[1]
                out = F.pad(out, (0, 0, 0, pad_len))  # (batch, original_len, hidden_size)
            else:
                # Truncate to original length
                out = out[:, :original_len, :]  # (batch, original_len, hidden_size)
        
        return out


class SCIBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size, dropout, groups):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Ensure groups divides hidden_size
        self.groups = min(groups, hidden_size)
        while hidden_size % self.groups != 0:
            self.groups -= 1
        
        # Separate convolutions for even and odd branches
        self.even_conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size//2, groups=self.groups),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size//2, groups=self.groups)
        )
        
        self.odd_conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size//2, groups=self.groups),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size//2, groups=self.groups)
        )
        
        # Interaction terms
        self.even_to_odd = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size//2, groups=self.groups)
        self.odd_to_even = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size//2, groups=self.groups)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Split into even and odd indices
        even_idx = torch.arange(0, x.shape[1], 2, device=x.device)
        odd_idx = torch.arange(1, x.shape[1], 2, device=x.device)
        
        even = x[:, even_idx, :]  # (batch, ceil(seq_len/2), hidden_size)
        odd = x[:, odd_idx, :]    # (batch, floor(seq_len/2), hidden_size)
        
        # Pad if necessary to make equal length
        if even.shape[1] != odd.shape[1]:
            if even.shape[1] > odd.shape[1]:
                odd = F.pad(odd, (0, 0, 0, 1))  # (batch, ceil(seq_len/2), hidden_size)
            else:
                even = F.pad(even, (0, 0, 0, 1))  # (batch, floor(seq_len/2)+1, hidden_size)
        
        # Permute for conv1d
        even_perm = even.permute(0, 2, 1)  # (batch, hidden_size, seq_len//2)
        odd_perm = odd.permute(0, 2, 1)    # (batch, hidden_size, seq_len//2)
        
        # Main convolutions
        even_conv = self.even_conv(even_perm)  # (batch, hidden_size, seq_len//2)
        odd_conv = self.odd_conv(odd_perm)     # (batch, hidden_size, seq_len//2)
        
        # Cross interactions
        even_to_odd = self.even_to_odd(even_perm)  # (batch, hidden_size, seq_len//2)
        odd_to_even = self.odd_to_even(odd_perm)   # (batch, hidden_size, seq_len//2)
        
        # Gating
        combined = torch.cat([even_conv, odd_conv], dim=1)  # (batch, hidden_size*2, seq_len//2)
        gate = self.gate(combined.permute(0, 2, 1)).permute(0, 2, 1)  # (batch, hidden_size, seq_len//2)
        
        # Update with gating
        even_out = even_perm + gate * odd_to_even  # (batch, hidden_size, seq_len//2)
        odd_out = odd_perm + (1 - gate) * even_to_odd  # (batch, hidden_size, seq_len//2)
        
        # Permute back
        even_out = even_out.permute(0, 2, 1)  # (batch, seq_len//2, hidden_size)
        odd_out = odd_out.permute(0, 2, 1)    # (batch, seq_len//2, hidden_size)
        
        return even_out, odd_out
