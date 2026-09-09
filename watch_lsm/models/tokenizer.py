import torch
import torch.nn as nn

class MultiRatePatchTokenizer(nn.Module):
    """
    1D Multi-Rate Patch Tokenizer for Smartwatch Time Series.
    Aligns heterogeneous sampling rates (e.g. PPG 64 Hz and ACC 32 Hz)
    to a shared sequence of temporal tokens based on physical time intervals.
    """
    def __init__(
        self,
        hidden_dim=128,
        ppg_patch_size=16, # 16 samples = 0.25s at 64 Hz
        acc_patch_size=8,   # 8 samples = 0.25s at 32 Hz
        num_patches=32      # 32 tokens * 0.25s = 8.0 seconds
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.ppg_patch_size = ppg_patch_size
        self.acc_patch_size = acc_patch_size
        
        # Linear projection for PPG patches (1 channel * patch_size -> hidden_dim)
        self.ppg_proj = nn.Linear(1 * ppg_patch_size, hidden_dim)
        # Linear projection for ACC patches (3 channels * patch_size -> hidden_dim)
        self.acc_proj = nn.Linear(3 * acc_patch_size, hidden_dim)
        
        # Learnable temporal position embeddings
        self.pos_embed_ppg = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.pos_embed_acc = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        
        # Learnable mask tokens for MAE
        self.mask_token_ppg = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.mask_token_acc = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def patchify_ppg(self, x_ppg):
        # x_ppg: (B, 1, L_ppg) e.g. (B, 1, 512)
        B, C, L = x_ppg.shape
        x = x_ppg.reshape(B, C, self.num_patches, self.ppg_patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_patches, C * self.ppg_patch_size)
        return x # (B, N, patch_size)

    def patchify_acc(self, x_acc):
        # x_acc: (B, 3, L_acc) e.g. (B, 3, 256)
        B, C, L = x_acc.shape
        x = x_acc.reshape(B, C, self.num_patches, self.acc_patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_patches, C * self.acc_patch_size)
        return x # (B, N, 3 * patch_size)

    def forward(self, x_ppg, x_acc, mask_ppg=None, mask_acc=None):
        """
        Forward tokenization with optional masking.
        mask: (B, num_patches) binary tensor, 1=masked, 0=kept
        """
        B = x_ppg.size(0)
        
        # Patchify and project
        p_raw = self.patchify_ppg(x_ppg)
        a_raw = self.patchify_acc(x_acc)
        
        tok_ppg = self.ppg_proj(p_raw) + self.pos_embed_ppg # (B, N, D)
        tok_acc = self.acc_proj(a_raw) + self.pos_embed_acc # (B, N, D)
        
        # Apply mask tokens if provided
        if mask_ppg is not None:
            mask_m = mask_ppg.unsqueeze(-1) # (B, N, 1)
            tok_ppg = tok_ppg * (1 - mask_m) + self.mask_token_ppg * mask_m
            
        if mask_acc is not None:
            mask_m = mask_acc.unsqueeze(-1)
            tok_acc = tok_acc * (1 - mask_m) + self.mask_token_acc * mask_m
            
        return tok_ppg, tok_acc, p_raw, a_raw
