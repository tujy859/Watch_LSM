import torch
import torch.nn as nn
from .tokenizer import MultiRatePatchTokenizer

class CrossModalAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=128, n_heads=4, dropout=0.1):
        super().__init__()
        # Self-attention within modality
        self.self_attn_ppg = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.self_attn_acc = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention: PPG queries ACC (motion artifact removal)
        self.cross_attn_ppg2acc = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        
        # Norms and MLPs
        self.norm_p1 = nn.LayerNorm(hidden_dim)
        self.norm_a1 = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        
        self.mlp_ppg = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.mlp_acc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm_p2 = nn.LayerNorm(hidden_dim)
        self.norm_a2 = nn.LayerNorm(hidden_dim)

    def forward(self, h_ppg, h_acc):
        # 1. Self Attention
        p_self, _ = self.self_attn_ppg(h_ppg, h_ppg, h_ppg)
        h_ppg = self.norm_p1(h_ppg + p_self)
        
        a_self, _ = self.self_attn_acc(h_acc, h_acc, h_acc)
        h_acc = self.norm_a1(h_acc + a_self)
        
        # 2. Cross Attention (PPG attends to ACC motion reference)
        p_cross, _ = self.cross_attn_ppg2acc(query=h_ppg, key=h_acc, value=h_acc)
        h_ppg = self.norm_cross(h_ppg + p_cross)
        
        # 3. Feedforward
        h_ppg = self.norm_p2(h_ppg + self.mlp_ppg(h_ppg))
        h_acc = self.norm_a2(h_acc + self.mlp_acc(h_acc))
        
        return h_ppg, h_acc

class WatchLSMBackbone(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalAttentionBlock(hidden_dim=hidden_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_final_ppg = nn.LayerNorm(hidden_dim)
        self.norm_final_acc = nn.LayerNorm(hidden_dim)
        
        # Fusion projection
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, tok_ppg, tok_acc):
        h_p, h_a = tok_ppg, tok_acc
        for layer in self.layers:
            h_p, h_a = layer(h_p, h_a)
            
        h_p = self.norm_final_ppg(h_p)
        h_a = self.norm_final_acc(h_a)
        
        # Fused sequence tokens (B, N, D)
        fused = self.fusion(torch.cat([h_p, h_a], dim=-1))
        
        # Global pooled representation (B, D)
        global_repr = fused.mean(dim=1)
        
        return h_p, h_a, fused, global_repr

class WatchLSMModel(nn.Module):
    """
    Complete Smartwatch Foundation Model with Tokenizer, Cross-Modal Backbone,
    and Pretraining Decoders for Masked Autoencoding.
    """
    def __init__(
        self,
        hidden_dim=128,
        n_layers=4,
        n_heads=4,
        ppg_patch_size=16,
        acc_patch_size=8,
        num_patches=32,
        dropout=0.1
    ):
        super().__init__()
        self.tokenizer = MultiRatePatchTokenizer(
            hidden_dim=hidden_dim,
            ppg_patch_size=ppg_patch_size,
            acc_patch_size=acc_patch_size,
            num_patches=num_patches
        )
        self.backbone = WatchLSMBackbone(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # MAE Reconstruction Decoders
        self.ppg_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ppg_patch_size)
        )
        self.acc_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * acc_patch_size)
        )

    def forward(self, x_ppg, x_acc, mask_ppg=None, mask_acc=None):
        tok_ppg, tok_acc, p_raw, a_raw = self.tokenizer(x_ppg, x_acc, mask_ppg, mask_acc)
        h_p, h_a, fused, global_repr = self.backbone(tok_ppg, tok_acc)
        
        # Reconstruct patches
        pred_p = self.ppg_decoder(h_p)
        pred_a = self.acc_decoder(h_a)
        
        return {
            "pred_ppg": pred_p,
            "pred_acc": pred_a,
            "target_ppg": p_raw,
            "target_acc": a_raw,
            "latent_ppg": h_p,
            "latent_acc": h_a,
            "fused_tokens": fused,
            "global_embedding": global_repr
        }
