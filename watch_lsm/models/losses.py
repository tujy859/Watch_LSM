import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositePretrainLoss(nn.Module):
    """
    Composite Self-Supervised Pretraining Loss for Smartwatch Foundation Models:
    1. Masked Autoencoding Reconstruction Loss (L1/L2)
    2. Cross-Modal Contrastive Alignment Loss (InfoNCE between PPG and ACC representations)
    3. Spectral Frequency Domain Loss (ensuring cardiac & movement harmonics are preserved)
    """
    def __init__(self, w_recon=1.0, w_cross=0.2, w_spectral=0.1, temperature=0.07):
        super().__init__()
        self.w_recon = w_recon
        self.w_cross = w_cross
        self.w_spectral = w_spectral
        self.temperature = temperature
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def info_nce_loss(self, emb_ppg, emb_acc):
        z_p = F.normalize(emb_ppg, dim=-1)
        z_a = F.normalize(emb_acc, dim=-1)
        
        sim_matrix = torch.matmul(z_p, z_a.T) / self.temperature
        labels = torch.arange(z_p.size(0), device=z_p.device)
        loss_p2a = F.cross_entropy(sim_matrix, labels)
        loss_a2p = F.cross_entropy(sim_matrix.T, labels)
        return (loss_p2a + loss_a2p) / 2.0

    def spectral_loss(self, pred_patches, target_patches):
        B, N, P = pred_patches.shape
        pred_sig = pred_patches.reshape(B, N * P)
        target_sig = target_patches.reshape(B, N * P)
        
        fft_pred = torch.abs(torch.fft.rfft(pred_sig, dim=-1))
        fft_target = torch.abs(torch.fft.rfft(target_sig, dim=-1))
        return self.l1(fft_pred, fft_target)

    def forward(self, outputs, mask_ppg=None, mask_acc=None):
        pred_p = outputs["pred_ppg"]
        target_p = outputs["target_ppg"]
        pred_a = outputs["pred_acc"]
        target_a = outputs["target_acc"]
        
        # 1. Masked Reconstruction Loss
        if mask_ppg is not None:
            m_p = mask_ppg.bool().unsqueeze(-1).expand_as(pred_p)
            recon_loss_p = self.mse(pred_p[m_p], target_p[m_p]) if m_p.any() else self.mse(pred_p, target_p)
        else:
            recon_loss_p = self.mse(pred_p, target_p)
            
        if mask_acc is not None:
            m_a = mask_acc.bool().unsqueeze(-1).expand_as(pred_a)
            recon_loss_a = self.mse(pred_a[m_a], target_a[m_a]) if m_a.any() else self.mse(pred_a, target_a)
        else:
            recon_loss_a = self.mse(pred_a, target_a)
            
        loss_recon = recon_loss_p + recon_loss_a
        
        # 2. Cross-Modal Contrastive Loss
        emb_p = outputs["latent_ppg"].mean(dim=1)
        emb_a = outputs["latent_acc"].mean(dim=1)
        loss_cross = self.info_nce_loss(emb_p, emb_a)
        
        # 3. Spectral Loss
        loss_spec = self.spectral_loss(pred_p, target_p)
        
        total_loss = (
            self.w_recon * loss_recon +
            self.w_cross * loss_cross +
            self.w_spectral * loss_spec
        )
        
        return {
            "total_loss": total_loss,
            "recon_loss": loss_recon,
            "cross_modal_loss": loss_cross,
            "spectral_loss": loss_spec
        }
