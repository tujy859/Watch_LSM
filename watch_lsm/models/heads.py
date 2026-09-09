import torch
import torch.nn as nn

class HeartRateHead(nn.Module):
    """
    Heart Rate Estimation Head. Predicts instantaneous heart rate (BPM)
    from the fused physiological latent representation.
    """
    def __init__(self, in_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, in_dim) global embedding
        return self.net(x).squeeze(-1) # (B,)

class StepCountHead(nn.Module):
    """
    Step Cadence Head. Predicts step frequency (Steps Per Minute - SPM)
    from the motion-aware latent representation.
    """
    def __init__(self, in_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1) # (B,)

class ActivityClassificationHead(nn.Module):
    """
    Physical Activity State Head (0: Sedentary/Rest, 1: Walking, 2: Running, 3: Cycling/Workout).
    """
    def __init__(self, in_dim=128, num_classes=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x) # (B, num_classes)

class WatchLSMDownstreamModel(nn.Module):
    """
    Multi-Task Downstream Model for Smartwatch Physiological Signal Tasks.
    """
    def __init__(self, foundation_model, task="heart_rate", num_classes=4, freeze_backbone=False):
        super().__init__()
        self.tokenizer = foundation_model.tokenizer
        self.backbone = foundation_model.backbone
        hidden_dim = foundation_model.tokenizer.hidden_dim
        
        if freeze_backbone:
            for p in self.tokenizer.parameters():
                p.requires_grad = False
            for p in self.backbone.parameters():
                p.requires_grad = False
                
        self.task = task
        if task == "heart_rate":
            self.head = HeartRateHead(in_dim=hidden_dim)
        elif task == "cadence":
            self.head = StepCountHead(in_dim=hidden_dim)
        elif task == "activity":
            self.head = ActivityClassificationHead(in_dim=hidden_dim, num_classes=num_classes)
        elif task == "multitask":
            self.hr_head = HeartRateHead(in_dim=hidden_dim)
            self.cad_head = StepCountHead(in_dim=hidden_dim)
            self.act_head = ActivityClassificationHead(in_dim=hidden_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x_ppg, x_acc):
        tok_ppg, tok_acc, _, _ = self.tokenizer(x_ppg, x_acc)
        _, _, _, global_repr = self.backbone(tok_ppg, tok_acc)
        
        if self.task == "multitask":
            return {
                "hr": self.hr_head(global_repr),
                "cadence": self.cad_head(global_repr),
                "activity": self.act_head(global_repr)
            }
        else:
            return self.head(global_repr)
