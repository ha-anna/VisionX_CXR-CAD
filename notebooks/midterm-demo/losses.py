"""Focal Loss для multi-label классификации (написан вручную, без внешних библиотек).

Для одной болезни с логитом z, p = sigmoid(z), меткой y in {0,1}:
    p_t  = p       если y = 1,   1 - p   если y = 0
    BCE  = -[ w*y*log(p) + (1-y)*log(1-p) ]          (w = pos_weight)
    FL   = (1 - p_t)^gamma * BCE

gamma = 0  ->  обычный BCEWithLogits (с pos_weight).
Чем больше gamma, тем сильнее "гасится" вклад лёгких примеров (p_t близко к 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        modulator = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        return (modulator * bce).mean()
