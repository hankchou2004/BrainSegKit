"""Response-based KD (Hinton et al., 2015).

Loss = CE(student, hard_label) + alpha * KL(student_soft || teacher_soft)

Temperature T softens the distributions:
    soft_prob = softmax(logits / T)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResponseKDLoss(nn.Module):
    """Soft-label Knowledge Distillation loss.

    Args:
        temperature: Distillation temperature T (higher → softer).
        alpha: Weight of KD term; (1-alpha) weights the CE hard-label term.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits:  torch.Tensor,   # (B, C, ...)
        teacher_logits:  torch.Tensor,   # (B, C, ...)  — no grad
        hard_labels:     torch.Tensor,   # (B, 1, ...)  — integer class indices
    ) -> torch.Tensor:
        # Hard-label CE
        ce = F.cross_entropy(student_logits, hard_labels.squeeze(1).long())

        # Soft KL divergence — flatten spatial dims
        B, C = student_logits.shape[:2]
        s_flat = student_logits.view(B, C, -1).permute(0, 2, 1)   # (B, N, C)
        t_flat = teacher_logits.view(B, C, -1).permute(0, 2, 1)

        kl = F.kl_div(
            F.log_softmax(s_flat / self.T, dim=-1),
            F.softmax(t_flat  / self.T, dim=-1),
            reduction="batchmean",
        ) * (self.T ** 2)

        return (1 - self.alpha) * ce + self.alpha * kl
