"""Contrastive Representation Distillation (CRD-style).

Pulls student feature closer to teacher feature for same sample,
pushes away from other samples in the batch.

Reference: Tian et al., "Contrastive Representation Distillation", ICLR 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveKDLoss(nn.Module):
    """Simplified CRD: InfoNCE loss between teacher and student projections.

    Args:
        student_dim: feature dimension of student.
        teacher_dim: feature dimension of teacher.
        proj_dim: shared projection dimension.
        temperature: InfoNCE temperature.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        proj_dim:    int   = 128,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.T = temperature
        # Projection heads (GAP → MLP)
        self.s_proj = nn.Sequential(
            nn.Linear(student_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.t_proj = nn.Sequential(
            nn.Linear(teacher_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(
        self,
        student_feat: torch.Tensor,   # (B, Cs, H, W, D)
        teacher_feat: torch.Tensor,   # (B, Ct, H, W, D) — no grad
    ) -> torch.Tensor:
        # Global average pool → (B, C)
        s = student_feat.mean(dim=[2, 3, 4])
        t = teacher_feat.detach().mean(dim=[2, 3, 4])

        # Project and L2-normalise
        s = F.normalize(self.s_proj(s), dim=1)   # (B, proj_dim)
        t = F.normalize(self.t_proj(t), dim=1)

        # InfoNCE: positives are same-index pairs (diagonal)
        logits = torch.matmul(s, t.T) / self.T   # (B, B)
        labels = torch.arange(s.size(0), device=s.device)
        return F.cross_entropy(logits, labels)
