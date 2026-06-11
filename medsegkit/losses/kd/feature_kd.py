"""Feature-based KD — aligns intermediate encoder feature maps.

Methods supported:
  "l2"   : MSE between teacher/student features (FitNets style)
  "at"   : Attention Transfer (Zagoruyko & Komodakis, 2017)
  "l1"   : L1 distance
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureKDLoss(nn.Module):
    """Penalises difference between teacher and student feature maps.

    A 1×1 adaptor conv is learned if channel dims differ.

    Args:
        student_channels: channel dim of student feature map.
        teacher_channels: channel dim of teacher feature map.
        method: "l2" | "at" | "l1"
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        method: str = "l2",
    ):
        super().__init__()
        self.method = method
        # Project student channels to match teacher if needed
        self.adaptor = (
            nn.Conv3d(student_channels, teacher_channels, 1, bias=False)
            if student_channels != teacher_channels
            else nn.Identity()
        )

    @staticmethod
    def _attention_map(feat: torch.Tensor) -> torch.Tensor:
        """Channel-wise L2 attention map, normalised."""
        a = feat.pow(2).mean(dim=1, keepdim=True)           # (B,1,H,W,D)
        return F.normalize(a.view(a.size(0), -1), dim=1)    # (B, H*W*D)

    def forward(
        self,
        student_feat: torch.Tensor,   # (B, Cs, H, W, D)
        teacher_feat: torch.Tensor,   # (B, Ct, H, W, D)  — no grad
    ) -> torch.Tensor:
        s = self.adaptor(student_feat)

        # Resize student to teacher spatial dims if patch sizes differ
        if s.shape[2:] != teacher_feat.shape[2:]:
            s = F.interpolate(s, size=teacher_feat.shape[2:], mode="trilinear",
                              align_corners=False)

        if self.method == "l2":
            return F.mse_loss(s, teacher_feat.detach())
        elif self.method == "l1":
            return F.l1_loss(s, teacher_feat.detach())
        elif self.method == "at":
            return F.mse_loss(
                self._attention_map(s),
                self._attention_map(teacher_feat.detach()),
            )
        else:
            raise ValueError(f"Unknown method '{self.method}'")
