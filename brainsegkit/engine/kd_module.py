"""Knowledge Distillation Lightning Module.

Teacher is frozen; student is trained with:
    total_loss = seg_loss(student, label)
               + kd_weight * kd_loss(student, teacher)

Supports all three KD types from brainsegkit.losses.kd:
    "response"   — soft-label KD (Hinton 2015)
    "feature"    — intermediate feature alignment
    "contrastive"— InfoNCE-based CRD
    "combined"   — response + feature
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import lightning as L
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from brainsegkit.models.registry import build_model
from brainsegkit.losses.seg_losses import build_seg_loss
from brainsegkit.losses.kd import ResponseKDLoss, FeatureKDLoss, ContrastiveKDLoss
from brainsegkit.data.transforms import NUM_CLASSES


class KDModule(L.LightningModule):
    """Teacher → Student Knowledge Distillation Module.

    Args:
        teacher_name / teacher_kwargs: Teacher model (loaded from checkpoint).
        teacher_ckpt:   Path to teacher .ckpt; None = random init (debug only).
        student_name / student_kwargs: Student model to train.
        kd_type:        "response" | "feature" | "contrastive" | "combined"
        kd_weight:      Weight of KD loss term.
        seg_loss_name:  Hard-label segmentation loss name.
        feature_layers: Which named sub-modules to hook for feature KD.
    """

    def __init__(
        self,
        teacher_name:    str   = "dynunet",
        teacher_kwargs:  dict  = {},
        teacher_ckpt:    str | None = None,
        student_name:    str   = "unet",
        student_kwargs:  dict  = {},
        kd_type:         str   = "response",
        kd_weight:       float = 0.5,
        seg_loss_name:   str   = "dice_ce",
        temperature:     float = 4.0,
        lr:              float = 1e-4,
        weight_decay:    float = 1e-5,
        patch_size:      tuple = (128, 128, 128),
        sw_batch_size:   int   = 4,
        num_classes:     int   = NUM_CLASSES,
        feature_layers:  list  = [],
    ):
        super().__init__()
        self.save_hyperparameters()

        # Teacher — frozen
        self.teacher = build_model(teacher_name, out_channels=num_classes, **teacher_kwargs)
        if teacher_ckpt:
            state = torch.load(teacher_ckpt, map_location="cpu")
            # SegModule wraps model under self.model
            self.teacher.load_state_dict(
                {k.replace("model.", "", 1): v
                 for k, v in state["state_dict"].items() if k.startswith("model.")},
                strict=False,
            )
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        # Student — trainable
        self.student = build_model(student_name, out_channels=num_classes, **student_kwargs)
        self.seg_loss = build_seg_loss(seg_loss_name)

        # KD losses
        self.kd_type   = kd_type
        self.kd_weight = kd_weight
        if kd_type in ("response", "combined"):
            self.response_kd = ResponseKDLoss(temperature=temperature, alpha=kd_weight)
        if kd_type in ("feature", "combined"):
            self.feature_kd = FeatureKDLoss(
                student_channels=student_kwargs.get("channels", (32,))[0],
                teacher_channels=teacher_kwargs.get("channels", (32,))[0],
            )
        if kd_type == "contrastive":
            self.contrastive_kd = ContrastiveKDLoss(
                student_dim=student_kwargs.get("channels", (32,))[-1],
                teacher_dim=teacher_kwargs.get("channels", (32,))[-1],
            )

        self.inferer = SlidingWindowInferer(
            roi_size=patch_size, sw_batch_size=sw_batch_size, overlap=0.5
        )
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean"
        )
        self._post_pred  = AsDiscrete(argmax=True, to_onehot=num_classes)
        self._post_label = AsDiscrete(to_onehot=num_classes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student(x)

    # ------------------------------------------------------------------
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]

        with torch.no_grad():
            teacher_out = self.teacher(images)
            t_logits = teacher_out[0] if isinstance(teacher_out, (list, tuple)) else teacher_out

        student_out = self.student(images)
        s_logits = student_out[0] if isinstance(student_out, (list, tuple)) else student_out

        seg_loss = self.seg_loss(s_logits, labels)

        if self.kd_type == "response":
            kd_loss = self.response_kd(s_logits, t_logits, labels)
            total   = kd_loss   # ResponseKDLoss already combines CE + KL
        elif self.kd_type == "feature":
            kd_loss = self.feature_kd(s_logits, t_logits)
            total   = seg_loss + self.kd_weight * kd_loss
        elif self.kd_type == "contrastive":
            kd_loss = self.contrastive_kd(s_logits, t_logits)
            total   = seg_loss + self.kd_weight * kd_loss
        elif self.kd_type == "combined":
            kd_loss = (
                self.response_kd(s_logits, t_logits, labels)
                + self.feature_kd(s_logits, t_logits)
            )
            total = seg_loss + self.kd_weight * kd_loss
        else:
            total = seg_loss
            kd_loss = torch.tensor(0.0)

        self.log("train/seg_loss", seg_loss, prog_bar=True, on_epoch=True)
        self.log("train/kd_loss",  kd_loss,  prog_bar=True, on_epoch=True)
        self.log("train/total",    total,    prog_bar=True, on_epoch=True)
        return total

    # ------------------------------------------------------------------
    def validation_step(self, batch: dict, batch_idx: int):
        images, labels = batch["image"], batch["label"]
        preds = self.inferer(inputs=images, network=self.student)
        preds_list  = [self._post_pred(p)  for p in decollate_batch(preds)]
        labels_list = [self._post_label(l) for l in decollate_batch(labels)]
        self.dice_metric(preds_list, labels_list)
        self.hd95_metric(preds_list, labels_list)

    def on_validation_epoch_end(self):
        self.log("val/dice", self.dice_metric.aggregate().item(), prog_bar=True)
        self.log("val/hd95", self.hd95_metric.aggregate().item(), prog_bar=True)
        self.dice_metric.reset()
        self.hd95_metric.reset()

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
