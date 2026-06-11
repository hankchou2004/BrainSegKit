"""Generic Lightning training module for segmentation models.

Supports:
  - DynUNet deep supervision (returns list of tensors)
  - Sliding-window inference at validation / test
  - Dice + HD95 metrics per epoch
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

from medsegkit.losses.seg_losses import build_seg_loss
from medsegkit.models.registry import build_model
from medsegkit.data.transforms import NUM_CLASSES


class SegModule(L.LightningModule):
    """Segmentation Lightning Module.

    Args:
        model_name:   Key in MODEL_REGISTRY (e.g. "unet", "dynunet").
        model_kwargs: Forwarded to build_model().
        loss_name:    "dice" | "dice_ce" | "focal" | "tversky"
        loss_kwargs:  Forwarded to build_seg_loss().
        lr:           Peak learning rate.
        weight_decay: AdamW weight decay.
        patch_size:   Sliding-window patch size for val/test inference.
        sw_batch_size:Number of patches per sliding-window step.
        num_classes:  Number of segmentation classes.
    """

    def __init__(
        self,
        model_name:    str   = "unet",
        model_kwargs:  dict  = {},
        loss_name:     str   = "dice_ce",
        loss_kwargs:   dict  = {},
        lr:            float = 1e-4,
        weight_decay:  float = 1e-5,
        patch_size:    tuple = (128, 128, 128),
        sw_batch_size: int   = 4,
        num_classes:   int   = NUM_CLASSES,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(model_name, out_channels=num_classes, **model_kwargs)
        self.loss_fn = build_seg_loss(loss_name, **loss_kwargs)

        self.inferer = SlidingWindowInferer(
            roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            overlap=0.5,
            mode="gaussian",
        )

        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.hd95_metric = HausdorffDistanceMetric(
            include_background=False, percentile=95, reduction="mean"
        )

        self._post_pred  = AsDiscrete(argmax=True, to_onehot=num_classes)
        self._post_label = AsDiscrete(to_onehot=num_classes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels = batch["image"], batch["label"]
        preds = self(images)

        # DynUNet with deep supervision returns a list; use only the first
        if isinstance(preds, (list, tuple)):
            loss = sum(self.loss_fn(p, labels) for p in preds) / len(preds)
        else:
            loss = self.loss_fn(preds, labels)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    def _shared_eval(self, batch: dict, prefix: str):
        images, labels = batch["image"], batch["label"]
        preds = self.inferer(inputs=images, network=self.model)

        preds_list  = [self._post_pred(p)  for p in decollate_batch(preds)]
        labels_list = [self._post_label(l) for l in decollate_batch(labels)]

        self.dice_metric(preds_list, labels_list)
        self.hd95_metric(preds_list, labels_list)

    def on_validation_epoch_end(self):
        self._log_metrics("val")

    def on_test_epoch_end(self):
        self._log_metrics("test")

    def _log_metrics(self, prefix: str):
        dice = self.dice_metric.aggregate().item()
        hd95 = self.hd95_metric.aggregate().item()
        self.log(f"{prefix}/dice", dice, prog_bar=True)
        self.log(f"{prefix}/hd95", hd95, prog_bar=True)
        self.dice_metric.reset()
        self.hd95_metric.reset()

    def validation_step(self, batch: dict, batch_idx: int):
        self._shared_eval(batch, "val")

    def test_step(self, batch: dict, batch_idx: int):
        self._shared_eval(batch, "test")

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
