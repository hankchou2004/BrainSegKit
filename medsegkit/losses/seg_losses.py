"""Segmentation losses — thin wrappers around MONAI with a factory function."""

from __future__ import annotations

import torch.nn as nn
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, TverskyLoss


def build_seg_loss(name: str, **kwargs) -> nn.Module:
    """Factory: returns a segmentation loss by name.

    Supported names: "dice", "dice_ce", "focal", "tversky"
    kwargs are forwarded to the MONAI constructor.
    """
    _LOSSES = {
        "dice":    DiceLoss,
        "dice_ce": DiceCELoss,
        "focal":   FocalLoss,
        "tversky": TverskyLoss,
    }
    if name not in _LOSSES:
        raise KeyError(f"Loss '{name}' not found. Available: {list(_LOSSES)}")

    defaults: dict = {
        "include_background": False,
        "to_onehot_y": True,
        "softmax": True,
    }
    # FocalLoss does not accept to_onehot_y
    if name == "focal":
        defaults.pop("to_onehot_y")

    defaults.update(kwargs)
    return _LOSSES[name](**defaults)
