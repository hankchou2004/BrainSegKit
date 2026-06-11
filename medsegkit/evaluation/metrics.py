"""Evaluation utilities — multi-model comparison table.

Computes per-class and mean Dice, HD95, NSD for one or more models
and prints a formatted comparison table.
"""

from __future__ import annotations

import torch
import numpy as np
from monai.inferers import SlidingWindowInferer
from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
)
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from medsegkit.data.transforms import NUM_CLASSES, ASEG_SRC


# Human-readable names for the 36 aseg classes
CLASS_NAMES = [
    "Background", "L-WM", "L-Cortex", "L-LatVent", "L-InfLatVent",
    "L-Cereb-WM", "L-Cereb-Ctx", "L-Thalamus", "L-Caudate", "L-Putamen",
    "L-Pallidum", "3rd-Vent", "4th-Vent", "Brain-Stem", "L-Hippocampus",
    "L-Amygdala", "CSF", "L-Accumbens", "L-VentralDC",
    "R-WM", "R-Cortex", "R-LatVent", "R-InfLatVent",
    "R-Cereb-WM", "R-Cereb-Ctx", "R-Thalamus", "R-Caudate", "R-Putamen",
    "R-Pallidum", "R-Hippocampus", "R-Amygdala", "R-Accumbens",
    "R-VentralDC", "WM-Hypo", "Optic-Chiasm",
]


def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: str = "cuda",
    patch_size: tuple = (128, 128, 128),
) -> dict:
    """Run sliding-window inference and return metric dict.

    Returns:
        {
            "mean_dice": float,
            "mean_hd95": float,
            "mean_nsd":  float,
            "per_class_dice": list[float],  # length NUM_CLASSES-1 (no bg)
        }
    """
    model.eval().to(device)
    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=4, overlap=0.5)

    dice_m = DiceMetric(include_background=False, reduction="none")
    hd95_m = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
    nsd_m  = SurfaceDistanceMetric(include_background=False, reduction="none")

    post_pred  = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            preds  = inferer(inputs=images, network=model)

            pl = [post_pred(p)  for p in decollate_batch(preds)]
            ll = [post_label(l) for l in decollate_batch(labels)]
            dice_m(pl, ll)
            hd95_m(pl, ll)
            nsd_m(pl, ll)

    dice_per_class = dice_m.aggregate().nanmean(dim=0).cpu().numpy()   # (C-1,)
    hd95_per_class = hd95_m.aggregate().nanmean(dim=0).cpu().numpy()
    nsd_per_class  = nsd_m.aggregate().nanmean(dim=0).cpu().numpy()

    return {
        "mean_dice":      float(np.nanmean(dice_per_class)),
        "mean_hd95":      float(np.nanmean(hd95_per_class)),
        "mean_nsd":       float(np.nanmean(nsd_per_class)),
        "per_class_dice": dice_per_class.tolist(),
    }


def print_comparison_table(results: dict[str, dict]) -> None:
    """Print a comparison table.

    Args:
        results: {"model_name": evaluate_model(...), ...}
    """
    header = f"{'Model':<20} {'Mean Dice':>10} {'Mean HD95':>10} {'Mean NSD':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, r in results.items():
        print(
            f"{name:<20} {r['mean_dice']:>10.4f} "
            f"{r['mean_hd95']:>10.2f} {r['mean_nsd']:>10.4f}"
        )
    print("=" * len(header) + "\n")
