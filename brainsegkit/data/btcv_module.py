"""Lightning DataModule for the BTCV abdominal CT segmentation dataset (13 organs).

Directory layout expected:
    dataset_root/
        btcv/
            Training/
                img/    img{id}.nii.gz   (CT, HU int16)
                label/  label{id}.nii.gz (0–13)
            Testing/
                img/    img{id}.nii.gz   (no labels)
            splits.json

Splits JSON format:
    { "splits": { "train": ["0001", ...], "val": [...], "test": [...] } }

IDs are zero-padded 4-digit strings (e.g. "0001").
Training/val images live in Training/img & Training/label.
Test images live in Testing/img (no labels available).
"""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
from monai.data import CacheDataset, DataLoader

from brainsegkit.data.btcv_transforms import build_btcv_transforms

# BTCV organ class names (index = label value)
BTCV_CLASSES = [
    "Background",
    "Spleen", "Right Kidney", "Left Kidney", "Gallbladder",
    "Esophagus", "Liver", "Stomach", "Aorta",
    "Inferior Vena Cava", "Portal Vein", "Pancreas",
    "Right Adrenal Gland", "Left Adrenal Gland",
]


class BTCVDataModule(L.LightningDataModule):
    """BTCV abdominal CT segmentation DataModule (14 classes including background).

    Args:
        dataset_root: Path to dataset/ (contains btcv/ subdir).
        splits_json:  Path to splits.json inside btcv/.
        patch_size:   3-D crop size for training.
        spacing:      Target voxel spacing (mm).
        batch_size:   Per-GPU batch size.
        num_workers:  DataLoader workers.
        cache_rate:   Fraction of dataset to cache in RAM (0.0–1.0).
    """

    def __init__(
        self,
        dataset_root: str   = "/home/hank/medical_segmention/dataset",
        splits_json:  str   = "/home/hank/medical_segmention/dataset/btcv/splits.json",
        patch_size:   tuple = (96, 96, 96),
        spacing:      tuple = (1.5, 1.5, 2.0),
        batch_size:   int   = 2,
        num_workers:  int   = 4,
        cache_rate:   float = 0.1,
    ):
        super().__init__()
        self.btcv_root   = Path(dataset_root) / "btcv"
        self.splits_json = Path(splits_json)
        self.patch_size  = patch_size
        self.spacing     = spacing
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.cache_rate  = cache_rate

    # ------------------------------------------------------------------
    def _load_split(self, split: str) -> list[dict]:
        with open(self.splits_json) as f:
            ids: list[str] = json.load(f)["splits"][split]

        records = []
        for sid in ids:
            if split == "test":
                image = self.btcv_root / "Testing" / "img" / f"img{sid}.nii.gz"
                if image.exists():
                    records.append({"image": str(image), "label": str(image)})
            else:
                image = self.btcv_root / "Training" / "img"   / f"img{sid}.nii.gz"
                label = self.btcv_root / "Training" / "label" / f"label{sid}.nii.gz"
                if image.exists() and label.exists():
                    records.append({"image": str(image), "label": str(label)})
        return records

    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        def _ds(split):
            return CacheDataset(
                data=self._load_split(split),
                transform=build_btcv_transforms(split, self.patch_size, self.spacing),
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )

        if stage in ("fit", None):
            self.train_ds = _ds("train")
            self.val_ds   = _ds("val")
        if stage in ("test", None):
            self.test_ds  = _ds("test")

    # ------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=1,
            shuffle=False, num_workers=self.num_workers, pin_memory=True,
        )
