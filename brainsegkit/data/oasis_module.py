"""Lightning DataModule for the OASIS-1 brain MRI dataset.

Directory layout expected:
    data_root/
        oasis1_freesurfer/{subject}/mri/T1.mgz
        oasis1_freesurfer/{subject}/mri/aseg.mgz

Splits are read from a CSV with columns: oasis_id, split
    split values: train | val | test
"""

from __future__ import annotations

import csv
from pathlib import Path

import lightning as L
from monai.data import CacheDataset, DataLoader

from brainsegkit.data.transforms import build_transforms


class OasisDataModule(L.LightningDataModule):
    """OASIS-1 FreeSurfer segmentation DataModule.

    Args:
        data_root:   Path to brain_data/.
        splits_csv:  Path to oasis1_splits.csv.
        patch_size:  3-D crop size for training.
        batch_size:  Per-GPU batch size.
        num_workers: DataLoader workers.
        cache_rate:  Fraction of dataset to cache in RAM (0.0–1.0).
    """

    def __init__(
        self,
        data_root:   str  = "/home/hank/medical_segmention/brain_data",
        splits_csv:  str  = "/home/hank/medical_segmention/oasis1_splits.csv",
        patch_size:  tuple = (128, 128, 128),
        batch_size:  int  = 2,
        num_workers: int  = 4,
        cache_rate:  float = 0.1,
    ):
        super().__init__()
        self.data_root   = Path(data_root)
        self.splits_csv  = Path(splits_csv)
        self.patch_size  = patch_size
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.cache_rate  = cache_rate

    # ------------------------------------------------------------------
    def _load_split(self, split: str) -> list[dict]:
        """Return list of {image, label} dicts for the requested split."""
        records = []
        with open(self.splits_csv) as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                sid = row["oasis_id"]
                mri_dir = self.data_root / "oasis1_freesurfer" / sid / "mri"
                image = mri_dir / "T1.mgz"
                label = mri_dir / "aseg.mgz"
                if image.exists() and label.exists():
                    records.append({"image": str(image), "label": str(label)})
        return records

    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        def _ds(split):
            return CacheDataset(
                data=self._load_split(split),
                transform=build_transforms(split, self.patch_size),
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
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
