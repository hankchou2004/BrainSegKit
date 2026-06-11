"""Lightning DataModule for the OASIS-1 brain MRI dataset.

Directory layout expected (nnUNet-style, .nii.gz):
    dataset_root/
        freesurfer/
            imagesTr/  {subject}_0000.nii.gz   (train + val, T1)
            labelsTr/  {subject}.nii.gz          (train + val, aseg — 41 classes)
            imagesTs/  {subject}_0000.nii.gz
            labelsTs/  {subject}.nii.gz
        fsl/
            imagesTr/  {subject}_0000.nii.gz   (train + val, same T1)
            labelsTr/  {subject}.nii.gz          (train + val, fseg — 3 classes)
            imagesTs/  {subject}_0000.nii.gz
            labelsTs/  {subject}.nii.gz

Splits are read from splits.json:
    { "splits": { "train": [...], "val": [...], "test": [...] } }
Both modalities share the same JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
from monai.data import CacheDataset, DataLoader

from brainsegkit.data.transforms import build_transforms

MODALITIES = ("freesurfer", "fsl")


class OasisDataModule(L.LightningDataModule):
    """OASIS-1 segmentation DataModule — supports FreeSurfer and FSL labels.

    Args:
        dataset_root: Path to dataset/  (contains freesurfer/ and fsl/ subdirs).
        splits_csv:   Path to oasis1_splits.csv.
        modality:     "freesurfer" (41-class aseg) or "fsl" (3-class CSF/GM/WM).
        patch_size:   3-D crop size for training.
        batch_size:   Per-GPU batch size.
        num_workers:  DataLoader workers.
        cache_rate:   Fraction of dataset to cache in RAM (0.0–1.0).
    """

    def __init__(
        self,
        dataset_root: str   = "/home/hank/medical_segmention/dataset",
        splits_json:  str   = "/home/hank/medical_segmention/dataset/splits.json",
        modality:     str   = "freesurfer",
        patch_size:   tuple = (128, 128, 128),
        batch_size:   int   = 2,
        num_workers:  int   = 4,
        cache_rate:   float = 0.1,
    ):
        super().__init__()
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got '{modality}'")
        self.modality_root = Path(dataset_root) / modality
        self.splits_json   = Path(splits_json)
        self.modality      = modality
        self.patch_size    = patch_size
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.cache_rate    = cache_rate

    # ------------------------------------------------------------------
    def _load_split(self, split: str) -> list[dict]:
        """Return list of {image, label} dicts for the requested split."""
        img_dir = self.modality_root / ("imagesTr" if split in ("train", "val") else "imagesTs")
        lbl_dir = self.modality_root / ("labelsTr" if split in ("train", "val") else "labelsTs")

        with open(self.splits_json) as f:
            subject_ids: list[str] = json.load(f)["splits"][split]

        records = []
        for sid in subject_ids:
            image = img_dir / f"{sid}_0000.nii.gz"
            label = lbl_dir / f"{sid}.nii.gz"
            if image.exists() and label.exists():
                records.append({"image": str(image), "label": str(label)})
        return records

    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        def _ds(split):
            return CacheDataset(
                data=self._load_split(split),
                transform=build_transforms(split, self.patch_size, self.modality),
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
