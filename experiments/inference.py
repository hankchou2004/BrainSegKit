"""MedSegKit inference script.

Runs sliding-window inference on a dataset split or a single image file,
and saves predicted segmentation masks as .nii.gz files.

Usage — dataset split:
    # BTCV test split (20 cases, no labels — inference only)
    python experiments/inference.py \
        --config configs/btcv_dynunet.yaml \
        --ckpt   logs/dynunet_btcv/best_model.ckpt \
        --split  test \
        --out    ./predictions/btcv_test

    # OASIS-1 val split
    python experiments/inference.py \
        --config configs/unet.yaml \
        --ckpt   logs/unet_oasis1/best_model.ckpt \
        --split  val \
        --out    ./predictions/oasis1_val

Usage — single image:
    python experiments/inference.py \
        --config configs/btcv_dynunet.yaml \
        --ckpt   logs/dynunet_btcv/best_model.ckpt \
        --image  /path/to/img0061.nii.gz \
        --out    ./predictions

Output:
    One .nii.gz per input, named after the source file stem.
    Saved in the resampled space (same spacing as during training).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.inferers import SlidingWindowInferer

from medsegkit.engine.seg_module import SegModule


# ── helpers ───────────────────────────────────────────────────────────────────

def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_transform(dataset: str, cfg: dict):
    """Return the val/test MONAI transform pipeline (no augmentation)."""
    patch_size = tuple(cfg["data"].get("patch_size", [96, 96, 96]))
    spacing    = tuple(cfg["data"].get("spacing",    [1.5, 1.5, 2.0]))

    if dataset == "btcv":
        from medsegkit.data.btcv_transforms import build_btcv_transforms
        return build_btcv_transforms("val", patch_size, spacing)
    else:
        from medsegkit.data.transforms import build_transforms
        return build_transforms("val", patch_size, spacing)


def collect_images(dataset: str, split: str, cfg: dict) -> list[dict]:
    """Build list of {image, label} dicts for the requested split."""
    import json

    dataset_root = Path(cfg["data"].get("dataset_root",
                                        "/home/hank/medical_segmention/dataset"))

    if dataset == "btcv":
        btcv_root  = dataset_root / "btcv"
        splits_json = Path(cfg["data"].get("splits_json",
                                           str(btcv_root / "splits.json")))
        with open(splits_json) as f:
            ids = json.load(f)["splits"][split]

        records = []
        for sid in ids:
            if split == "test":
                img = btcv_root / "Testing" / "img" / f"img{sid}.nii.gz"
                if img.exists():
                    records.append({"image": str(img), "label": str(img)})
            else:
                img = btcv_root / "Training" / "img"   / f"img{sid}.nii.gz"
                lbl = btcv_root / "Training" / "label" / f"label{sid}.nii.gz"
                if img.exists():
                    records.append({"image": str(img),
                                    "label": str(lbl) if lbl.exists() else str(img)})
        return records

    else:  # oasis1
        fs_root    = dataset_root / "oasis1_freesurfer"
        splits_json = Path(cfg["data"].get("splits_json",
                                           str(fs_root / "splits.json")))
        with open(splits_json) as f:
            ids = json.load(f)["splits"][split]

        img_dir = fs_root / ("imagesTs" if split == "test" else "imagesTr")
        lbl_dir = fs_root / ("labelsTs" if split == "test" else "labelsTr")

        records = []
        for sid in ids:
            img = img_dir / f"{sid}_0000.nii.gz"
            lbl = lbl_dir / f"{sid}.nii.gz"
            if img.exists():
                records.append({"image": str(img),
                                "label": str(lbl) if lbl.exists() else str(img)})
        return records


def save_prediction(pred_np: np.ndarray, affine: np.ndarray,
                    out_path: Path) -> None:
    """Save (H, W, D) int16 prediction array as .nii.gz."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred_np.astype(np.int16), affine), str(out_path))


# ── main ──────────────────────────────────────────────────────────────────────

def run_inference(cfg: dict, ckpt: str, records: list[dict],
                  transform, out_dir: Path, device: str) -> None:
    patch_size  = tuple(cfg["data"].get("patch_size", [96, 96, 96]))
    sw_batch    = cfg["data"].get("batch_size", 1)

    print(f"Loading checkpoint: {ckpt}")
    module = SegModule.load_from_checkpoint(ckpt, map_location=device)
    module.to(device).eval()

    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch,
        overlap=0.5,
        mode="gaussian",
    )

    print(f"Running inference on {len(records)} case(s) → {out_dir}\n")
    for i, rec in enumerate(records):
        stem = Path(rec["image"]).name.replace(".nii.gz", "").replace(".nii", "")
        out_path = out_dir / f"{stem}_pred.nii.gz"

        data = transform(rec)
        image = data["image"].unsqueeze(0).to(device)          # (1,1,H,W,D)
        affine = data["image"].meta["affine"].numpy()           # (4,4)

        with torch.no_grad():
            logits = inferer(inputs=image, network=module.model)  # (1,C,H,W,D)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()   # (H,W,D)

        save_prediction(pred, affine, out_path)
        print(f"  [{i+1:3d}/{len(records)}]  {stem}  →  {out_path.name}")

    print(f"\nDone. {len(records)} file(s) saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="MedSegKit inference")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--ckpt",   required=True, help="Checkpoint .ckpt path")
    parser.add_argument("--out",    required=True, help="Output directory")
    parser.add_argument("--split",  default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to infer on (default: test)")
    parser.add_argument("--image",  default=None,
                        help="Single image .nii.gz (overrides --split)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg     = load_cfg(args.config)
    dataset = cfg["data"].get("dataset", "oasis1")
    out_dir = Path(args.out)
    tf      = build_transform(dataset, cfg)

    if args.image:
        records = [{"image": args.image, "label": args.image}]
    else:
        records = collect_images(dataset, args.split, cfg)
        if not records:
            raise FileNotFoundError(
                f"No files found for dataset='{dataset}' split='{args.split}'. "
                "Check dataset_root in config."
            )
        if dataset == "btcv" and args.split == "test":
            print("[inference] BTCV test split: no ground-truth labels available "
                  "(inference only).")

    run_inference(cfg, args.ckpt, records, tf, out_dir, args.device)


if __name__ == "__main__":
    main()
