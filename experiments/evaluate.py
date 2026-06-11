"""Multi-model comparison evaluation script.

Usage:
    python experiments/evaluate.py \
        --ckpts unet:logs/unet_oasis1/best_model.ckpt \
                dynunet:logs/dynunet_oasis1/best_model.ckpt \
        --config configs/unet.yaml
"""

from __future__ import annotations

import argparse

import torch
import yaml

from medsegkit.data.oasis_module import OasisDataModule
from medsegkit.engine.seg_module import SegModule
from medsegkit.evaluation.metrics import evaluate_model, print_comparison_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpts", nargs="+", metavar="NAME:PATH",
        help='List of "model_name:ckpt_path" pairs',
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dm = OasisDataModule(**cfg["data"])
    dm.setup(stage="test")
    loader = dm.test_dataloader()

    results = {}
    for entry in args.ckpts:
        name, ckpt_path = entry.split(":", 1)
        print(f"\nLoading {name} from {ckpt_path} ...")
        module = SegModule.load_from_checkpoint(ckpt_path, map_location=args.device)
        results[name] = evaluate_model(
            module.model,
            loader,
            device=args.device,
            patch_size=cfg["data"].get("patch_size", [128, 128, 128]),
        )

    print_comparison_table(results)


if __name__ == "__main__":
    main()
