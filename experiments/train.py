"""BrainSegKit training entry point.

Usage:
    python experiments/train.py --config configs/unet.yaml
    python experiments/train.py --config configs/dynunet.yaml --gpus 1
    python experiments/train.py --config configs/kd/dynunet_to_unet.yaml --mode kd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from brainsegkit.data.oasis_module import OasisDataModule
from brainsegkit.engine.seg_module import SegModule
from brainsegkit.engine.kd_module import KDModule


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_logger(cfg: dict):
    exp = cfg.get("experiment", {})
    log_dir = exp.get("log_dir", "./logs")
    name    = exp.get("name", "brainsegkit")
    try:
        return WandbLogger(project="BrainSegKit", name=name, save_dir=log_dir)
    except Exception:
        return TensorBoardLogger(save_dir=log_dir, name=name)


def train_seg(cfg: dict, gpus: int):
    dm = OasisDataModule(**cfg["data"])

    m_cfg = cfg["model"]
    t_cfg = cfg["training"]
    module = SegModule(
        model_name=m_cfg.pop("name"),
        model_kwargs=m_cfg,
        loss_name=t_cfg.get("loss", "dice_ce"),
        lr=t_cfg.get("lr", 1e-4),
        weight_decay=t_cfg.get("weight_decay", 1e-5),
        patch_size=cfg["data"].get("patch_size", [128, 128, 128]),
    )

    _run(module, dm, cfg, gpus)


def train_kd(cfg: dict, gpus: int):
    dm = OasisDataModule(**cfg["data"])

    kd_cfg = cfg["kd"]
    t_cfg  = cfg["training"]
    module = KDModule(
        teacher_name=kd_cfg["teacher"]["name"],
        teacher_ckpt=kd_cfg["teacher"].get("ckpt"),
        student_name=kd_cfg["student"]["name"],
        student_kwargs={k: v for k, v in kd_cfg["student"].items() if k != "name"},
        kd_type=kd_cfg.get("type", "response"),
        kd_weight=kd_cfg.get("kd_weight", 0.5),
        temperature=kd_cfg.get("temperature", 4.0),
        seg_loss_name=t_cfg.get("seg_loss", "dice_ce"),
        lr=t_cfg.get("lr", 1e-4),
        weight_decay=t_cfg.get("weight_decay", 1e-5),
        patch_size=cfg["data"].get("patch_size", [128, 128, 128]),
    )

    _run(module, dm, cfg, gpus)


def _run(module, dm, cfg: dict, gpus: int):
    exp_name = cfg.get("experiment", {}).get("name", "brainsegkit")
    log_dir  = cfg.get("experiment", {}).get("log_dir", "./logs")
    max_epochs = cfg["training"].get("max_epochs", 300)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(Path(log_dir) / exp_name),
            filename="best_model",
            monitor="val/dice",
            mode="max",
            save_top_k=1,
        ),
        EarlyStopping(monitor="val/dice", patience=50, mode="max"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,
        callbacks=callbacks,
        logger=build_logger(cfg),
        log_every_n_steps=10,
        precision="16-mixed",
    )
    trainer.fit(module, dm)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode",   default="seg", choices=["seg", "kd"])
    parser.add_argument("--gpus",   type=int, default=1)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    if args.mode == "kd":
        train_kd(cfg, args.gpus)
    else:
        train_seg(cfg, args.gpus)
