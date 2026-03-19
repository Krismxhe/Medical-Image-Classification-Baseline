"""
Standalone evaluation / inference script.

Usage
-----
# Evaluate on test split
python eval.py --config configs/examples/resnet50.yaml \\
               --checkpoint outputs/resnet50/best.pth \\
               --split test \\
               --data.root /data/fundus

# Evaluate on val split
python eval.py --config configs/examples/resnet50.yaml \\
               --checkpoint outputs/resnet50/best.pth \\
               --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import MedicalImageDataset, build_dataloader
from data.transforms import get_transforms
from engine.evaluator import evaluate
from models.model_factory import build_model, load_checkpoint
from utils.metrics import format_metrics
from utils.misc import load_config, merge_configs, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medical image classification — evaluate")
    parser.add_argument("--config",      required=True,               help="Experiment YAML config")
    parser.add_argument("--base_config", default="configs/base.yaml", help="Base YAML config")
    parser.add_argument("--checkpoint",  required=True,               help="Path to .pth checkpoint")
    parser.add_argument("--split",       default="test",              choices=["train", "val", "test"])
    parser.add_argument("--output_json", default=None,                help="Save metrics to this JSON file")

    args, unknown = parser.parse_known_args()
    # Re-use the same override parser from train.py
    from train import _parse_overrides
    args.overrides = _parse_overrides(unknown)
    return args


def main() -> None:
    args = parse_args()

    base_cfg = load_config(args.base_config)
    exp_cfg  = load_config(args.config)
    cfg      = merge_configs(base_cfg, exp_cfg)
    cfg      = merge_configs(cfg, args.overrides)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    tf = get_transforms(args.split, data_cfg)

    ds = MedicalImageDataset(
        root=data_cfg["root"],
        split=args.split,
        transform=tf,
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
    )
    logger.info(ds)

    loader = build_dataloader(
        ds,
        batch_size=cfg["train"]["batch_size"] * 2,
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes = len(ds.classes)
    model = build_model(cfg["model"], num_classes=num_classes).to(device)
    load_checkpoint(args.checkpoint, model)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(
        model, loader, device,
        class_names=ds.classes,
        amp_enabled=cfg["train"].get("amp", True),
    )

    # ── Report ────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  Split      : {args.split}")
    logger.info(f"  Checkpoint : {args.checkpoint}")
    logger.info(f"  N samples  : {len(ds)}")
    logger.info(f"  Classes    : {ds.classes}")
    logger.info(f"{'='*60}")
    logger.info(f"  {format_metrics(metrics)}")
    logger.info(f"{'='*60}")
    logger.info(f"\nClassification Report:\n{metrics['report']}")
    logger.info(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")

    if args.output_json:
        save_metrics = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in metrics.items()
            if k != "report"
        }
        Path(args.output_json).write_text(json.dumps(save_metrics, indent=2))
        logger.info(f"\nMetrics saved to {args.output_json}")


if __name__ == "__main__":
    main()
