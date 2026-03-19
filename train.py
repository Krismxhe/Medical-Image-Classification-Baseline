"""
Training entry point.

Single-GPU
----------
    python train.py --config configs/examples/resnet50.yaml \\
                    --data.root /data/fundus \\
                    --train.batch_size 64

Multi-GPU DDP (recommended launcher: torchrun)
----------------------------------------------
    torchrun --nproc_per_node 4 train.py --config configs/examples/resnet50.yaml \\
             --data.root /data/fundus

Key options
-----------
  --config          Path to a YAML config file (required)
  --base_config     Base config to merge with (default: configs/base.yaml)
  --<section>.<key> Override any config value on the CLI, e.g. --train.lr 1e-3
  --resume          Path to a checkpoint .pth to resume from
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# ── project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import MedicalImageDataset, build_dataloader
from data.transforms import get_transforms
from engine.trainer import run_training
from models.model_factory import build_model, load_checkpoint
from utils.logger import TrainingLogger
from utils.misc import flatten_config, get_output_dir, load_config, merge_configs, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medical image classification — train")
    parser.add_argument("--config",      required=True,                help="Model/experiment YAML config")
    parser.add_argument("--base_config", default="configs/base.yaml",  help="Base YAML config")
    parser.add_argument("--resume",      default=None,                  help="Checkpoint path to resume from")

    # Allow arbitrary --section.key value overrides
    args, unknown = parser.parse_known_args()
    overrides = _parse_overrides(unknown)
    args.overrides = overrides
    return args


def _parse_overrides(tokens: list[str]) -> dict:
    """Convert ['--train.lr', '1e-3', '--data.img_size', '256'] → nested dict."""
    overrides: dict = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok.lstrip("-")
            val = tokens[i + 1] if i + 1 < len(tokens) and not tokens[i + 1].startswith("--") else "true"
            i += 2 if val != "true" else 1
            # Nested key: "train.lr" → {"train": {"lr": ...}}
            parts = key.split(".")
            d = overrides
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = _cast(val)
        else:
            i += 1
    return overrides


def _cast(val: str):
    """Try to cast a CLI string to int / float / bool / str."""
    if val.lower() == "true":  return True
    if val.lower() == "false": return False
    try:    return int(val)
    except ValueError: pass
    try:    return float(val)
    except ValueError: pass
    return val


# ---------------------------------------------------------------------------
# DDP initialisation
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int, bool]:
    """
    Initialise DDP if launched via torchrun.

    Returns (rank, local_rank, world_size, is_main).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:
        rank = local_rank = 0
        world_size = 1

    is_main = rank == 0
    return rank, local_rank, world_size, is_main


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── DDP ──────────────────────────────────────────────────────────────────
    rank, local_rank, world_size, is_main = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    distributed = world_size > 1

    # ── Config ───────────────────────────────────────────────────────────────
    base_cfg  = load_config(args.base_config)
    exp_cfg   = load_config(args.config)
    cfg       = merge_configs(base_cfg, exp_cfg)
    cfg       = merge_configs(cfg, args.overrides)

    # CLI --resume overrides config
    if args.resume:
        cfg["resume"] = args.resume

    set_seed(cfg.get("seed", 42) + rank)   # different seed per rank for data aug

    # ── Output dir ───────────────────────────────────────────────────────────
    output_dir = get_output_dir(cfg)

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    data_cfg = cfg["data"]
    train_tf = get_transforms("train", data_cfg)
    val_tf   = get_transforms("val",   data_cfg)

    train_ds = MedicalImageDataset(
        root=data_cfg["root"], split="train", transform=train_tf,
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
    )
    val_ds = MedicalImageDataset(
        root=data_cfg["root"], split="val", transform=val_tf,
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        seed=cfg.get("seed", 42),
    )

    if is_main:
        print(f"\n{train_ds}")
        print(f"{val_ds}")
        print(f"Classes: {train_ds.classes}\n")

    class_weights = train_ds.get_class_weights()

    train_loader = build_dataloader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=data_cfg.get("pin_memory", True),
        use_weighted_sampler=data_cfg.get("use_weighted_sampler", True) and not distributed,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        seed=cfg.get("seed", 42),
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=cfg["train"]["batch_size"] * 2,
        num_workers=data_cfg.get("num_workers", 8),
        pin_memory=data_cfg.get("pin_memory", True),
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        seed=cfg.get("seed", 42),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes = len(train_ds.classes)
    model = build_model(cfg["model"], num_classes=num_classes)
    model = model.to(device)

    if is_main:
        from utils.misc import count_parameters
        print(f"Model: {cfg['model']['name']}  |  Parameters: {count_parameters(model):,}\n")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_ckpt = None
    if cfg.get("resume"):
        resume_ckpt = load_checkpoint(cfg["resume"], model)

    # ── Logger ────────────────────────────────────────────────────────────────
    log_cfg = cfg["logging"]
    training_logger = TrainingLogger(
        output_dir=output_dir,
        exp_name=log_cfg["exp_name"],
        use_tensorboard=log_cfg.get("use_tensorboard", True),
        use_wandb=log_cfg.get("use_wandb", False),
        wandb_project=log_cfg.get("wandb_project", "medical-cls"),
        cfg=flatten_config(cfg),
        is_main=is_main,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
            class_names=train_ds.classes,
            class_weights=class_weights,
            training_logger=training_logger,
            resume_ckpt=resume_ckpt,
            world_size=world_size,
            rank=rank,
        )
    finally:
        training_logger.close()
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
