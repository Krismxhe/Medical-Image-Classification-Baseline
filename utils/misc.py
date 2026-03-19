"""
Miscellaneous utilities: seeds, AverageMeter, EarlyStopping, config loading.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all relevant RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA ops (may slow training)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file and return as nested dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep-merge *override* into *base*.  Override values take precedence.
    Both dicts are left unchanged; the merged result is returned.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = v
    return result


def flatten_config(cfg: dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict for logging (e.g. to W&B)."""
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_config(v, key))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------------------
# AverageMeter — tracks running mean of a scalar (e.g. loss)
# ---------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when a monitored metric has stopped improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement after which training stops.
    mode : str
        ``'max'`` (higher is better, e.g. AUC) or ``'min'`` (lower is better,
        e.g. loss).
    min_delta : float
        Minimum change that qualifies as an improvement.
    """

    def __init__(self, patience: int = 15, mode: str = "max", min_delta: float = 1e-4) -> None:
        assert mode in ("min", "max")
        self.patience  = patience
        self.mode      = mode
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("-inf") if mode == "max" else float("inf")
        self.stopped   = False

    def step(self, metric: float) -> bool:
        """
        Call once per epoch.

        Returns ``True`` if training should stop.
        """
        improved = (
            metric > self.best + self.min_delta
            if self.mode == "max"
            else metric < self.best - self.min_delta
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped

    @property
    def improved(self) -> bool:
        return self.counter == 0


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def is_dist_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    return torch.distributed.get_rank() if is_dist_available() else 0


def get_world_size() -> int:
    return torch.distributed.get_world_size() if is_dist_available() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-reduce a scalar tensor and divide by world_size."""
    rt = tensor.clone()
    torch.distributed.all_reduce(rt)
    rt /= world_size
    return rt


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_output_dir(cfg: dict) -> Path:
    """Construct and create the experiment output directory."""
    base = Path(cfg["logging"]["output_dir"])
    name = cfg["logging"]["exp_name"]
    out  = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out
