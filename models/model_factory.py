"""
Model factory built on top of the `timm` library.

Supported architecture families (selected from timm's 1000+ models):
  CNN          : resnet*, resnext*, densenet*, efficientnet*, convnext*,
                 mobilenet*, vgg*, regnet*
  ViT          : vit_*, deit_*, beit_*
  Swin         : swin_*, swinv2_*
  MaxViT       : maxvit_*
  NAS          : efficientnetv2_*, mnasnet*, nasnet*
  Hybrid       : coat_*, convit_*, pit_*

Usage
-----
    model = build_model(cfg)                 # cfg is the 'model' config section
    model = build_model(cfg, num_classes=5)  # override num_classes
"""

from __future__ import annotations

from typing import Dict, Optional

import timm
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(
    cfg: dict,
    num_classes: Optional[int] = None,
) -> nn.Module:
    """
    Create a timm model from the model config section.

    Parameters
    ----------
    cfg : dict
        Must contain ``name``.  Optional keys:
        ``pretrained``, ``num_classes``, ``drop_rate``, ``drop_path_rate``,
        ``global_pool``, ``checkpoint``.
    num_classes : int, optional
        Overrides ``cfg['num_classes']`` when provided.  Useful when the
        number of classes is derived from the dataset at runtime.

    Returns
    -------
    nn.Module
        The model ready for training (in train mode).
    """
    model_name     = cfg["name"]
    pretrained     = cfg.get("pretrained", True)
    n_classes      = num_classes if num_classes is not None else cfg.get("num_classes", 1000)
    drop_rate      = cfg.get("drop_rate", 0.0)
    drop_path_rate = cfg.get("drop_path_rate", 0.0)
    global_pool    = cfg.get("global_pool", "avg")

    # Build via timm
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=n_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        global_pool=global_pool,
    )

    # Optionally load a local checkpoint (feature extraction / fine-tuning)
    ckpt_path = cfg.get("checkpoint", None)
    if ckpt_path is not None:
        _load_checkpoint(model, ckpt_path, n_classes)

    return model


def list_models(filter_str: str = "") -> list[str]:
    """Convenience wrapper around ``timm.list_models``."""
    return timm.list_models(f"*{filter_str}*", pretrained=True)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(
    model: nn.Module,
    path: str,
    num_classes: int,
) -> None:
    """
    Load weights from *path* into *model*, handling the common cases:
      - exact match
      - classifier head size mismatch (head weights are skipped)
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Support various checkpoint formats
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]

    # Strip 'module.' prefix from DDP-saved checkpoints
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Filter out classifier head if shapes mismatch
    model_state = model.state_dict()
    filtered = {}
    skipped  = []
    for k, v in state.items():
        if k in model_state and v.shape == model_state[k].shape:
            filtered[k] = v
        else:
            skipped.append(k)

    if skipped:
        print(f"[model_factory] Skipped keys (shape mismatch / not found): {skipped}")

    model.load_state_dict(filtered, strict=False)
    print(f"[model_factory] Loaded weights from '{path}' ({len(filtered)} tensors).")


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    cfg: dict,
    scaler=None,
) -> None:
    """Save a full training checkpoint."""
    # Unwrap DDP
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "epoch":       epoch,
        "best_metric": best_metric,
        "model":       raw_model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict() if scheduler is not None else None,
        "scaler":      scaler.state_dict() if scaler is not None else None,
        "cfg":         cfg,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
) -> Dict:
    """
    Load a full training checkpoint.

    Returns the checkpoint dict so the caller can retrieve ``epoch`` and
    ``best_metric``.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    raw_model = model.module if hasattr(model, "module") else model
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    raw_model.load_state_dict(state, strict=True)

    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])

    print(
        f"[model_factory] Resumed from '{path}' "
        f"(epoch {ckpt['epoch']}, best_metric={ckpt['best_metric']:.4f})"
    )
    return ckpt
