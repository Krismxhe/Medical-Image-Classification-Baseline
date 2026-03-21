"""
Training engine.

Features
--------
• Mixed precision (torch.amp)
• Gradient accumulation
• DDP-aware loss reduction
• Learning-rate schedulers: cosine (+ linear warmup), step, plateau
• Focal loss + label-smoothing CE + class-weighted CE
• Early stopping
• Best-checkpoint saving
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from engine.evaluator import evaluate
from utils.logger import TrainingLogger
from utils.metrics import format_metrics
from utils.misc import AverageMeter, EarlyStopping, is_main_process, reduce_tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        p_t = torch.exp(-ce)
        focal = (1 - p_t) ** self.gamma * ce
        return focal.mean()


def build_criterion(cfg: dict, class_weights: Optional[np.ndarray], device: torch.device) -> nn.Module:
    """Build the loss function from the config."""
    loss_cfg  = cfg.get("loss", {})
    name      = loss_cfg.get("name", "ce")
    smoothing = loss_cfg.get("label_smoothing", 0.0)
    gamma     = loss_cfg.get("focal_gamma", 2.0)

    weight_tensor = None
    if loss_cfg.get("use_class_weights", True) and class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        logger.info(f"Class weights: {np.round(class_weights, 4)}")

    if name == "focal":
        return FocalLoss(gamma=gamma, weight=weight_tensor, label_smoothing=smoothing)
    else:
        # 'ce' or 'label_smoothing'
        return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=smoothing)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    name    = opt_cfg.get("name", "adamw").lower()
    lr      = opt_cfg.get("lr", 1e-4)
    wd      = opt_cfg.get("weight_decay", 0.05)
    mom     = opt_cfg.get("momentum", 0.9)

    # Separate weight-decay params (skip bias / norm layers)
    decay_params     = []
    no_decay_params  = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias"):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params,    "weight_decay": wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=mom, nesterov=True)
    elif name == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    else:  # adamw
        return torch.optim.AdamW(param_groups, lr=lr)


# ---------------------------------------------------------------------------
# Scheduler (with optional linear warmup)
# ---------------------------------------------------------------------------

def build_scheduler(cfg: dict, optimizer: torch.optim.Optimizer, n_epochs: int):
    sched_cfg     = cfg.get("scheduler", {})
    name          = sched_cfg.get("name", "cosine")
    warmup_epochs = sched_cfg.get("warmup_epochs", 5)
    min_lr        = sched_cfg.get("min_lr", 1e-6)
    step_size     = sched_cfg.get("step_size", 30)
    gamma         = sched_cfg.get("gamma", 0.1)
    base_lr       = cfg["optimizer"].get("lr", 1e-4)

    if name == "cosine":
        # Linear warmup → cosine decay
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return max(1e-6, epoch / max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr / base_lr, cosine)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=gamma, patience=5, min_lr=min_lr
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")


# ---------------------------------------------------------------------------
# One-epoch training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: dict,
    training_logger: Optional[TrainingLogger] = None,
    world_size: int = 1,
) -> Dict[str, float]:
    model.train()

    amp_enabled   = cfg["train"].get("amp", True)
    accum_steps   = cfg["train"].get("grad_accum_steps", 1)
    clip_norm     = cfg["train"].get("clip_grad_norm", 1.0)
    log_interval  = cfg["logging"].get("log_interval", 10)

    loss_meter = AverageMeter("loss")
    optimizer.zero_grad()

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch:3d} [train]",
        leave=True,
    )

    global_step_offset = epoch * len(dataloader)

    for step, (images, labels) in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss   = criterion(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            if clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Reduce loss across GPUs for display
        loss_val = loss.item() * accum_steps
        if world_size > 1:
            loss_t = torch.tensor(loss_val, device=device)
            torch.distributed.all_reduce(loss_t)
            loss_val = (loss_t / world_size).item()

        loss_meter.update(loss_val, images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        # Log to TensorBoard / W&B every N steps
        if is_main_process() and training_logger and step % log_interval == 0:
            global_step = global_step_offset + step
            training_logger.log_scalars(
                {"train/loss_step": loss_val,
                 "train/lr": optimizer.param_groups[0]["lr"]},
                step=global_step,
            )

    return {"loss": loss_meter.avg}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: dict,
    device: torch.device,
    output_dir: Path,
    class_names: List[str],
    class_weights: Optional[np.ndarray] = None,
    training_logger: Optional[TrainingLogger] = None,
    resume_ckpt: Optional[dict] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """
    Full training loop with validation, early stopping, and checkpointing.
    """
    is_main = is_main_process()
    n_epochs = cfg["train"]["epochs"]

    criterion = build_criterion(cfg, class_weights, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, n_epochs)
    scaler    = GradScaler(enabled=cfg["train"].get("amp", True))

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_metric  = float("-inf")
    if resume_ckpt is not None:
        start_epoch = resume_ckpt["epoch"] + 1
        best_metric = resume_ckpt["best_metric"]
        if resume_ckpt.get("optimizer"):
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if resume_ckpt.get("scheduler") and scheduler:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
        if resume_ckpt.get("scaler"):
            scaler.load_state_dict(resume_ckpt["scaler"])

    # ── Early stopping ────────────────────────────────────────────────────────
    es_cfg    = cfg.get("early_stopping", {})
    es_active = es_cfg.get("enabled", True)
    monitor   = es_cfg.get("monitor", "auc")
    mode      = es_cfg.get("mode", "max")
    early_stopper = EarlyStopping(
        patience=es_cfg.get("patience", 15),
        mode=mode,
    ) if es_active else None

    amp_enabled = cfg["train"].get("amp", True)

    for epoch in range(start_epoch, n_epochs):
        # Synchronise DDP sampler epoch (reshuffles each epoch)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # ── Train ─────────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, cfg, training_logger, world_size,
        )

        # ── Validate ──────────────────────────────────────────────────────────
        val_metrics = evaluate(
            model, val_loader, device,
            class_names=class_names,
            amp_enabled=amp_enabled,
        )

        # ── LR scheduler step ─────────────────────────────────────────────────
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics.get(monitor, val_metrics["acc"]))
            else:
                scheduler.step()

        current_metric = val_metrics.get(monitor, val_metrics["acc"])

        # ── Logging ───────────────────────────────────────────────────────────
        if is_main:
            scalar_val = {k: v for k, v in val_metrics.items()
                          if isinstance(v, (int, float))}
            if training_logger:
                training_logger.log_scalars(
                    {**{f"train/{k}": v for k, v in train_metrics.items()},
                     **{f"val/{k}":   v for k, v in scalar_val.items()},
                     "lr": optimizer.param_groups[0]["lr"]},
                    step=epoch,
                )
                training_logger.log_confusion_matrix(
                    val_metrics["confusion_matrix"], class_names, step=epoch
                )

            logger.info(
                f"Epoch {epoch:3d}/{n_epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  "
                + format_metrics(val_metrics, prefix="val_")
                + f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        # ── Save best checkpoint ───────────────────────────────────────────────
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric

        if is_main:
            from models.model_factory import save_checkpoint
            save_checkpoint(
                path=str(output_dir / "last.pth"),
                model=model, optimizer=optimizer,
                scheduler=scheduler, epoch=epoch,
                best_metric=best_metric, cfg=cfg, scaler=scaler,
            )
            if is_best:
                save_checkpoint(
                    path=str(output_dir / "best.pth"),
                    model=model, optimizer=optimizer,
                    scheduler=scheduler, epoch=epoch,
                    best_metric=best_metric, cfg=cfg, scaler=scaler,
                )
                logger.info(
                    f"  ✓ New best {monitor}={best_metric:.4f}  → best.pth saved"
                )

        # ── Early stopping ────────────────────────────────────────────────────
        if early_stopper is not None:
            if is_main:
                stop_training = early_stopper.step(current_metric)
            if world_size>1:
                stop_tensor = torch.tensor([int(stop_training)], device=device)
                torch.distributed.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())
            
        if stop_training:
            if is_main:
                logger.info(
                    f"Early stopping triggered after {early_stopper.patience} "
                    f"epochs without improvement."
                )
            break

    if is_main:
        logger.info(f"Training complete.  Best {monitor}={best_metric:.4f}")
