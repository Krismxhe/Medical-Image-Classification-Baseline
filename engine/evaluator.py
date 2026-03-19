"""
Evaluator: runs inference on a dataloader and returns metrics.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    amp_enabled: bool = True,
) -> dict:
    """
    Run inference and return a metrics dict.

    Parameters
    ----------
    model       : nn.Module (can be DDP-wrapped)
    dataloader  : val or test DataLoader
    device      : target device
    class_names : list of class name strings (for the report)
    amp_enabled : use torch automatic mixed precision for inference

    Returns
    -------
    dict with acc, auc, f1_macro, f1_weighted, kappa, confusion_matrix,
    report, loss (if the dataloader carries labels).
    """
    model.eval()

    all_probs:  List[np.ndarray] = []
    all_labels: List[int]        = []

    pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)                         # (B, C)

        probs = torch.softmax(logits, dim=-1)              # (B, C)
        all_probs.append(probs.cpu().float().numpy())
        all_labels.extend(labels.cpu().numpy().tolist())

    y_prob = np.concatenate(all_probs, axis=0)             # (N, C)
    y_true = np.array(all_labels, dtype=int)               # (N,)

    metrics = compute_metrics(y_true, y_prob, class_names=class_names)
    return metrics
