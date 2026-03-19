"""
Medical image classification metrics.

All functions accept:
  y_true : array-like of int, shape (N,)
  y_prob : array-like of float, shape (N, C)  — softmax probabilities
           (or (N,) for binary tasks)

Returned value is always a plain Python dict of {metric_name: float}.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Compute a standard suite of classification metrics.

    Parameters
    ----------
    y_true : (N,) int array of ground-truth class indices.
    y_prob : (N, C) float array of predicted probabilities.
    class_names : list of str, optional.  Used only in the report string.

    Returns
    -------
    dict with keys:
        acc, auc, f1_macro, f1_weighted, kappa,
        sensitivity (binary only), specificity (binary only),
        confusion_matrix (np.ndarray),
        report (str)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2

    y_pred = y_prob.argmax(axis=-1) if y_prob.ndim == 2 else (y_prob >= 0.5).astype(int)

    results: dict = {}

    # ── Accuracy ─────────────────────────────────────────────────────────────
    results["acc"] = float(accuracy_score(y_true, y_pred))

    # ── AUC ──────────────────────────────────────────────────────────────────
    try:
        if n_classes == 2:
            probs_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            results["auc"] = float(roc_auc_score(y_true, probs_pos))
        else:
            results["auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    except ValueError:
        # Happens if only one class is present in y_true (small val set)
        results["auc"] = float("nan")

    # ── F1 ───────────────────────────────────────────────────────────────────
    results["f1_macro"]    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    results["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # ── Cohen's Kappa (quadratic weighting common in grading tasks) ───────────
    results["kappa"] = float(cohen_kappa_score(y_true, y_pred))
    try:
        results["kappa_quadratic"] = float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
        )
    except Exception:
        results["kappa_quadratic"] = float("nan")

    # ── Sensitivity / Specificity (binary only) ───────────────────────────────
    if n_classes == 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        results["sensitivity"] = float(tp / (tp + fn + 1e-8))
        results["specificity"] = float(tn / (tn + fp + 1e-8))

    # ── Confusion matrix ─────────────────────────────────────────────────────
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # ── Per-class report ─────────────────────────────────────────────────────
    results["report"] = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    return results


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Return a one-line summary string of the main scalar metrics."""
    keys = ["acc", "auc", "f1_macro", "f1_weighted", "kappa"]
    parts = []
    for k in keys:
        if k in metrics and not (isinstance(metrics[k], float) and np.isnan(metrics[k])):
            parts.append(f"{prefix}{k}={metrics[k]:.4f}")
    return "  ".join(parts)
