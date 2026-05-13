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
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
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
        acc, balanced_acc,
        auc, avg_precision_macro,
        f1_macro, f1_weighted,
        precision_macro, precision_weighted,
        recall_macro, recall_weighted,
        mcc, kappa, kappa_quadratic,
        top_k_acc (multi-class only, k=min(3, C-1)),
        sensitivity, specificity (binary only),
        confusion_matrix (np.ndarray),
        report (str)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2

    y_pred = y_prob.argmax(axis=-1) if y_prob.ndim == 2 else (y_prob >= 0.5).astype(int)

    results: dict = {}

    # ── Accuracy ──────────────────────────────────────────────────────────────
    results["acc"] = float(accuracy_score(y_true, y_pred))

    # ── Balanced accuracy (mean per-class recall) ─────────────────────────────
    results["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))

    # ── AUC (ROC) ─────────────────────────────────────────────────────────────
    try:
        if n_classes == 2:
            probs_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            results["auc"] = float(roc_auc_score(y_true, probs_pos))
        else:
            results["auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    except ValueError:
        results["auc"] = float("nan")

    # ── Average Precision (PR-AUC) ────────────────────────────────────────────
    try:
        if n_classes == 2:
            probs_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            results["avg_precision_macro"] = float(
                average_precision_score(y_true, probs_pos)
            )
        else:
            results["avg_precision_macro"] = float(
                average_precision_score(y_true, y_prob, average="macro")
            )
    except ValueError:
        results["avg_precision_macro"] = float("nan")

    # ── F1 ────────────────────────────────────────────────────────────────────
    results["f1_macro"]    = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    results["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # ── Precision ─────────────────────────────────────────────────────────────
    results["precision_macro"]    = float(precision_score(y_true, y_pred, average="macro",    zero_division=0))
    results["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))

    # ── Recall ────────────────────────────────────────────────────────────────
    results["recall_macro"]    = float(recall_score(y_true, y_pred, average="macro",    zero_division=0))
    results["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    # ── Matthews Correlation Coefficient ─────────────────────────────────────
    results["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    # ── Cohen's Kappa ─────────────────────────────────────────────────────────
    results["kappa"] = float(cohen_kappa_score(y_true, y_pred))
    try:
        results["kappa_quadratic"] = float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
        )
    except Exception:
        results["kappa_quadratic"] = float("nan")

    # ── Top-k accuracy (multi-class only) ────────────────────────────────────
    if n_classes >= 3 and y_prob.ndim == 2:
        k = min(3, n_classes - 1)
        try:
            results[f"top{k}_acc"] = float(
                top_k_accuracy_score(y_true, y_prob, k=k)
            )
        except Exception:
            results[f"top{k}_acc"] = float("nan")

    # ── Sensitivity / Specificity (binary only) ───────────────────────────────
    if n_classes == 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        results["sensitivity"] = float(tp / (tp + fn + 1e-8))
        results["specificity"] = float(tn / (tn + fp + 1e-8))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # ── Per-class report ──────────────────────────────────────────────────────
    results["report"] = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    return results


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Return a one-line summary string of the main scalar metrics."""
    keys = [
        "acc", "balanced_acc",
        "auc", "avg_precision_macro",
        "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro",
        "mcc", "kappa",
    ]
    parts = []
    for k in keys:
        if k in metrics and not (isinstance(metrics[k], float) and np.isnan(metrics[k])):
            parts.append(f"{prefix}{k}={metrics[k]:.4f}")
    # append top-k if present
    for k in metrics:
        if k.startswith("top") and k.endswith("_acc"):
            if not (isinstance(metrics[k], float) and np.isnan(metrics[k])):
                parts.append(f"{prefix}{k}={metrics[k]:.4f}")
    return "  ".join(parts)


def format_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None) -> str:
    """
    Return a formatted text table of the confusion matrix with class-name
    headers and row / column margin totals.

    Rows = True class, Columns = Predicted class.
    """
    n = cm.shape[0]
    labels = class_names if class_names and len(class_names) == n \
        else [str(i) for i in range(n)]

    col_w = max(max(len(lb) for lb in labels), 8)
    row_label_w = max(len(lb) for lb in labels)

    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    total    = cm.sum()

    sep = "-" * (row_label_w + 2 + (col_w + 2) * (n + 1))

    lines = []
    lines.append("")
    lines.append("Confusion Matrix  (rows = True class, cols = Predicted class)")
    lines.append("")

    # Header row
    header = f"{'':>{row_label_w}}  " + "  ".join(f"{lb:>{col_w}}" for lb in labels) + f"  {'Total':>{col_w}}"
    lines.append(header)
    lines.append(sep)

    for i, lb in enumerate(labels):
        row_str = f"{lb:>{row_label_w}}  "
        for j in range(n):
            pct = cm[i, j] / (row_sums[i] + 1e-8) * 100
            cell = f"{cm[i,j]}({pct:.1f}%)"
            row_str += f"{cell:>{col_w}}  "
        row_str += f"{row_sums[i]:>{col_w}}"
        lines.append(row_str)

    lines.append(sep)

    # Column totals row
    total_row = f"{'Total':>{row_label_w}}  " + "  ".join(f"{col_sums[j]:>{col_w}}" for j in range(n)) + f"  {total:>{col_w}}"
    lines.append(total_row)
    lines.append("")

    return "\n".join(lines)
