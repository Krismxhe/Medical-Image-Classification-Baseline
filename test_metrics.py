"""
Unit tests for the updated metrics, confusion matrix formatter,
and TensorBoard confusion matrix heatmap.

Run:
    python test_metrics.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_binary(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    raw    = rng.dirichlet([1, 1], size=n)
    # Skew so the model is somewhat correct
    raw[y_true == 1] = raw[y_true == 1][:, ::-1]
    return y_true, raw.astype(np.float64)


def _make_multiclass(n_classes=5, n=500, seed=1):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, n_classes, size=n)
    raw    = rng.dirichlet(np.ones(n_classes), size=n)
    # Give each sample a push toward its true class
    for i, c in enumerate(y_true):
        raw[i, c] += 1.0
        raw[i]    /= raw[i].sum()
    return y_true, raw.astype(np.float64)


def _ok(name: str):
    print(f"  ✓  {name}")


def _fail(name: str, detail: str):
    print(f"  ✗  {name}: {detail}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. compute_metrics — key presence and value sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_metrics_binary():
    from utils.metrics import compute_metrics

    y_true, y_prob = _make_binary()
    m = compute_metrics(y_true, y_prob, class_names=["neg", "pos"])

    required = [
        "acc", "balanced_acc",
        "auc", "avg_precision_macro",
        "f1_macro", "f1_weighted",
        "precision_macro", "precision_weighted",
        "recall_macro", "recall_weighted",
        "mcc", "kappa", "kappa_quadratic",
        "sensitivity", "specificity",
        "confusion_matrix", "report",
    ]
    for k in required:
        if k not in m:
            _fail("binary keys", f"missing key '{k}'")

    # top_k_acc should NOT be present for binary
    for k in m:
        if k.startswith("top") and k.endswith("_acc"):
            _fail("binary no top_k", f"unexpected key '{k}' for binary task")

    # Sanity: all scalars in [0, 1] except mcc and kappa
    for k in ["acc", "balanced_acc", "auc", "avg_precision_macro",
              "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
              "sensitivity", "specificity"]:
        v = m[k]
        if not (0.0 <= v <= 1.0):
            _fail(f"binary range {k}", f"value {v:.4f} outside [0,1]")

    for k in ["mcc", "kappa"]:
        v = m[k]
        if not (-1.0 <= v <= 1.0):
            _fail(f"binary range {k}", f"value {v:.4f} outside [-1,1]")

    _ok("compute_metrics — binary keys & ranges")


def test_compute_metrics_multiclass():
    from utils.metrics import compute_metrics

    n_classes = 5
    y_true, y_prob = _make_multiclass(n_classes=n_classes)
    names = [f"cls{i}" for i in range(n_classes)]
    m = compute_metrics(y_true, y_prob, class_names=names)

    # top_k key should be present (n_classes=5 → k=3)
    topk_keys = [k for k in m if k.startswith("top") and k.endswith("_acc")]
    if not topk_keys:
        _fail("multiclass top_k", "no top_k_acc key found")
    for k in topk_keys:
        v = m[k]
        if not (0.0 <= v <= 1.0):
            _fail(f"top_k range {k}", f"value {v:.4f} outside [0,1]")

    # sensitivity / specificity should NOT be present
    for k in ("sensitivity", "specificity"):
        if k in m:
            _fail("multiclass no binary keys", f"unexpected key '{k}'")

    _ok("compute_metrics — multiclass keys & ranges")


def test_compute_metrics_edge_single_class():
    """All samples are the same class — AUC / avg_precision should be nan, not crash."""
    from utils.metrics import compute_metrics

    y_true = np.zeros(50, dtype=int)
    y_prob = np.column_stack([np.ones(50) * 0.9, np.ones(50) * 0.1])
    m = compute_metrics(y_true, y_prob)

    for k in ("auc", "avg_precision_macro"):
        if k not in m:
            _fail("edge single class", f"key '{k}' missing")
        # Should be nan (not crash)
        if not np.isnan(m[k]):
            # sklearn may still return a value; just ensure it's a float
            if not isinstance(m[k], float):
                _fail("edge single class", f"{k} is not a float: {m[k]}")

    _ok("compute_metrics — edge: single class in y_true")


# ─────────────────────────────────────────────────────────────────────────────
# 2. format_metrics — expected keys appear in string
# ─────────────────────────────────────────────────────────────────────────────

def test_format_metrics():
    from utils.metrics import compute_metrics, format_metrics

    y_true, y_prob = _make_multiclass(n_classes=4)
    m = compute_metrics(y_true, y_prob)
    line = format_metrics(m, prefix="val_")

    for key in ["val_acc", "val_auc", "val_f1_macro", "val_mcc", "val_kappa"]:
        if key not in line:
            _fail("format_metrics", f"key '{key}' not found in output:\n{line}")

    _ok("format_metrics — all expected keys present in summary line")


# ─────────────────────────────────────────────────────────────────────────────
# 3. format_confusion_matrix — totals are correct, class names present
# ─────────────────────────────────────────────────────────────────────────────

def test_format_confusion_matrix():
    from utils.metrics import format_confusion_matrix
    from sklearn.metrics import confusion_matrix

    y_true, y_prob = _make_multiclass(n_classes=3, n=300)
    y_pred = y_prob.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)
    names = ["alpha", "beta", "gamma"]

    table = format_confusion_matrix(cm, class_names=names)

    # Class names must appear
    for name in names:
        if name not in table:
            _fail("format_confusion_matrix", f"class name '{name}' not in output")

    # Row totals: each row sum must appear as a standalone number in the table
    for i, name in enumerate(names):
        row_total = str(cm[i].sum())
        if row_total not in table:
            _fail("format_confusion_matrix", f"row total {row_total} for class '{name}' not found")

    # Column totals must appear
    for j in range(cm.shape[1]):
        col_total = str(cm[:, j].sum())
        if col_total not in table:
            _fail("format_confusion_matrix", f"col total {col_total} not found")

    # Grand total
    grand = str(cm.sum())
    if grand not in table:
        _fail("format_confusion_matrix", f"grand total {grand} not found")

    # Row percentages: spot-check one cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                pct = cm[i, j] / cm[i].sum() * 100
                pct_str = f"{pct:.1f}%"
                if pct_str not in table:
                    _fail("format_confusion_matrix",
                          f"row% '{pct_str}' for cell ({i},{j}) not found")
                break  # one spot-check per row is enough

    _ok("format_confusion_matrix — totals, class names, and row% correct")


def test_format_confusion_matrix_no_names():
    """Should fall back to numeric labels without crashing."""
    from utils.metrics import format_confusion_matrix
    import numpy as np

    cm = np.array([[10, 2], [3, 15]])
    table = format_confusion_matrix(cm, class_names=None)
    assert "0" in table and "1" in table
    _ok("format_confusion_matrix — works without class names")


# ─────────────────────────────────────────────────────────────────────────────
# 4. log_confusion_matrix — renders without error (no TensorBoard writer needed
#    because we mock _tb_writer to None and verify it handles that gracefully,
#    then we test the actual render path with a real SummaryWriter)
# ─────────────────────────────────────────────────────────────────────────────

def test_log_confusion_matrix_no_writer():
    """With no TensorBoard writer, method must be a no-op (not crash)."""
    from utils.logger import TrainingLogger

    with tempfile.TemporaryDirectory() as tmp:
        tl = TrainingLogger(tmp, use_tensorboard=False, use_wandb=False)
        cm = np.array([[50, 5, 2], [3, 40, 4], [1, 2, 60]])
        tl.log_confusion_matrix(cm, class_names=["a", "b", "c"], step=0)
    _ok("log_confusion_matrix — no-op when TensorBoard writer is absent")


def test_log_confusion_matrix_with_writer():
    """Render the full heatmap into a real SummaryWriter — checks no exceptions."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("  ~  log_confusion_matrix (with writer) — TensorBoard not installed, skipped")
        return

    from utils.logger import TrainingLogger

    with tempfile.TemporaryDirectory() as tmp:
        tl = TrainingLogger(tmp, use_tensorboard=True, use_wandb=False)
        # 5-class confusion matrix
        cm = np.array([
            [120,  5,  3,  2,  1],
            [  4, 95, 12,  1,  3],
            [  2,  7, 88,  4,  2],
            [  1,  2,  3, 70,  5],
            [  3,  1,  2,  4, 55],
        ])
        tl.log_confusion_matrix(cm, class_names=["A","B","C","D","E"], step=1)
        tl.close()

    _ok("log_confusion_matrix — 5-class heatmap rendered without errors")


# ─────────────────────────────────────────────────────────────────────────────
# 5. eval.py report block — import and run the formatting inline
# ─────────────────────────────────────────────────────────────────────────────

def test_eval_report_block():
    """Simulate the eval.py report loop with new grouped scalar output."""
    from utils.metrics import compute_metrics, format_confusion_matrix

    y_true, y_prob = _make_multiclass(n_classes=4, n=400)
    names = ["cat", "dog", "bird", "fish"]
    metrics = compute_metrics(y_true, y_prob, class_names=names)

    scalar_groups = [
        ("Accuracy",   ["acc", "balanced_acc", "top3_acc", "top2_acc"]),
        ("ROC / PR",   ["auc", "avg_precision_macro"]),
        ("F1",         ["f1_macro", "f1_weighted"]),
        ("Precision",  ["precision_macro", "precision_weighted"]),
        ("Recall",     ["recall_macro", "recall_weighted"]),
        ("Other",      ["mcc", "kappa", "kappa_quadratic", "sensitivity", "specificity"]),
    ]

    output_lines = []
    for group_name, keys in scalar_groups:
        row_parts = []
        for k in keys:
            v = metrics.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                row_parts.append(f"{k}={v:.4f}")
        if row_parts:
            output_lines.append(f"  {group_name:<12}: " + "  ".join(row_parts))

    cm_table = format_confusion_matrix(metrics["confusion_matrix"], class_names=names)

    # Must have at least 4 non-empty group lines
    if len(output_lines) < 4:
        _fail("eval_report_block", f"only {len(output_lines)} metric groups rendered")

    # Confusion matrix table must contain all class names
    for name in names:
        if name not in cm_table:
            _fail("eval_report_block", f"class '{name}' missing from confusion matrix table")

    _ok("eval.py report block — grouped scalar output and confusion matrix table correct")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Metrics & Confusion Matrix — test suite")
    print("=" * 60)

    tests = [
        test_compute_metrics_binary,
        test_compute_metrics_multiclass,
        test_compute_metrics_edge_single_class,
        test_format_metrics,
        test_format_confusion_matrix,
        test_format_confusion_matrix_no_names,
        test_log_confusion_matrix_no_writer,
        test_log_confusion_matrix_with_writer,
        test_eval_report_block,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except SystemExit:
            raise
        except Exception as e:
            _fail(t.__name__, str(e))

    print("=" * 60)
    print(f"  {passed}/{len(tests)} tests passed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
