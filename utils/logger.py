"""
Unified logging for training runs.

Supports:
  • TensorBoard  (always enabled when use_tensorboard=True)
  • Weights & Biases  (optional, use_wandb=True)
  • CSV file  (always written to output_dir/metrics.csv)
  • Console stdout
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TrainingLogger:
    """
    Wraps TensorBoard, W&B, and CSV logging behind a single interface.

    Parameters
    ----------
    output_dir : str | Path
        Directory where logs / artefacts are stored.
    exp_name : str
        Experiment name used for W&B runs and TensorBoard sub-dir.
    use_tensorboard : bool
    use_wandb : bool
    wandb_project : str
    cfg : dict, optional
        Full config dict logged to W&B as hyperparameters.
    is_main : bool
        Only the main process (rank 0) should write logs.
    """

    def __init__(
        self,
        output_dir: str | Path,
        exp_name: str = "exp",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "medical-cls",
        cfg: Optional[dict] = None,
        is_main: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.exp_name   = exp_name
        self.is_main    = is_main
        self._tb_writer = None
        self._wandb_run = None
        self._csv_path  = self.output_dir / "metrics.csv"
        self._csv_file  = None
        self._csv_writer = None
        self._csv_header_written = False

        if not is_main:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Console handler ───────────────────────────────────────────────────
        _setup_console_logging(self.output_dir / "train.log")

        # ── TensorBoard ───────────────────────────────────────────────────────
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.output_dir / "tensorboard"
                self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
                logging.info(f"TensorBoard logs → {tb_dir}")
            except ImportError:
                logging.warning("TensorBoard not installed; skipping.")

        # ── Weights & Biases ──────────────────────────────────────────────────
        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    name=exp_name,
                    config=cfg,
                    dir=str(self.output_dir),
                )
                logging.info(f"W&B run: {self._wandb_run.url}")
            except ImportError:
                logging.warning("wandb not installed; skipping.")

        # ── CSV ───────────────────────────────────────────────────────────────
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)

    # ------------------------------------------------------------------

    def log_scalars(self, metrics: Dict[str, Any], step: int, prefix: str = "") -> None:
        """Log a dict of scalar metrics at the given global step."""
        if not self.is_main:
            return

        tagged = {f"{prefix}/{k}" if prefix else k: v
                  for k, v in metrics.items()
                  if isinstance(v, (int, float)) and not np.isnan(float(v))}

        # TensorBoard
        if self._tb_writer is not None:
            for name, val in tagged.items():
                self._tb_writer.add_scalar(name, val, global_step=step)

        # W&B
        if self._wandb_run is not None:
            import wandb
            self._wandb_run.log(tagged, step=step)

        # CSV
        if self._csv_writer is not None:
            row = {"step": step, **tagged}
            if not self._csv_header_written:
                self._csv_writer.writerow(["step"] + list(tagged.keys()))
                self._csv_header_written = True
            self._csv_writer.writerow([step] + list(tagged.values()))
            self._csv_file.flush()

    def log_image(self, tag: str, img_tensor, step: int) -> None:
        """Log a single image tensor (C, H, W) to TensorBoard."""
        if not self.is_main or self._tb_writer is None:
            return
        self._tb_writer.add_image(tag, img_tensor, global_step=step)

    def log_confusion_matrix(self, cm: np.ndarray, class_names, step: int) -> None:
        """Render confusion matrix as a matplotlib figure and log to TensorBoard.

        Each main cell shows the exact sample count and its row-percentage
        (i.e. share of that true class predicted as each label).
        An extra "Total" row and column show margin sums so that per-class
        sample counts are always visible without mental arithmetic.
        """
        if not self.is_main or self._tb_writer is None:
            return
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import seaborn as sns

            n = cm.shape[0]
            labels = list(class_names) if class_names else [str(i) for i in range(n)]

            row_sums = cm.sum(axis=1, keepdims=True)   # (n, 1)
            col_sums = cm.sum(axis=0, keepdims=True)   # (1, n)
            total    = cm.sum()

            # Extended (n+1) x (n+1) matrix for heatmap colour scaling
            cm_ext = np.zeros((n + 1, n + 1), dtype=float)
            cm_ext[:n, :n] = cm
            cm_ext[:n,  n] = row_sums[:, 0]
            cm_ext[n,  :n] = col_sums[0, :]
            cm_ext[n,   n] = total

            # Build annotation strings
            annot = np.empty((n + 1, n + 1), dtype=object)
            for i in range(n):
                for j in range(n):
                    pct = cm[i, j] / (row_sums[i, 0] + 1e-8) * 100
                    annot[i, j] = f"{cm[i, j]}\n({pct:.1f}%)"
                annot[i, n] = str(int(row_sums[i, 0]))   # row total
            for j in range(n):
                annot[n, j] = str(int(col_sums[0, j]))   # col total
            annot[n, n] = str(int(total))

            tick_labels = labels + ["Total"]
            fig_size = max(7, n + 2)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))

            # Draw heatmap; mask the margin row/col so they use a separate colour
            mask_main   = np.zeros((n + 1, n + 1), dtype=bool)
            mask_margin = np.ones((n + 1, n + 1),  dtype=bool)
            mask_main[n,  :] = True   # hide last row from main heatmap
            mask_main[:,  n] = True   # hide last col from main heatmap
            mask_margin[:n, :n] = True  # hide main block from margin heatmap

            sns.heatmap(
                cm_ext, mask=mask_main, annot=annot, fmt="", cmap="Blues",
                xticklabels=tick_labels, yticklabels=tick_labels,
                ax=ax, cbar=False,
                linewidths=0.5, linecolor="white",
            )
            sns.heatmap(
                cm_ext, mask=mask_margin, annot=annot, fmt="", cmap="Oranges",
                xticklabels=tick_labels, yticklabels=tick_labels,
                ax=ax, cbar=False,
                linewidths=0.5, linecolor="white",
            )

            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("True",      fontsize=11)
            ax.set_title("Confusion Matrix  (count · row%)", fontsize=12)
            ax.tick_params(axis="x", rotation=45)
            ax.tick_params(axis="y", rotation=0)

            fig.tight_layout()
            self._tb_writer.add_figure("confusion_matrix", fig, global_step=step)
            plt.close(fig)
        except Exception as e:
            logging.warning(f"Could not log confusion matrix: {e}")

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
        if self._csv_file is not None:
            self._csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Module-level console + file logging
# ---------------------------------------------------------------------------

def _setup_console_logging(log_file: Optional[Path] = None) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        root.addHandler(fh)
