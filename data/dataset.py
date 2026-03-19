"""
Medical Image Dataset with flexible split handling.

Supported dataset structures
------------------------------
Case 1 – All splits already exist:
    root/
    ├── train/ ├── val/ └── test/
        └── <class_name>/*.jpg

Case 2 – Only train exists (no val):
    root/
    ├── train/   ← val is carved out here via stratified split
    └── test/

Case 3 – Only an 'all/' folder exists:
    root/
    └── all/     ← train / val / test all come from a single stratified split

Case 4 – Any subset of the above (e.g. only train+test with no val folder).

Class names are inferred from the sub-directory names of the split folder.
Images under a class folder are discovered recursively so nested structures
(e.g. root/train/class/subclass/img.jpg) are also handled correctly.

Transforms
----------
Expects albumentations Compose objects because CLAHE requires numpy arrays.
Images are loaded with OpenCV in BGR format and converted to RGB before
being passed to the transform pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _discover_classes(folder: Path) -> List[str]:
    """Return sorted list of sub-directory names (= class names)."""
    return sorted(d.name for d in folder.iterdir() if d.is_dir())


def _scan_folder(
    folder: Path,
    class_to_idx: Dict[str, int],
) -> List[Tuple[Path, int]]:
    """Recursively collect (image_path, label) pairs from a split folder."""
    samples: List[Tuple[Path, int]] = []
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = folder / cls_name
        if not cls_dir.exists():
            continue
        for img_path in sorted(cls_dir.rglob("*")):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((img_path, cls_idx))
    return samples


def _stratified_split(
    samples: List[Tuple[Path, int]],
    ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Split *samples* so that *ratio* fraction goes to the second partition.
    Stratified by label; falls back to non-stratified if a class has only 1
    member (common with very small medical datasets).
    """
    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    try:
        p_a, p_b, l_a, l_b = train_test_split(
            paths, labels,
            test_size=ratio,
            random_state=seed,
            stratify=labels,
        )
    except ValueError:
        p_a, p_b, l_a, l_b = train_test_split(
            paths, labels,
            test_size=ratio,
            random_state=seed,
        )
    return list(zip(p_a, l_a)), list(zip(p_b, l_b))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MedicalImageDataset(Dataset):
    """
    Dataset for 2-D medical images (e.g. fundus photographs).

    Parameters
    ----------
    root : str | Path
        Root directory of the dataset (contains split sub-directories).
    split : str
        One of ``'train'``, ``'val'``, ``'test'``.
    transform : albumentations.Compose, optional
        Transform pipeline applied to each image (numpy → tensor).
    val_ratio : float
        Fraction of train data used as validation when no ``val/`` folder
        exists.  Ignored if ``val/`` is already present.
    test_ratio : float
        Fraction of ``all/`` data reserved for test when splitting from
        ``all/``.  Ignored when explicit split folders exist.
    seed : int
        Random seed for reproducible splits.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        assert split in ("train", "val", "test"), \
            f"split must be one of 'train', 'val', 'test', got '{split}'"

        self.root      = Path(root)
        self.split     = split
        self.transform = transform

        self.classes:       List[str]            = []
        self.class_to_idx:  Dict[str, int]       = {}
        self.samples:       List[Tuple[Path, int]] = []

        self._build(val_ratio, test_ratio, seed)

    # ------------------------------------------------------------------
    # Internal build logic
    # ------------------------------------------------------------------

    def _build(self, val_ratio: float, test_ratio: float, seed: int) -> None:
        split_dir = self.root / self.split
        all_dir   = self.root / "all"
        train_dir = self.root / "train"

        # ── Case 1: requested split folder exists directly ────────────────
        if split_dir.exists():
            self.classes      = _discover_classes(split_dir)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.samples      = _scan_folder(split_dir, self.class_to_idx)
            return

        # ── Case 2: only 'all/' exists → stratified split ─────────────────
        if all_dir.exists() and not train_dir.exists():
            self.classes      = _discover_classes(all_dir)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            all_samples       = _scan_folder(all_dir, self.class_to_idx)
            self.samples      = self._split_all(all_samples, val_ratio, test_ratio, seed)
            return

        # ── Case 3: train/ exists but val/ is missing ─────────────────────
        if self.split == "val" and train_dir.exists():
            self.classes      = _discover_classes(train_dir)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            train_samples     = _scan_folder(train_dir, self.class_to_idx)
            _, val_samples    = _stratified_split(train_samples, val_ratio, seed)
            self.samples      = val_samples
            return

        # ── Fallback: infer classes from whatever split folder exists ──────
        # (handles mixed cases, e.g. train+test present, requesting train)
        for candidate in ("train", "test", "val", "all"):
            candidate_dir = self.root / candidate
            if candidate_dir.exists():
                self.classes      = _discover_classes(candidate_dir)
                self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
                break

        if not self.classes:
            raise FileNotFoundError(
                f"No recognisable split folder found under '{self.root}'. "
                f"Expected one of: all/, train/, val/, test/"
            )

        # If we still couldn't find the requested split, raise
        raise FileNotFoundError(
            f"Split '{self.split}' not found under '{self.root}'. "
            f"Available: {[d.name for d in self.root.iterdir() if d.is_dir()]}"
        )

    def _split_all(
        self,
        all_samples: List[Tuple[Path, int]],
        val_ratio: float,
        test_ratio: float,
        seed: int,
    ) -> List[Tuple[Path, int]]:
        """Three-way stratified split from a single 'all/' folder."""
        # 1. Carve off test
        trainval, test = _stratified_split(all_samples, test_ratio, seed)
        # 2. Carve off val from the remainder
        adjusted_val = val_ratio / (1.0 - test_ratio)
        train, val   = _stratified_split(trainval, adjusted_val, seed)
        return {"train": train, "val": val, "test": test}[self.split]

    # ------------------------------------------------------------------
    # Class-weight helpers (for loss weighting / balanced sampling)
    # ------------------------------------------------------------------

    def get_class_weights(self) -> np.ndarray:
        """
        Inverse-frequency weights per class (normalised so they sum to
        ``num_classes``).  Pass to the loss function as ``weight=``.
        """
        labels = np.array([s[1] for s in self.samples])
        counts = np.bincount(labels, minlength=len(self.classes)).astype(float)
        weights = 1.0 / np.where(counts == 0, 1.0, counts)
        weights = weights / weights.sum() * len(self.classes)
        return weights

    def get_sample_weights(self) -> np.ndarray:
        """
        Per-sample weights for ``WeightedRandomSampler``.
        """
        class_weights = self.get_class_weights()
        return np.array([class_weights[s[1]] for s in self.samples], dtype=float)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Read as BGR, convert to RGB (uint8 numpy array expected by albumentations)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]  # (C, H, W) float32 tensor after ToTensorV2

        return image, label

    # ------------------------------------------------------------------
    # Convenience repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MedicalImageDataset("
            f"split={self.split}, "
            f"n_samples={len(self)}, "
            f"classes={self.classes})"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: MedicalImageDataset,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    seed: int = 42,
) -> DataLoader:
    """
    Build a DataLoader for the given dataset.

    For DDP training, pass ``distributed=True``: a ``DistributedSampler``
    is used and weighted sampling is disabled (they are mutually exclusive).

    For single-GPU training with class imbalance, set
    ``use_weighted_sampler=True`` to oversample minority classes.
    """
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(dataset.split == "train"),
            seed=seed,
        )
        shuffle = False  # sampler handles shuffling
    elif use_weighted_sampler and dataset.split == "train":
        sample_weights = torch.from_numpy(dataset.get_sample_weights()).float()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = (dataset.split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(dataset.split == "train"),
        persistent_workers=(num_workers > 0),
    )
