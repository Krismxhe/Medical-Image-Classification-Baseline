"""
Transforms for medical image classification.

Uses albumentations for augmentation (supports CLAHE natively).
All transforms receive/return numpy arrays (H, W, C) in uint8 RGB format.
ToTensorV2 converts to (C, H, W) float32 tensor as the final step.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(split: str, cfg: dict) -> A.Compose:
    """
    Build albumentations transform pipeline for a given split.

    Args:
        split: One of 'train', 'val', 'test'.
        cfg:   data section of the config dict. Keys used:
                 img_size, mean, std, use_clahe.

    Returns:
        An albumentations Compose object.
    """
    img_size  = cfg.get("img_size", 224)
    mean      = cfg.get("mean", [0.485, 0.456, 0.406])
    std       = cfg.get("std",  [0.229, 0.224, 0.225])
    use_clahe = cfg.get("use_clahe", True)

    clahe = [A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)] if use_clahe else []

    normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0)

    if split == "train":
        return A.Compose([
            # ── Spatial ───────────────────────────────────────────────────────
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=30,
                border_mode=0, p=0.5
            ),
            # ── Fundus-specific: geometric distortions ────────────────────────
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                A.OpticalDistortion(distort_limit=0.2, p=1.0),
            ], p=0.3),
            # ── Fundus-specific: contrast enhancement (CLAHE) ─────────────────
            *clahe,
            # ── Color / photometric ───────────────────────────────────────────
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # ── Noise / blur ──────────────────────────────────────────────────
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.2),
            A.GaussNoise(p=0.2),
            # ── Regularisation ────────────────────────────────────────────────
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.2,
            ),
            # ── Normalize + to tensor ─────────────────────────────────────────
            normalize,
            ToTensorV2(),
        ])
    else:
        # val / test: deterministic pipeline
        return A.Compose([
            A.Resize(img_size, img_size),
            *clahe,
            normalize,
            ToTensorV2(),
        ])
