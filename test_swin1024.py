"""
Quick smoke test: load Swin-Tiny 1024 and run a random-tensor forward pass.

Usage
-----
    python test_swin1024.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from models.model_factory import build_model
from utils.misc import load_config, merge_configs


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] device = {device}")

    # ── Load config ────────────────────────────────────────────────────────────
    base_cfg = load_config("configs/base.yaml")
    exp_cfg  = load_config("configs/examples/swin_tiny_1024.yaml")
    cfg      = merge_configs(base_cfg, exp_cfg)

    model_cfg = cfg["model"]
    img_size  = cfg["data"]["img_size"]   # 1024
    num_classes = 5                        # arbitrary for smoke test

    print(f"[test] model   = {model_cfg['name']}")
    print(f"[test] img_size passed to timm = {model_cfg.get('img_size')}")
    print(f"[test] pretrained = {model_cfg.get('pretrained')}")

    # ── Build model ────────────────────────────────────────────────────────────
    print("[test] Building model (may download pretrained weights) ...")
    model = build_model(model_cfg, num_classes=num_classes).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[test] Total parameters: {total_params / 1e6:.1f} M")

    # ── Random tensor forward pass ─────────────────────────────────────────────
    B = 1
    x = torch.randn(B, 3, img_size, img_size, device=device)
    print(f"[test] Input  shape : {tuple(x.shape)}")

    with torch.no_grad():
        logits = model(x)

    print(f"[test] Output shape : {tuple(logits.shape)}  (expected: ({B}, {num_classes}))")

    assert logits.shape == (B, num_classes), (
        f"Shape mismatch: got {tuple(logits.shape)}, expected ({B}, {num_classes})"
    )
    print("[test] ✓  Forward pass succeeded — Swin-Tiny 1024 is working correctly.")


if __name__ == "__main__":
    main()
