# Medical Image Classification Baseline Evaluation

A clean, extensible baseline for 2-D medical image classification (fundus, dermoscopy, histology, …).

Powered by **timm** (700+ pretrained models), **albumentations** (CLAHE + fundus-specific augmentation), and native PyTorch **DDP** for multi-GPU training.

---

## Features

| Feature | Detail |
|---|---|
| Model zoo | Any timm model: ResNet, EfficientNet(V2), ConvNeXt, ViT, DeiT, Swin, MaxViT, RegNet, NASNet, … |
| Augmentation | CLAHE, GridDistortion, ElasticTransform, ColorJitter, GaussNoise, CoarseDropout |
| Loss | CrossEntropy, Label-Smoothing CE, Focal Loss; all with optional class weights |
| Scheduler | Cosine (+ linear warmup), StepLR, ReduceLROnPlateau |
| Mixed precision | `torch.cuda.amp` |
| Multi-GPU | `torchrun` DDP, `DistributedSampler` |
| Class imbalance | `WeightedRandomSampler` (single-GPU) + inverse-frequency class weights |
| Metrics | Accuracy, AUC (OvR macro), F1 macro/weighted, Cohen's Kappa (quadratic), Sensitivity/Specificity |
| Logging | TensorBoard + W&B (optional) + CSV |
| Dataset | Auto-detects split structure; handles missing val/test via stratified split |

---

## Dataset Structure

The dataset root must follow one of these layouts:

```
# Layout A — all splits present
dataset_root/
├── train/
│   ├── class_A/  *.jpg / *.png / ...
│   └── class_B/
├── val/
│   ├── class_A/
│   └── class_B/
└── test/
    ├── class_A/
    └── class_B/

# Layout B — no val/ folder (carved from train/ automatically)
dataset_root/
├── train/
└── test/

# Layout C — single pool (train/val/test all split automatically)
dataset_root/
└── all/
    ├── class_A/
    └── class_B/
```

Class names are inferred from sub-directory names. Images are discovered **recursively**, so nested structures work too.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Edit the config

Copy an example config and set `data.root`:

```bash
cp configs/examples/resnet50.yaml configs/my_exp.yaml
# Edit configs/my_exp.yaml → set data.root
```

### 3. Single-GPU training

```bash
python train.py --config configs/my_exp.yaml
```

### 4. Multi-GPU training (4 GPUs)

```bash
torchrun --nproc_per_node 4 train.py --config configs/my_exp.yaml
```

### 5. Override config on the fly

```bash
python train.py --config configs/my_exp.yaml \
    --data.root /data/fundus \
    --train.batch_size 64 \
    --model.name efficientnet_b3 \
    --logging.exp_name my_run
```

### 6. Evaluate

```bash
python eval.py --config configs/my_exp.yaml \
               --checkpoint outputs/my_run/best.pth \
               --split test \
               --output_json results.json
```

### 7. Resume training

```bash
python train.py --config configs/my_exp.yaml \
                --resume outputs/my_run/last.pth
```

---

## Pre-configured Models

| Config | Architecture | img_size | Notes |
|---|---|---|---|
| `resnet50.yaml` | ResNet-50 | 224 | Reliable baseline |
| `efficientnet_b3.yaml` | EfficientNet-B3 (NAS) | 300 | Focal loss, good for imbalance |
| `swin_tiny.yaml` | Swin-T | 224 | Hierarchical ViT |
| `vit_base.yaml` | ViT-B/16 | 224 | Vanilla transformer |
| `convnext_small.yaml` | ConvNeXt-S | 224 | Modern CNN |

Browse all available pretrained models:

```python
import timm
print(timm.list_models("*", pretrained=True))
```

---

## Output Directory

```
outputs/<exp_name>/
├── best.pth          ← best checkpoint (monitored metric)
├── last.pth          ← most recent checkpoint
├── train.log         ← full console log
├── metrics.csv       ← per-epoch scalars
└── tensorboard/      ← TensorBoard event files
```

Launch TensorBoard:

```bash
tensorboard --logdir outputs/
```

---

## Config Reference

All options live in `configs/base.yaml`.  Model-specific YAMLs override only what they need; everything else inherits from base.

Key sections: `data`, `model`, `train`, `optimizer`, `scheduler`, `loss`, `early_stopping`, `logging`.

---

## Project Structure

```
cls-baseline/
├── configs/
│   ├── base.yaml
│   └── examples/
├── data/
│   ├── dataset.py       ← MedicalImageDataset + build_dataloader
│   └── transforms.py    ← albumentations pipelines (CLAHE included)
├── models/
│   └── model_factory.py ← timm wrapper + checkpoint save/load
├── engine/
│   ├── trainer.py       ← full training loop (DDP, AMP, early stopping)
│   └── evaluator.py     ← inference + metrics collection
├── utils/
│   ├── metrics.py       ← AUC, F1, Kappa, confusion matrix
│   ├── logger.py        ← TensorBoard / W&B / CSV
│   └── misc.py          ← seed, config, AverageMeter, EarlyStopping
├── train.py
├── eval.py
└── requirements.txt
```
