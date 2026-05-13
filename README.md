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
| Metrics | Accuracy, Balanced Accuracy, AUC (OvR macro), PR-AUC, F1/Precision/Recall (macro+weighted), MCC, Cohen's Kappa, Top-k Accuracy, Sensitivity/Specificity |
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

## Metrics & Confusion Matrix

### Computed metrics

| Key | Full name | Notes |
|---|---|---|
| `acc` | Accuracy | Fraction of correctly classified samples |
| `balanced_acc` | Balanced accuracy | Mean per-class recall; robust to class imbalance |
| `auc` | ROC-AUC (macro OvR) | Area under ROC curve; binary uses positive-class score |
| `avg_precision_macro` | PR-AUC (macro) | Area under Precision-Recall curve; preferred over AUC on imbalanced data |
| `f1_macro` | F1 macro | Unweighted mean F1 across classes |
| `f1_weighted` | F1 weighted | F1 weighted by class support |
| `precision_macro` | Precision macro | Unweighted mean precision |
| `precision_weighted` | Precision weighted | Precision weighted by class support |
| `recall_macro` | Recall macro | Unweighted mean recall (= balanced_acc for uniform weighting) |
| `recall_weighted` | Recall weighted | Recall weighted by class support |
| `mcc` | Matthews Correlation Coefficient | Single value in [−1, +1]; handles imbalanced multi-class well |
| `kappa` | Cohen's Kappa | Agreement above chance (linear weighting) |
| `kappa_quadratic` | Cohen's Kappa (quadratic) | Penalises large disagreements; common in ordinal grading tasks |
| `top3_acc` | Top-3 accuracy | Whether true label is in model's top-3 predictions (≥3 classes only) |
| `sensitivity` | Sensitivity / Recall | TP / (TP + FN); binary classification only |
| `specificity` | Specificity | TN / (TN + FP); binary classification only |

All scalar metrics are logged every epoch to **TensorBoard**, **W&B**, and **`metrics.csv`**.  
They are also visible in the training console as a one-line summary per epoch.

### Confusion matrix

The confusion matrix appears in **two places**:

#### 1. TensorBoard heatmap (during training and eval)

Each cell in the main block shows:
```
N
(xx.x%)
```
where **N** is the exact sample count and **xx.x%** is the row-normalised percentage (share of that true class predicted as each label).

A shaded **"Total" column** (right) shows the total number of true samples per class.  
A shaded **"Total" row** (bottom) shows the total number of samples predicted as each class.  
The bottom-right cell shows the **grand total** sample count.

Navigate to the confusion matrix in TensorBoard under the **Images** tab → `confusion_matrix`.

#### 2. Console / log file (eval.py only)

`eval.py` prints a formatted text table directly to `stdout` and `train.log`:

```
Confusion Matrix  (rows = True class, cols = Predicted class)

               class_A    class_B    class_C      Total
--------------------------------------------------------
       class_A  120(93.8%)   5(3.9%)   3(2.3%)      128
       class_B    8(7.0%)  95(82.6%)  12(10.4%)     115
       class_C    2(2.1%)   7(7.2%)  88(90.7%)       97
--------------------------------------------------------
         Total      130       107       103          340
```

Each cell shows **count(row%)** so you can read both the absolute number of samples and the classification rate at a glance.

#### Saving full results to JSON

```bash
python eval.py --config configs/my_exp.yaml \
               --checkpoint outputs/my_run/best.pth \
               --split test \
               --output_json results.json
```

The JSON file contains all scalar metrics plus the confusion matrix as a nested list, suitable for downstream analysis or CI comparisons.

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
