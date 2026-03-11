# EfficientPIE Replication — Session Notes (2026-03-10)

## Goal

Replicate the **EfficientPIE** model from the IJCAI-25 paper
*"EfficientPIE: Real-Time Prediction on Pedestrian Crossing Intention with Sole Observation"*
targeting these metrics from Table 3:

| Dataset | Accuracy | AUC  | F1   | Precision |
|---------|----------|------|------|-----------|
| PIE     | 0.92     | 0.92 | 0.95 | **0.96**  |
| JAAD    | 0.89     | 0.86 | 0.62 | 0.63      |

---

## Paper Summary

EfficientPIE predicts pedestrian crossing intention from a **single image** (300×300 crop
around the pedestrian) rather than a sequence of frames. Key contributions:

- **Architecture**: EfficientNet-inspired backbone (no RNN). Two Common-Conv blocks, two
  Fused-MBConv blocks, two MBConv blocks (with squeeze-excitation), then AvgPool + Linear
  classifier. Inputs one 300×300 RGB crop, outputs 2-class crossing probability.

- **Progressive Perturbation**: At each training epoch `j` out of `E` total, adds random
  noise `δ = [p_j, -p_j]` where `p_j ~ Uniform(-mj/E, mj/E)` to the logits before
  computing the loss. Perturbation grows with epoch, forcing robustness to uncertain labels.

- **Intention Domain Incremental Learning (IDIL)**: The 15-frame observation window (time
  steps 0–14) is treated as 15 "intention domains". The model is first trained on step 0
  (earliest frame), then incrementally fine-tuned on steps 2, 4, 6, 8, 10, 12, 14 (every
  other step). At each IL step, a frozen copy of the previous model provides distillation
  supervision via an adaptive loss:
  - If `L_new > L_old`: use `L_new` only (new data is harder, trust it)
  - Else: use `0.5 * L_old + L_new` (blend for stability)

- **Training setup**: RMSProp, lr=1e-5, weight_decay=1e-4, cosine annealing, 50 epochs
  base training + 30 epochs per IL step, batch size 32. ImageNet pre-trained weights used
  as starting point.

---

## Repository State (Before Session)

The repo at `/data/repos/EfficientPIE` contained most code but had several blockers:

| Problem | Impact |
|---------|--------|
| `models/EfficientPIE_backup.py` imports `AdaptiveEncoder` (doesn't exist in `common.py`) | Import error on any script |
| `utils/my_dataset.py` imports `adaptive_selection` (module doesn't exist, never used) | Import error |
| `utils/my_dataset.py` hardcodes `reverse_step = 1` | Wrong frame selected for IL steps |
| All scripts import from `models.EfficientPIE_backup` | File had broken import |
| Hardcoded paths to `/home/yphe/...` throughout | Nothing runs on this machine |
| `test_EfficientPIE.py` hardcodes weight path in code | Not configurable |
| `robust_noisy()` hardcodes `total_epochs=30` | Incorrect perturbation scale at 50 epochs |
| `incremental_learning_train()` doesn't freeze `prev_model` | Gradients flow into frozen model |
| No `models/__init__.py` or `utils/__init__.py` | Import resolution issues |
| No `requirements.txt` | Dependencies undocumented |
| No JAAD IL script | JAAD incremental learning not runnable |
| No automation pipeline scripts | Manual step-by-step execution only |
| Dataset annotations not linked | PIE/JAAD APIs couldn't find annotation files |
| `PIE_clips/` directory missing | PIE API expects clips in `PIE_clips/setXX/` layout |
| Frames not extracted from videos | Training requires individual PNG frames |

---

## What Was Done

### 1. Read & Understood the Full Codebase

Read all 10 source files to understand the existing logic before making changes:
- `models/EfficientPIE_backup.py` — model definition
- `models/common.py` — DropPath, SqueezeExcite, ConvBNAct, FusedMBConv, MBConv
- `utils/my_dataset.py` — PyTorch Dataset wrapper
- `utils/train_val.py` — training, evaluation, IL loss functions
- `train_EfficientPIE.py` — PIE base training script
- `test_EfficientPIE.py` — PIE evaluation script
- `train_EfficientPIE_JAAD.py` — JAAD base training script
- `test_EfficientPIE_JAAD.py` — JAAD evaluation script
- `pie_domain_incremental_learning.py` — PIE IL training script
- `pretrain_imagenet.py` — ImageNet pre-training script

Read the full IJCAI-25 paper to understand the methodology and training hyperparameters.

---

### 2. Dataset Preparation

#### 2a. Annotation Symlinks

Annotations live in `/data/repos/PedestrianIntent++/`. Symlinked into dataset directories:

```bash
# PIE
ln -sfn /data/repos/PedestrianIntent++/PIE/PIE/annotations/annotations      /data/datasets/PIE/annotations
ln -sfn /data/repos/PedestrianIntent++/PIE/PIE/annotations/annotations_attributes  /data/datasets/PIE/annotations_attributes
ln -sfn /data/repos/PedestrianIntent++/PIE/PIE/annotations/annotations_vehicle     /data/datasets/PIE/annotations_vehicle

# JAAD
ln -sfn /data/repos/PedestrianIntent++/JAAD/annotations  /data/datasets/JAAD/annotations
```

#### 2b. PIE_clips Layout

The PIE API expects video files at `PIE_clips/setXX/video_XXXX.mp4`. The actual layout is
`setXX/video_XXXX.mp4` directly under `/data/datasets/PIE/`.

**First attempt**: `ln -s /data/datasets/PIE /data/datasets/PIE/PIE_clips` (self-symlink) —
this created an infinite loop when the API listed the directory (found `PIE_clips` inside
`PIE_clips` recursively). Fixed by removing it.

**Fix**: Created a real directory with individual set symlinks:
```bash
mkdir /data/datasets/PIE/PIE_clips
for i in 01 02 03 04 05 06; do
    ln -s /data/datasets/PIE/set$i /data/datasets/PIE/PIE_clips/set$i
done
```

#### 2c. Frame Extraction

Both APIs store video as `.mp4` and require individual `.png` frames for training.
Extraction writes frames to:
- PIE: `/data/datasets/PIE/images/setXX/video_XXXX/XXXXX.png`
- JAAD: `/data/datasets/JAAD/images/video_XXXX/XXXXX.png`

Launched both in parallel as background processes:
```python
# PIE — extracts only annotated frames (not all frames)
pie.extract_and_save_images(extract_frame_type='annotated')

# JAAD
jaad.extract_and_save_images()
```

Datasets: 53 PIE videos, 346 JAAD videos. Both are running at time of writing.

---

### 3. Code Fixes

#### 3a. Created `models/EfficientPIE.py` (NEW)

Clean copy of `EfficientPIE_backup.py` with the broken import removed:
```python
# BEFORE (backup):
from models.common import DropPath, SqueezeExcite, ConvBNAct, FusedMBConv, MBConv, AdaptiveEncoder
# AdaptiveEncoder doesn't exist in common.py — ImportError

# AFTER (EfficientPIE.py):
from models.common import DropPath, SqueezeExcite, ConvBNAct, FusedMBConv, MBConv
```
Also removed the two commented-out `adaptive_encoder` lines from `__init__` and `forward`.

#### 3b. Fixed `utils/my_dataset.py`

**Remove broken import** (line 15):
```python
# REMOVED:
from utils.adaptive_selection import adaptive_selection  # module never existed
```

**Fix hardcoded `reverse_step`** (line 93):
```python
# BEFORE:
reverse_step = 1
# → always picks last frame, ignores IL step completely

# AFTER:
reverse_step = (self.data_opts['max_size_observe'] - self.step) if self.step is not None else 1
# → step=0 picks frame index 0 (earliest), step=14 picks frame index 14 (latest)
```

This is critical: the IDIL approach trains each step on a different temporal observation.
`reverse_step + step = 15 = max_size_observe` as noted in the original code comments.

#### 3c. Fixed `utils/train_val.py`

**Parameterize `robust_noisy()`** — the perturbation scale was hardcoded to 30 epochs,
but training runs for 50 epochs base and 30 epochs IL:
```python
# BEFORE:
def robust_noisy(pred, epoch):
    max_range = 0.5 * (epoch / 30)

# AFTER:
def robust_noisy(pred, epoch, total_epochs=30):
    max_range = 0.5 * (epoch / total_epochs)
```

**Parameterize `train_one_epoch()`** — added `total_epochs` parameter, passes it to
`robust_noisy()`.

**Fix `incremental_learning_train()`** — two important fixes:
1. Freeze `prev_model` properly:
```python
prev_model.eval()
for p in prev_model.parameters():
    p.requires_grad = False
```
2. Wrap prev_model forward pass in `torch.no_grad()` and detach output:
```python
# BEFORE:
prev_pred = prev_model(images.to(device))
loss_old = loss_old_func(prev_pred, labels)

# AFTER:
with torch.no_grad():
    prev_pred = prev_model(images.to(device))
loss_old = loss_old_func(prev_pred.detach(), labels)
```
This prevents gradients flowing into the frozen model and avoids unnecessary computation
graph memory.

#### 3d. Fixed `train_EfficientPIE.py`

| Change | Before | After |
|--------|--------|-------|
| Import | `from models.EfficientPIE_backup import EfficientPIE` | `from models.EfficientPIE import EfficientPIE` |
| `--data-path` default | `/home/yphe/.../PIEDataset` | `/data/datasets/PIE` |
| `--weights` default | `pre_train_weights_efficientpie/min_loss_...backup.pth` | `pre_train_weights/min_loss_pretrained_model_imagenet.pth` |
| `--device` default | `cuda:2` | `cuda:0` |
| `MyDataSet` call | Missing `step=` argument | Added `step=args.step` |
| Scheduler | `T_max=30` | `T_max=args.epochs` |
| `train_one_epoch` call | No `total_epochs` | Passes `total_epochs=args.epochs` |

#### 3e. Fixed `test_EfficientPIE.py`

| Change | Before | After |
|--------|--------|-------|
| Import | `from models.EfficientPIE_backup import EfficientPIE` | `from models.EfficientPIE import EfficientPIE` |
| `--data-path` default | `/home/yphe/.../PIEDataset` | `/data/datasets/PIE` |
| `--device` default | `cuda:6` | `cuda:0` |
| Weight path | Hardcoded `f"./weights_v{version}/model_8_PIE_IL_step14_new.pth"` | `args.weights` via new `--weights` CLI arg |
| Removed | `--version` arg (no longer needed) | — |

#### 3f. Fixed `train_EfficientPIE_JAAD.py`

| Change | Before | After |
|--------|--------|-------|
| `--data-path` default | `/home/yphe/.../JAAD` | `/data/datasets/JAAD` |
| `--device` default | `cuda:1` | `cuda:0` |
| `MyDataSet` call | No `step=` | Added `step=args.step` |
| `train_one_epoch` call | No `total_epochs` | Passes `total_epochs=args.epochs` |
| Added | — | `--step` CLI arg (default 0) |

#### 3g. Fixed `test_EfficientPIE_JAAD.py`

| Change | Before | After |
|--------|--------|-------|
| `--data-path` default | `/home/yphe/.../JAAD` | `/data/datasets/JAAD` |
| `--device` default | `cuda:2` | `cuda:0` |
| Weight path | Hardcoded `"./weights/transfer_noisy_model_JAAD.pth"` | `args.weights` via new `--weights` CLI arg |

#### 3h. Fixed `pie_domain_incremental_learning.py`

| Change | Before | After |
|--------|--------|-------|
| Import | `from models.EfficientPIE_backup import EfficientPIE` | `from models.EfficientPIE import EfficientPIE` |
| `--data-path` default | `/home/yphe/.../PIEDataset` | `/data/datasets/PIE` |
| `--weights` default | `pre_train_weights_efficientpie/...backup.pth` | `pre_train_weights/min_loss_pretrained_model_imagenet.pth` |
| Scheduler | `T_max=30` | `T_max=args.epochs` |
| `incremental_learning_train` | No `total_epochs` | Passes `total_epochs=args.epochs` |
| Removed stale comment | `# make sure reverse_step + step = 15` | (now handled in dataset code) |

#### 3i. Fixed `pretrain_imagenet.py`

| Change | Before | After |
|--------|--------|-------|
| Import | `from models.EfficientPIE_backup import EfficientPIE` | `from models.EfficientPIE import EfficientPIE` |
| Output dir | `pre_train_weights_efficientpie/` | `pre_train_weights/` |
| Save paths | `pre_train_weights_efficientpie/best_...` | `pre_train_weights/best_...` |
| Weight loading | Always loads (even from scratch) | Conditional: only if `args.resume` is set and exists |
| `--train_path` default | `/home/fqu/.../imagenet/train` | `/data/datasets/imagenet/train` |
| `--val_path` default | `/home/fqu/.../imagenet/val` | `/data/datasets/imagenet/val` |
| `--device` default | `cuda:6` | `cuda:0` |
| Added | — | `--resume` CLI arg for optional checkpoint |

#### 3j. Created `models/__init__.py` and `utils/__init__.py`

Both empty files to make the directories proper Python packages.

---

### 4. New Files Created

#### `requirements.txt`
```
torch>=1.7.0
torchvision>=0.8.0
numpy
scikit-learn
tqdm
Pillow
tensorboard
thop
opencv-python
```

#### `extract_frames.py`
Convenience script to run frame extraction for either or both datasets:
```bash
python extract_frames.py --dataset pie    # PIE only
python extract_frames.py --dataset jaad   # JAAD only
python extract_frames.py --dataset both   # both (default)
```

#### `jaad_domain_incremental_learning.py`
JAAD version of the IL training script, adapted from the PIE version:
- Imports `JAAD` instead of `PIE`
- Uses `output_type=['intent']` (JAAD label key) vs `['intention_binary']` (PIE)
- Default paths/weight names use JAAD naming convention
- Otherwise identical structure and hyperparameters

#### `run_pie_pipeline.sh`
Full PIE automation script:
1. Base training (step=0, 50 epochs, ImageNet weights)
2. IL steps 2→14 (30 epochs each, loading previous step's best weights)
3. Final evaluation on test set

#### `run_jaad_pipeline.sh`
Same structure for JAAD dataset.

#### `run_training_after_extraction.sh`
Combined script that:
1. Polls every 2 minutes until all 53 PIE videos are extracted
2. Automatically starts base training when done
3. Runs all IL steps sequentially
4. Runs final evaluation
5. Logs each stage to `training_logs/step{N}.log`

---

### 5. Smoke Test

Verified model imports and weight loading work correctly:

```
$ python -c "from models.EfficientPIE import EfficientPIE; m = EfficientPIE(); print(m)"
EfficientPIE(
  (commonConv): ConvBNAct(...)
  (fm1): FusedMBConv(...)
  (fm2): FusedMBConv(...)
  (mb1): MBConv(...)
  (mb2): MBConv(...)
  (commonConv1): ConvBNAct(...)
  (avg_pool): AdaptiveAvgPool2d(output_size=1)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (dropout): Dropout(p=0.2, inplace=True)
  (classifier): Linear(in_features=1280, out_features=2, bias=True)
)
```

Pre-trained ImageNet weights load correctly (74/76 keys — only classifier skipped since
ImageNet has 1000 classes, PIE uses 2):
```
Loaded keys: 74 / 76
_IncompatibleKeys(missing_keys=['classifier.weight', 'classifier.bias'], unexpected_keys=[])
```

All 12 Python files pass `py_compile` (syntax check).

---

### 6. Bugs Encountered and Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `PIE_clips` self-symlink caused infinite loop | `ls PIE_clips/` returned `PIE_clips` as entry, API tried to open `annotations/PIE_clips` | Replaced with real dir + per-set symlinks |
| Shell script CRLF errors | Write tool produced `\r\n` line endings, bash rejected them | `sed -i 's/\r//' *.sh` |
| First PIE extraction attempt failed | PIE_clips symlink bug (above) | Fixed symlink, re-launched |

---

### 7. Current Status

| Task | Status |
|------|--------|
| Code cleanup & bug fixes | ✅ Complete |
| Dataset symlinks | ✅ Complete |
| PIE frame extraction | 🔄 Running (~4/53 videos done) |
| JAAD frame extraction | 🔄 Running (~83/346 videos done) |
| Training pipeline | ⏳ Queued — auto-starts when PIE extraction finishes |
| Base training (step=0, 50 epochs) | ⏳ Pending |
| IL steps 2→14 (7 × 30 epochs) | ⏳ Pending |
| Final evaluation | ⏳ Pending |

GPU available: 2× NVIDIA RTX A6000 (49GB VRAM each, ~48GB free).

Training pipeline log: `/tmp/pie_training_pipeline.log`
Per-step logs (when training starts): `/data/repos/EfficientPIE/training_logs/`

---

### 8. Expected Training Timeline

| Phase | Duration estimate |
|-------|-----------------|
| PIE extraction (remaining ~49 videos) | ~1.5 hours |
| Base training step=0 (50 epochs, PIE) | ~2–4 hours |
| IL steps 2→14 (7 × 30 epochs) | ~7–14 hours |
| **Total** | **~10–20 hours** |

---

### 9. File Change Summary

| File | Action | Key changes |
|------|--------|-------------|
| `models/EfficientPIE.py` | **CREATED** | Clean model without broken AdaptiveEncoder import |
| `models/__init__.py` | **CREATED** | Empty package init |
| `utils/__init__.py` | **CREATED** | Empty package init |
| `requirements.txt` | **CREATED** | Full dependency list |
| `extract_frames.py` | **CREATED** | CLI wrapper for dataset frame extraction |
| `jaad_domain_incremental_learning.py` | **CREATED** | JAAD IL training script |
| `run_pie_pipeline.sh` | **CREATED** | Full PIE training automation |
| `run_jaad_pipeline.sh` | **CREATED** | Full JAAD training automation |
| `run_training_after_extraction.sh` | **CREATED** | Wait-for-extraction + full training |
| `utils/my_dataset.py` | **EDITED** | Remove broken import; fix `reverse_step` formula |
| `utils/train_val.py` | **EDITED** | Parameterize `total_epochs`; freeze `prev_model`; `no_grad` on prev_model forward |
| `train_EfficientPIE.py` | **EDITED** | Fix import, paths, `step=` to dataset, `T_max`, `total_epochs` |
| `test_EfficientPIE.py` | **EDITED** | Fix import, paths, configurable `--weights` arg |
| `train_EfficientPIE_JAAD.py` | **EDITED** | Fix paths, add `--step`, `step=` to dataset, `total_epochs` |
| `test_EfficientPIE_JAAD.py` | **EDITED** | Fix paths, configurable `--weights` arg |
| `pie_domain_incremental_learning.py` | **EDITED** | Fix import, paths, scheduler, `total_epochs` |
| `pretrain_imagenet.py` | **EDITED** | Fix import, paths, conditional resume, output dir |

---

### 10. Reproducing Results

Once training completes, evaluate with:

```bash
# PIE
python test_EfficientPIE.py \
    --data-path /data/datasets/PIE \
    --weights weights_v8/best_model_PIE_IL_step14_new.pth \
    --device cuda:0

# JAAD
python test_EfficientPIE_JAAD.py \
    --data-path /data/datasets/JAAD \
    --weights weights_v8/best_model_JAAD_IL_step14_new.pth \
    --device cuda:0
```

Target results (Table 3):

| Dataset | Accuracy | AUC  | F1   | Precision |
|---------|----------|------|------|-----------|
| PIE     | 0.92     | 0.92 | 0.95 | 0.96      |
| JAAD    | 0.89     | 0.86 | 0.62 | 0.63      |
