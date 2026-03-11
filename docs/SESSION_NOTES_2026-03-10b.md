# Session Notes — 2026-03-10 (Evening)

> Continuation of the morning session. See `SESSION_NOTES_2026-03-10.md` for the EfficientPIE replication work done earlier.

---

## What Was Done

### 1. Confirmed Replication Results (Live Eval)

Re-ran evaluation with fixed code to get verified numbers on record.

**Weights:** `weights_v8/model_8_PIE_IL_step14_new.pth` (author-provided)
**Log:** `training_logs/final_eval_fixed.log`

| Metric | Replicated | Paper (Table 3) |
|--------|-----------|-----------------|
| Accuracy | 0.918 | 0.92 |
| AUC | 0.917 | 0.92 |
| F1 | 0.952 | 0.95 |
| Precision | 0.961 | 0.96 |
| Recall | 0.943 | — |
| Inference time | 0.279 ms | 0.21 ms (diff GPU) |
| FLOPs | 909.43 M | — |
| Parameters | 0.69 M | — |

Results match within rounding on all four reported metrics. Summary saved to `REPLICATION_RESULTS.md`.

---

### 2. SparseTemporalPIE — Full Implementation

Decision: iterate immediately to SparseTemporalPIE rather than re-run full training from scratch. Rationale: eval numbers match paper, bugs were in data loading not training logic, backbone is frozen in SparseTemporalPIE so fresh training is fast.

#### Files Created

| File | Purpose |
|------|---------|
| `models/SparseTemporalPIE.py` | Full model — frozen Siamese EfficientPIE backbone + cross-attention + feedforward + classifier |
| `utils/change_detector.py` | ViTPose pose-signal based keyframe selector + calibration |
| `utils/sparse_dataset.py` | PyTorch Dataset returning `(f_current, f_star, absence_flag, label)` |
| `train_SparseTemporalPIE.py` | Base training script (step=0, 50 epochs) |
| `pie_sparse_incremental_learning.py` | IL training chain (steps 2→14, 30 epochs each) |
| `test_SparseTemporalPIE.py` | Evaluation — overall + v=0 (stationary pedestrian) subset |
| `extract_keypoints.py` | Offline ViTPose-B keypoint extraction |
| `calibrate_change_detector.py` | Threshold grid search against PIE crossing labels |
| `run_sparse_pie_pipeline.sh` | Full pipeline automation |

#### Additions to Existing Files

- `utils/train_val.py`: added `evaluate_sparse()` and `incremental_learning_train_sparse()` — same logic as EfficientPIE equivalents but handles 3-input forward `model(f_current, f_star, flag)`

#### Key Design Choices Made

**SparseDataset interface**: Accepts the same `images_seq` dict from `get_train_val_data()` as `MyDataSet` (not an internal PIE API call). Cleaner, no code duplication.

**Keypoint path convention**: Mirrors image directory structure.
- Image: `/data/datasets/PIE/images/set01/video_0002/05051.png`
- Keypoint: `/data/datasets/PIE/keypoints/set01/video_0002/05051.npy`
No pid-based lookup — derived directly from image path.

**ViTPose backend**: `usyd-community/vitpose-base-coco-aic-mpii` via HuggingFace `transformers` (installed: `transformers==5.3.0`, `timm==1.0.25`). MMPose could not be installed (setup.py KeyError). YOLO pose rejected — guide specifies ViTPose.

**Backbone weights**: `weights_v8/model_8_PIE_IL_step14_new.pth` used to initialise frozen backbone (best available EfficientPIE weights).

**Trainable parameter count**: 8,205,058 (cross-attention + feedforward + classifier head). Backbone: 684,176 frozen.

**Hard gate on cross-attention** (added 2026-03-11): `attn_out = attn_out * (1 - absence_flag)`. When f* didn't fire, the gate is 0 and the attention output is suppressed entirely — the classifier sees only `embed_current + absence_flag`. When f* fired, gate=1 and full attention flows. Prevents frame-0 fallback embeddings from polluting the representation. Added A5 to ablation table to measure gating value vs. implicit learning. See `models/SparseTemporalPIE.py` forward pass.

#### Smoke Tests Passed

```
Backbone weights loaded: 74/76 keys
Trainable: 8,205,058  |  Frozen (backbone): 684,176
Output (fired):  [4, 2]  — gate open, attention flows       ✓
Output (absent): [4, 2]  — gate closed, attention suppressed ✓
fired != absent (gate is active)                             ✓

SparseDataset length: 893 (test split, step=0)
f_current: [3, 300, 300], f_star: [3, 300, 300], flag: [1.], label: 1  ✓
```

---

### 3. To Run Next Session

**Step 1: Extract keypoints** (runs overnight, fully resumable)
```bash
python extract_keypoints.py \
    --dataset pie \
    --data-path /data/datasets/PIE \
    --output-dir /data/datasets/PIE/keypoints \
    --device cuda:0
```

**Step 2: Calibrate change detector** (after keypoints done)
```bash
python calibrate_change_detector.py \
    --data-path /data/datasets/PIE \
    --keypoints-dir /data/datasets/PIE/keypoints \
    --output change_detector_config.json
```

**Step 3: Full training pipeline**
```bash
bash run_sparse_pie_pipeline.sh
```

---

## Current Repo State

```
EfficientPIE/
├── models/
│   ├── common.py                         UNCHANGED
│   ├── EfficientPIE.py                   fixed (prev session)
│   └── SparseTemporalPIE.py              NEW
├── utils/
│   ├── change_detector.py                NEW
│   ├── my_dataset.py                     fixed (prev session)
│   ├── sparse_dataset.py                 NEW
│   └── train_val.py                      + evaluate_sparse, incremental_learning_train_sparse
├── train_EfficientPIE.py                 fixed (prev session)
├── test_EfficientPIE.py                  fixed (prev session)
├── pie_domain_incremental_learning.py    fixed (prev session)
├── train_SparseTemporalPIE.py            NEW
├── pie_sparse_incremental_learning.py    NEW
├── test_SparseTemporalPIE.py             NEW
├── extract_keypoints.py                  NEW
├── calibrate_change_detector.py          NEW
├── run_sparse_pie_pipeline.sh            NEW
├── REPLICATION_RESULTS.md                NEW — verified eval numbers
└── SESSION_NOTES_2026-03-10b.md          this file
```
