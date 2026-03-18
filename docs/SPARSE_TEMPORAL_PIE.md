# SparseTemporalPIE — Complete Implementation Guide

> Forked from [EfficientPIE](https://github.com/heinideyibadiaole/EfficientPIE) (IJCAI-25)
> Authors: Brandon Byrd, Abel Abebe Bzuayene — xDI Lab, NC A&T State University

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Design Decisions](#3-design-decisions)
4. [File-by-File Implementation](#4-file-by-file-implementation)
   - 4.1 [models/SparseTemporalPIE.py](#41-modelssparsetemporalpiepy)
   - 4.2 [utils/sparse_dataset.py](#42-utilssparse_datasetpy)
   - 4.3 [utils/change_detector.py](#43-utilschange_detectorpy-ablation-only)
   - 4.4 [train_SparseTemporalPIE.py](#44-train_sparsetemporalpiepy)
   - 4.5 [pie_sparse_incremental_learning.py](#45-pie_sparse_incremental_learningpy)
   - 4.6 [test_SparseTemporalPIE.py](#46-test_sparsetemporalpiepy)
   - 4.7 [run_sparse_pie_pipeline.sh](#47-run_sparse_pie_pipelinesh)
5. [Training Protocol](#5-training-protocol)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Ablation Study Design](#7-ablation-study-design)
8. [Expected Results & Baselines](#8-expected-results--baselines)
9. [Key Implementation Notes](#9-key-implementation-notes)

---

## 1. Architecture Overview

```
Observation window: 15 frames (indices 0–14), at IL step N we use index N.

F* SELECTION (in SparseDataset, CPU)
─────────────────────────────────────
For each frame t in [0, step-1]:
    embed_t = backbone(crop(frame_t, bbox_t))          # frozen backbone, CPU copy

f* = argmax  cosine_distance(embed_current, embed_t)
     t ∈ [0, step-1]

(at step=0, f* = frame 0 — no prior frames available)

FORWARD PASS (GPU)
──────────────────
f_current (300×300 crop) ──► frozen backbone ──► embed_current (1280)
f*        (300×300 crop) ──► frozen backbone ──► embed_fstar   (1280)

CrossAttention(Q=embed_current, K=embed_fstar, V=embed_fstar)
    │
    attn_out (1280)
    │
residual + LayerNorm:  enriched_0 = LayerNorm(embed_current + attn_out)
    │
FeedForward (1280 → 512 → 1280):  enriched = LayerNorm(enriched_0 + FF(enriched_0))
    │
    enriched (1280)

pose_feats (34) ← normalize_pose(keypoints_pid/{pid}/{frame_id}.npy, bbox)
                  17 joints × (norm_x, norm_y); low-conf joints set to 0

concat([enriched, pose_feats])  →  1314-d
    │
Classifier: Linear(1314→256) → ReLU → Dropout → Linear(256→2)
    │
logits  →  crossing probability
```

**What f\* represents semantically:**
The frame in the observation window that looks *most different* from the current moment,
according to the frozen backbone's visual representation. If the pedestrian is mid-turn,
shifting weight, or changing gaze direction, that frame will pull apart from f\_current
in embedding space. Cross-attention then asks: "given what f\* looked like then and what
f\_current looks like now, what changed?" The pose features independently encode the
current body configuration, providing a complementary signal.

**Why this replaces ChangeDetector:**
An earlier design used a ViTPose-based ChangeDetector (head orientation delta, body lean
delta, gaze vector delta) calibrated on PIE behavioral annotations. Calibration AUC
reached only 0.54 (barely above random), confirmed by 100%-hit-rate keypoints from
correctly-cropped ViTPose extraction. Root causes: many PIE pedestrians have small
bounding boxes (bbox\_h < 60px) degrading pose quality; the binary fired/not-fired
signal discards too much temporal information. Embedding-distance f\* selection uses the
backbone's own rich visual features and requires no calibration.

---

## 2. Repository Structure

Fork from EfficientPIE. New and modified files marked.

```
EfficientPIE/  (forked)
│
├── models/
│   ├── __init__.py
│   ├── common.py                          # UNCHANGED — DropPath, SE, FusedMBConv, MBConv
│   ├── EfficientPIE.py                    # UNCHANGED — baseline model
│   └── SparseTemporalPIE.py               # NEW — temporal cross-attention model
│
├── utils/
│   ├── __init__.py
│   ├── my_dataset.py                      # UNCHANGED — EfficientPIE dataset
│   ├── train_val.py                       # MODIFIED — added evaluate_sparse(),
│   │                                      #            incremental_learning_train_sparse()
│   ├── change_detector.py                 # NEW (ablation only — not used in pipeline)
│   └── sparse_dataset.py                  # NEW — SparseDataset (f_current, f*, pose_feats)
│
├── train_EfficientPIE.py                  # UNCHANGED
├── pie_domain_incremental_learning.py     # UNCHANGED
├── test_EfficientPIE.py                   # UNCHANGED
│
├── train_SparseTemporalPIE.py             # NEW — base training (step=0, 50 epochs)
├── pie_sparse_incremental_learning.py     # NEW — IL steps 2→14 (30 epochs each)
├── test_SparseTemporalPIE.py              # NEW — eval + v=0 subset metrics
│
├── run_sparse_pie_pipeline.sh             # NEW — step0 + IL + eval automation
├── extract_frames.py                      # EXISTING — video → frames (one-time)
├── extract_keypoints.py                   # NEW — ViTPose keypoint extraction (one-time)
├── calibrate_change_detector.py           # NEW (ablation tool — not in pipeline)
└── pre_train_weights/
    └── min_loss_pretrained_model_imagenet.pth   # EXISTING
```

**Data paths:**
```
/data/datasets/PIE/
  images/setXX/video_XXXX/XXXXX.png       # extracted frames
  keypoints_pid/{pid}/{frame_id}.npy       # ViTPose keypoints (17×3), abs frame number
  annotations/ (symlinked)
  PIE_clips/setXX/ (symlinked)

/data/datasets/JAAD/
  images/video_XXXX/XXXXX.png
  annotations/ (symlinked)
```

---

## 3. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Backbone | EfficientPIE (shared, Siamese) | Clean comparison; attribution of delta is purely from temporal mechanism |
| Backbone weights | Trained PIE IL-step-14 weights, **frozen** | Accuracy delta attributable to temporal mechanism alone; no catastrophic forgetting |
| f\* selection | argmax cosine distance from embed\_current in [0, step-1] | No calibration; uses backbone's own feature space; captures largest visual change |
| f\* at step=0 | frame 0 (same as f\_current) | No prior frames; cross-attention is a no-op (identical Q and K/V) |
| Pose features | 34-dim normalized keypoint vector for f\_current | Rich body-pose signal; complements temporal attention context; 17 joints, no conf channel needed after thresholding |
| Pose normalization | Divide by bbox width/height | Removes camera/distance variation; coords ∈ [0,1] approximately |
| Low-conf keypoints | Zeroed (conf < 0.25) | Avoids noisy joints misleading the classifier; zero is distinguishable from valid keypoints near bbox boundary |
| Hard gate | **Removed** | pose\_feats already encode "no pose" via zeros; gate introduced an information bottleneck without benefit |
| Classifier input | concat([enriched, pose\_feats]) → 1314-d | Pose provides explicit body config; attention provides temporal delta |
| Classifier head | Linear(1314→256) → ReLU → Dropout → Linear(256→2) | 256 hidden dim adequate for 1314-d input; extra layer vs. original 1281→2 |
| Backbone in dataset | CPU deep-copy (pickle-safe) | Embedding-distance f\* computed in `__getitem__` → must be multiprocessing-safe |
| IL strategy | Inherited from EfficientPIE (IDIL + adaptive loss) | Compatible; backbone freeze simplifies IL (only attention head + FF + classifier drift) |
| ViTPose model | `usyd-community/vitpose-base-coco-aic-mpii` (ViTPose-B) | Best HF-available model; COCO+AIC+MPII training |
| ChangeDetector | Kept in codebase (**ablation only**) | Ablation A6: compare embedding-distance vs. threshold-based f\* selection |

---

## 4. File-by-File Implementation

---

### 4.1 `models/SparseTemporalPIE.py`

**Purpose:** Full model definition. Wraps a frozen EfficientPIE backbone as a shared
encoder, adds cross-attention fusion, feedforward layer, and classifier head that accepts
normalized pose features.

**Key classes/functions:**

- `SparseTemporalPIE(num_heads=8, ff_hidden=512, dropout=0.1)` — the full model
- `encode(x)` — passes (B, 3, 300, 300) through frozen backbone → (B, 1280)
- `forward(f_current, f_star, pose_feats)` — main forward pass
- `load_backbone_weights(model, weights_path, device)` — loads EfficientPIE checkpoint
  into backbone by mapping attribute names to Sequential indices

**Parameter counts:**
- Backbone: ~684K (frozen, not trained)
- CrossAttention (8 heads, 1280-dim): ~6.6M
- FeedForward (1280→512→1280): ~1.3M
- Classifier (1314→256→2): ~336K
- Total trainable: ~8.2M

**Forward pass in detail:**
```
f_current  (B, 3, 300, 300) → backbone → embed_current  (B, 1280)
f_star     (B, 3, 300, 300) → backbone → embed_fstar    (B, 1280)

# Cross-attention (seq_len=1 for both Q and K/V)
q = embed_current.unsqueeze(0)   # (1, B, 1280)
k = embed_fstar.unsqueeze(0)     # (1, B, 1280)
v = embed_fstar.unsqueeze(0)     # (1, B, 1280)
attn_out, _ = cross_attn(q, k, v)
attn_out = attn_out.squeeze(0)   # (B, 1280)

# Transformer-style residual blocks
enriched_0 = attn_norm(embed_current + attn_out)          # (B, 1280)
enriched   = ff_norm(enriched_0 + ff(enriched_0))         # (B, 1280)

# Classifier
combined = torch.cat([enriched, pose_feats], dim=1)        # (B, 1314)
logits   = classifier(combined)                            # (B, 2)
```

---

### 4.2 `utils/sparse_dataset.py`

**Purpose:** PyTorch Dataset that returns `(f_current, f_star, pose_feats, label)` per
sample. Handles f\* selection via embedding cosine distance and pose feature loading.

**Key functions:**

`normalize_pose(kpts, bbox, conf_thresh=0.25) → np.ndarray (34,)`
- Input: `kpts` (17, 3) array [x, y, conf] in full-frame pixel coords, `bbox` [x1,y1,x2,y2]
- For each joint: if conf ≥ thresh, store `(x-x1)/bw` and `(y-y1)/bh`; else store 0,0
- Returns float32 array of shape (34,)

`SparseDataset.__init__(images_seq, data_opts, step, transform, keypoints_dir, backbone)`
- `backbone`: optional frozen nn.Sequential — deep-copied to CPU, set to eval mode
- A separate `_encode_transform` (no augmentation) is used for backbone encoding

`SparseDataset._select_fstar(frame_paths, bboxes) → int`
- Returns frame index in [0, step-1] with max cosine distance from f\_current embedding
- Falls back to 0 if backbone is None or step == 0
- Encodes [0..step] frames in one batch on CPU backbone; cosine\_similarity broadcasts

`SparseDataset._load_pose_feats(pid, t, bbox) → np.ndarray (34,)`
- Loads `{keypoints_dir}/{pid}/{frame_id}.npy` where frame\_id is extracted from image path
- Returns zeros if file missing (graceful degradation)

**Keypoint path convention:**
```
{keypoints_dir}/{pid}/{frame_id}.npy
```
where `frame_id` is the zero-padded absolute frame number from the image filename
(e.g., `00123`). This avoids collision across non-overlapping observation windows for
the same pedestrian. See SESSION_NOTES_2026-03-11 Bug 3.

**`__getitem__` flow:**
```
1. f_star_idx  = _select_fstar(frame_paths, bboxes)
2. pose_feats  = _load_pose_feats(pid, frame_id_at_step, bboxes[step])
3. Load + crop f_current (frame_paths[step]) and f_star (frame_paths[f_star_idx])
4. Apply self.transform to both crops
5. Return (f_current_t, f_star_t, pose_feats_t, label_t)
```

**`collate_fn`:** stacks all four tensors — returns
`(B,3,300,300), (B,3,300,300), (B,34), (B,)`

---

### 4.3 `utils/change_detector.py` (ablation only)

**Status:** Retained in codebase but **not called** by the training pipeline.

**Purpose:** Original ViTPose-based keyframe selector using head orientation delta, body
lean delta, and gaze vector delta signals. Threshold-based, calibrated via grid search
on PIE behavioral annotations.

**Why kept:** Needed for ablation A6 (compare embedding-distance f\* vs. threshold-based
f\*). Import it explicitly in ablation scripts; do not import from sparse\_dataset.py.

**Calibration result (2026-03-11):** Best AUC = 0.54 on PIE train set. Thresholds hit
grid edges, indicating the binary signal is too weak. See SESSION_NOTES_2026-03-11 for
full analysis.

---

### 4.4 `train_SparseTemporalPIE.py`

**Purpose:** Base training at IL step=0 (50 epochs). Trains all parameters except the
frozen backbone.

**Critical:** Model is constructed **before** datasets so `model.backbone` can be passed
to `SparseDataset` for embedding-distance f\* selection:

```python
model = SparseTemporalPIE(num_heads=8, ff_hidden=512).to(device)
model = load_backbone_weights(model, args.weights, device=args.device)

train_dataset = SparseDataset(..., backbone=model.backbone)
val_dataset   = SparseDataset(..., backbone=model.backbone)
```

**At step=0:** f\_star = f\_current (same frame, no history). Cross-attention is a no-op
(Q = K = V). The model learns from pose features and the embed\_current alone at this step.

**Outputs:**
- `weights_sparse/best_sparse_step0.pth` — best validation accuracy
- `weights_sparse/min_loss_sparse_step0.pth` — minimum validation loss

**CLI:**
```bash
python train_SparseTemporalPIE.py \
    --weights   weights_v8/model_8_PIE_IL_step14_new.pth \
    --keypoints-dir /data/datasets/PIE/keypoints_pid \
    --step 0 --epochs 50 --device cuda:0
```

---

### 4.5 `pie_sparse_incremental_learning.py`

**Purpose:** IL chain over steps [2, 4, 6, 8, 10, 12, 14]. Each step loads the previous
step's best weights as a frozen `prev_model` for IL loss, and trains a `curr_model` that
warm-starts from the same weights.

**Model construction order (same pattern as train script):**
```python
prev_model = SparseTemporalPIE().to(device)
prev_model.load_state_dict(torch.load(prev_weights))
prev_model.eval(); freeze all params

curr_model = SparseTemporalPIE().to(device)
curr_model.load_state_dict(torch.load(prev_weights))
freeze curr_model.backbone only

# Pass curr_model backbone to datasets (already frozen, weights same as prev_model)
train_dataset = SparseDataset(..., backbone=curr_model.backbone)
val_dataset   = SparseDataset(..., backbone=curr_model.backbone)
```

**IL loss (inherited from EfficientPIE):**
```
loss_new = CrossEntropy(robust_noisy(pred_curr), labels)
loss_old = loss_old_func(pred_prev.detach(), labels)
loss = loss_new if loss_new > loss_old else 0.5 * loss_old + loss_new
```

**CLI:**
```bash
python pie_sparse_incremental_learning.py \
    --weights weights_sparse/best_sparse_step0.pth \
    --keypoints-dir /data/datasets/PIE/keypoints_pid \
    --epochs 30 --device cuda:0
```

---

### 4.6 `test_SparseTemporalPIE.py`

**Purpose:** Evaluation on PIE test set. Reports two metric sets:
1. **Overall** — full test set
2. **v=0 subset** — stationary pedestrians (bbox center displacement < epsilon × bbox width)

**Metrics:** Accuracy, AUC, F1, Precision (via sklearn).

**Model loaded before dataset** (same pattern as train scripts):
```python
model = SparseTemporalPIE().to(device)
model.load_state_dict(torch.load(args.weights))
model.eval()

test_dataset = SparseDataset(..., backbone=model.backbone)
```

**v=0 stationarity:**
```python
delta = sqrt((cx_N - cx_0)^2 + (cy_N - cy_0)^2) / bbox_width
stationary = delta < epsilon   # default epsilon=5.0
```
Stationary pedestrians are harder cases — they exhibit less visual motion signal,
so the temporal f\* mechanism is hypothesized to help more here.

**CLI:**
```bash
python test_SparseTemporalPIE.py \
    --weights weights_sparse/best_sparse_step14.pth \
    --keypoints-dir /data/datasets/PIE/keypoints_pid \
    --step 14 --device cuda:0
```

---

### 4.7 `run_sparse_pie_pipeline.sh`

Full automation: step 0 → IL steps 2–14 → evaluation.

```bash
bash run_sparse_pie_pipeline.sh
```

Logs written to `training_logs_sparse/`:
- `step0.log` — step 0 training
- `il_steps.log` — IL steps 2–14
- `evaluation.log` — final test set metrics

Prerequisites:
1. `python extract_frames.py --dataset pie`
2. `python extract_keypoints.py --dataset pie --output-dir /data/datasets/PIE/keypoints_pid`

No calibration step required.

---

## 5. Training Protocol

### Step 0 (base training)

| Param | Value |
|---|---|
| Epochs | 50 |
| Optimizer | RMSprop |
| LR | 1e-4 |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealingLR (eta_min=1e-7) |
| Batch size | 32 |
| Augmentation | RandomHorizontalFlip, ColorJitter(b=0.5, c=0.5, s=0.5, h=0.1) |
| Loss | CrossEntropyLoss + robust\_noisy perturbation |

### IL Steps 2–14

Same optimizer / LR / schedule, but:
- 30 epochs per step
- IL loss: adaptive combination of `loss_new` and `loss_old`
- Backbone frozen in `curr_model`; all other params train
- `prev_model` fully frozen (eval mode, no grad)

### Perturbation (inherited from EfficientPIE)

`robust_noisy(logits, epoch, total_epochs)` applies progressive perturbation to logits:
- Perturbation magnitude grows with epoch: `m_j = epoch / total_epochs`
- Noise vector: `[p_j, -p_j]` where `p_j = rand(-m_j/E, m_j/E)`

---

## 6. Evaluation Protocol

**Dataset split:** PIE random split (train/val/test), `seq_overlap_rate=0.5`.

**Metrics:**
- Accuracy, AUC (sklearn `roc_auc_score`), F1, Precision
- v=0 subset: same metrics on stationary pedestrians only

**Checkpointing:** best validation accuracy checkpoint is used for final eval.

**Reference (EfficientPIE, author weights):**

| Metric | Value |
|---|---|
| Accuracy | 0.918 |
| AUC | 0.917 |
| F1 | 0.952 |
| Precision | 0.961 |
| Inference | 0.279 ms |

---

## 7. Ablation Study Design

All ablations use the same IL training protocol. Only one component changes per ablation.

| ID | Name | Change | Tests |
|---|---|---|---|
| A1 | No temporal comparison | f\* = f\_current (attention is no-op) | Does f\* selection contribute? |
| A2 | f\* = frame 0 always | Fixed first frame, no distance selection | Does *which* f\* matters? |
| A3 | Random f\* | f\* sampled uniformly from [0, step-1] | Is embedding distance better than random? |
| A4 | No pose features | pose\_feats = zeros (B,34) | How much does body pose contribute? |
| A5 | No cross-attention | Replace attn with identity; use only pose\_feats + embed\_current | How much does temporal attention contribute? |
| A6 | ChangeDetector f\* | Replace cosine-distance f\* with `ChangeDetector.detect()` | Is embedding distance better than threshold-based detection? |

Run individual ablations by modifying `SparseDataset._select_fstar()` or
`SparseTemporalPIE.forward()` and re-running `run_sparse_pie_pipeline.sh`.

---

## 8. Expected Results & Baselines

| Model | Acc | AUC | F1 | Prec | Notes |
|---|---|---|---|---|---|
| EfficientPIE (paper) | 0.92 | 0.92 | 0.95 | 0.96 | Table 3 |
| EfficientPIE (replicated) | 0.918 | 0.917 | 0.952 | 0.961 | author weights |
| SparseTemporalPIE | TBD | TBD | TBD | TBD | training in progress |
| A1: no f\* comparison | TBD | — | — | — | |
| A4: no pose\_feats | TBD | — | — | — | |

Target: beat EfficientPIE replicated results. The pose features add direct body-config
signal not available in the original; embedding-distance f\* provides a calibration-free
temporal reference.

---

## 9. Key Implementation Notes

### Backbone is frozen everywhere

The backbone in `SparseTemporalPIE` has `requires_grad=False` on all parameters.
`load_backbone_weights()` maps EfficientPIE state-dict keys by attribute name prefix
to backbone Sequential indices (0=commonConv, 1=fm1, ..., 8=dropout).
The classifier head keys in the EfficientPIE checkpoint are simply not mapped (missing
in target → ignored by `strict=False`).

### Backbone copy in SparseDataset

`SparseDataset.__init__` deep-copies the backbone to CPU and sets it to eval mode.
This is required because DataLoader workers are forked subprocesses — CUDA tensors
are not safe to share across fork. The CPU copy is ~2.7 MB; with 8 workers the total
overhead is ~22 MB (negligible).

The encode\_transform used inside `_select_fstar` is deterministic (no augmentation),
matching the val\_transform used at training time.

### Keypoint path convention

```
{keypoints_dir}/{pid}/{frame_id}.npy
```

`frame_id` is the zero-padded absolute frame number extracted from the image filename
(e.g., image path `.../00123.png` → `frame_id = "00123"`). Using the window index
(0–14) would cause collisions when the same pedestrian appears in multiple
non-overlapping observation windows — different sequences would overwrite each other's
keypoints. The absolute frame number is unique per (pid, frame).

Extracted by `extract_keypoints.py` using ViTPose-B (`usyd-community/vitpose-base-coco-aic-mpii`).
PIE dataset: 150,240 files, 100% hit rate (centroid within bbox ±20px).

### IL step and f* history

At IL step N, the dataset uses frame at index N as f\_current. f\* is selected from
frames [0, N-1]. As N increases (observation window grows), the model sees more
temporal context and the embedding-distance f\* can pick a more diverse reference frame.

Steps: 0 → 2 → 4 → 6 → 8 → 10 → 12 → 14 (even indices only, following EfficientPIE IDIL protocol).

### Robust noisy perturbation

Inherited from `utils/train_val.py:robust_noisy()`. Applies to logits before loss:
```python
m_j = epoch / total_epochs
p_j = random.uniform(-m_j / E, m_j / E)   # E = embedding_dim conceptually
logits[:, 0] += p_j
logits[:, 1] -= p_j
```
Perturbation grows with epoch, acting as a curriculum that gradually increases
difficulty during training.

### Cross-attention with seq_len=1

`nn.MultiheadAttention` expects `(seq_len, batch, embed_dim)`. Both Q and K/V have
seq\_len=1 (single embedding per image). The attention reduces to a learned linear
transformation of embed\_fstar gated by similarity to embed\_current — effectively
a soft selection of which aspects of f\* to bring into the enriched representation.
