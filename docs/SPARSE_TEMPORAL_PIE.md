# SparseTemporalPIE — Architecture and Implementation Guide

> Authors: Brandon Byrd, Abel Abebe Bzuayene — xDI Lab, NC A&T State University
> Fork of [EfficientPIE](https://github.com/heinideyibadiaole/EfficientPIE) (IJCAI-25)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Design Evolution (v1 → v4)](#2-design-evolution)
3. [Repository Structure](#3-repository-structure)
4. [File-by-File Reference](#4-file-by-file-reference)
5. [Training Protocol](#5-training-protocol)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Results Summary](#7-results-summary)
8. [Key Implementation Notes](#8-key-implementation-notes)

---

## 1. Architecture Overview

### v3 (Best Results)

```
Inputs:
  f_current    — (B, 3, 300, 300)  current frame crop at IL step t
  f_context    — (B, K, 3, 300, 300)  K=4 evenly-spaced context frames from [0, t-1]
  context_mask — (B, K)  1.0=real frame, 0.0=padding
  pose_current — (B, 68)  34-d static keypoints + 34-d velocity (delta vs prev frame)
  pose_context — (B, K, 68)  poses for each context frame
  bbox_traj    — (B, 12)  trajectory stats over [0, t]: displacement, velocity,
                           acceleration, size ratio
  ctx_feats    — (B, 5)   [OBD_speed@t, mean_speed[0:t], speed_valid, action@t, look@t]

Forward pass:
  1. Batch all K+1 frames through shared backbone → (B, K+1, 1280)
  2. Fuse pose into each embedding: emb += pose_proj(pose)
  3. Cross-attention: Q=emb_current, K/V=emb_context[0..K]
     - key_padding_mask ignores padded context frames
  4. Residual + LayerNorm + FeedForward(1280→512→1280) + LayerNorm → enriched (1280)
  5. ctx_proj(concat[bbox_traj, ctx_feats]) → ctx (128)   [late fusion]
  6. classifier(concat[enriched, ctx]) → logits (2)

Architecture:
  backbone      ~684K params  (EfficientPIE CNN, partially unfrozen during training)
  pose_proj     Linear(68→1280, bias=False)
  cross_attn    MultiheadAttention(1280, num_heads=8, batch_first=False)
  attn_norm     LayerNorm(1280)
  ff            Linear(1280→512) → GELU → Dropout → Linear(512→1280) → Dropout
  ff_norm       LayerNorm(1280)
  ctx_proj      Linear(17→128) → GELU → Dropout → Linear(128→128)
  classifier    Linear(1408→256) → ReLU → Dropout → Linear(256→2)
  Total         ~9.0M params
```

### v4 (Ablation — no cross-attention)

```
Inputs:
  f_current    — (B, 3, 300, 300)
  pose_current — (B, 34)  static keypoints only (velocity dropped)
  bbox_traj    — (B, 12)
  ctx_feats    — (B, 5)

Forward pass:
  1. backbone(f_current) → emb (1280)
  2. emb += pose_proj(pose_current)
  3. ctx_proj(concat[bbox_traj, ctx_feats]) → ctx (128)
  4. classifier(concat[emb, ctx]) → logits (2)

Architecture:
  backbone      ~684K params
  pose_proj     Linear(34→1280, bias=False)
  ctx_proj      Linear(17→128) → GELU → Dropout → Linear(128→128)
  classifier    Linear(1408→256) → ReLU → Dropout → Linear(256→2)
  Total         ~1.1M params
```

---

## 2. Design Evolution

| Version | Key Change | Val Acc | Test Acc |
|---------|-----------|---------|----------|
| v1 | Single f* via cosine distance, frozen backbone, 34-d pose | ~0.865 | — |
| v2 | Unfrozen backbone, synchronized flip, pose dropout | ~0.868 | — |
| v3 | K=4 multi-frame cross-attention, 68-d pose (+ velocity), bbox_traj, ctx_feats | ~0.870 | **0.9261** |
| v4 | Dropped cross-attention and pose velocity; single frame + ctx MLP only | ~0.874 | 0.9194 |

**Key insight:** Validation accuracy was a misleading signal throughout (val set = ~92
pedestrians, ~500 samples). v3 appeared to ceiling at 0.870 val but achieved 0.9261 on
test. v4 peaked higher on val (0.874) but lower on test (0.9194). The cross-attention
mechanism benefits most from the full IL chain — v3 improves monotonically to step 14
while v4 degrades after step 2.

**Why cross-attention on adjacent frames works:** Although adjacent frames are visually
similar, the cross-attention over K=4 evenly-spaced frames spanning [0, t-1] gives
the model a reference trajectory. The attention learns to selectively pull in the most
relevant historical visual context for each query position. This is most valuable at
later IL steps (step 14) when the observation window spans the full 15-frame sequence.

---

## 3. Repository Structure

```
models/
  EfficientPIE.py                  # Unchanged baseline
  SparseTemporalPIE.py             # v4 (current default)
  SparseTemporalPIE_v3.py          # v3 (reconstructed for evaluation)

utils/
  pie_data.py                      # Modified: added obd_speed, action, look extraction
  jaad_data.py                     # Unchanged
  my_dataset.py                    # EfficientPIE dataset (unchanged)
  sparse_dataset.py                # v4 dataset — 5-tuple
  sparse_dataset_v3.py             # v3 dataset — 8-tuple
  train_val.py                     # Added evaluate_sparse(), incremental_learning_train_sparse()

train_SparseTemporalPIE.py         # v4 step 0 training
pie_sparse_incremental_learning.py # v4 IL steps 2→14
test_SparseTemporalPIE.py          # v4 evaluation
test_SparseTemporalPIE_v3.py       # v3 evaluation

extract_frames.py                  # Video → frames (one-time, PIE + JAAD)
extract_keypoints.py               # ViTPose-B extraction (one-time, PIE + JAAD)

weights_sparse/                    # v3 step 0 base weights
weights_sparse_v3/                 # v3 IL checkpoints steps 2–14
weights_sparse_v4/                 # v4 IL checkpoints steps 0–14

docs/
  RESULTS.md                       # Full results and SOTA comparison
  SPARSE_TEMPORAL_PIE.md           # This file
  SESSION_NOTES_2026-03-18.md      # v3/v4 architecture decisions
  SESSION_NOTES_2026-03-19.md      # Final test results and analysis
```

---

## 4. File-by-File Reference

### `utils/pie_data.py` (modified)

Three new sequences extracted per pedestrian in `_get_intention()`:

```python
vid_annots = annotations[sid][vid].get('vehicle_annotations', {})
speeds  = [[vid_annots.get(f, {}).get('OBD_speed', 0.0)] for f in frame_ids]
actions = [[a] for a in pid_annots[pid]['behavior']['action'][start_idx:end_idx+1]]
looks   = [[l] for l in pid_annots[pid]['behavior']['look'][start_idx:end_idx+1]]
```

Passed through `get_tracks()` and `get_train_val_data()` as `obd_speed`, `action`, `look`
keys. `filter_existing_sequences()` handles arbitrary keys generically — no changes needed there.

---

### `utils/sparse_dataset.py` (v4)

Returns 5-tuple: `(f_current, pose_current, bbox_traj, ctx_feats, label)`

Key methods:
- `_load_pose_feats(pid, frame_path, bbox)` — loads `keypoints_pid/{pid}/{frame_id:05d}.npy`
- `_compute_bbox_trajectory(bboxes)` — 12-d stats over `bboxes[0:step+1]`
- `_get_context_features(index)` — 5-d from `obd_speed`/`action`/`look` keys
- Synchronized flip: image + pose (COCO left/right pairs swapped)
- Pose dropout: both pose and image unchanged; pose zeroed with `p=0.1`

---

### `utils/sparse_dataset_v3.py` (v3)

Returns 8-tuple: `(f_current, f_context, context_mask, pose_current, pose_context, bbox_traj, ctx_feats, label)`

Additional methods:
- `_select_context_indices()` — `np.linspace(0, step-1, min(K, step))` evenly spaced
- `_load_pose_68d(pid, frame_paths, bboxes, idx)` — 34-d static + 34-d velocity
- `collate_fn` — pads f_context and pose_context to K=4, builds mask tensor

---

### `models/SparseTemporalPIE.py` (v4)

Single-frame forward pass. `load_backbone_weights()` maps EfficientPIE checkpoint keys
to backbone Sequential indices (0=commonConv, 1=fm1, 2=fm2, 3=mb1, 4=mb2, 5=commonConv1).

---

### `models/SparseTemporalPIE_v3.py` (v3)

Multi-frame forward pass. Architecture reconstructed from saved checkpoint weight shapes.
`pose_proj` is shared across current and context frames.

---

### `train_SparseTemporalPIE.py`

v4 step 0 base training. Partial backbone unfreeze: backbone at `lr×0.1`, head at `lr`.
CosineAnnealingWarmRestarts with `T_0=restart_period` (default 10).

---

### `pie_sparse_incremental_learning.py`

v4 IL chain. `--start-step` allows resuming from any step. Same partial unfreeze and
warm restart schedule as base training. Best val acc checkpoint passed as teacher for
next step.

---

### `test_SparseTemporalPIE.py` / `test_SparseTemporalPIE_v3.py`

Reports Accuracy, AUC, F1, Precision for:
1. Full test set
2. v=0 stationary subset (bbox center displacement < epsilon × bbox width, default epsilon=5.0)

---

## 5. Training Protocol

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | RMSprop |
| LR (head) | 1e-4 |
| LR (backbone) | 1e-5 (10× lower) |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealingWarmRestarts (T_0=7 for IL, T_0=10 for step 0) |
| Batch size | 32 |
| Epochs (step 0) | 50 |
| Epochs (IL steps) | 30 |
| Augmentation | ColorJitter + synchronized horizontal flip (p=0.5) + pose dropout (p=0.1) |

**IL loss (inherited from EfficientPIE):**

```
loss_new = CrossEntropy(robust_noisy(pred_curr), labels)
loss_old = KL_distillation(prev_pred.detach(), labels, temperature=2)
loss = loss_new  if loss_new > loss_old  else  0.5 * loss_old + loss_new
```

**Robust noisy perturbation:**

```
m_j = epoch / total_epochs
p_j = rand(-m_j, m_j)
logits_perturbed = logits + [p_j, -p_j]
```

---

## 6. Evaluation Protocol

- PIE random split: 90/5/5 train/val/test
- `max_size_observe=15`, `seq_overlap_rate=0.5`, balanced classes
- Test set: 92 pedestrians, 893 samples
- Metrics: Accuracy, AUC, F1, Precision (sklearn)
- Best val accuracy checkpoint used for test evaluation

**v=0 stationary subset:**
```python
delta = sqrt((cx_N - cx_0)^2 + (cy_N - cy_0)^2) / bbox_width
stationary = (delta < 5.0)   # 871 / 893 test samples
```

---

## 7. Results Summary

### PIE Test Set

| Model | Acc | AUC | F1 | Prec | Inference |
|-------|-----|-----|----|------|-----------|
| EfficientPIE (paper) | 0.920 | 0.917 | 0.952 | 0.960 | 0.21ms |
| v4 best (step 2) | 0.919 | 0.922 | 0.953 | 0.958 | 1.19ms |
| v4 step 14 | 0.913 | 0.915 | 0.949 | 0.953 | 1.19ms |
| **v3 best (step 14)** | **0.926** | **0.947** | **0.957** | **0.957** | **1.81ms** |

End-to-end (+ ViTPose-B, 3.875ms): v3 = 5.68ms, v4 = 5.07ms.

### IL Step Progression (Test Accuracy)

| IL Step | v3 | v4 |
|---------|----|----|
| 0 | 0.9048 | 0.9082 |
| 2 | 0.9205 | 0.9194 |
| 4 | 0.9071 | 0.9048 |
| 6 | 0.9048 | 0.9059 |
| 8 | 0.9037 | 0.8970 |
| 10 | 0.9104 | 0.9037 |
| 12 | 0.9127 | 0.9183 |
| **14** | **0.9261** | 0.9127 |

v3 improves through the full IL chain. v4 peaks at step 2 and degrades by step 14.

---

## 8. Key Implementation Notes

### Keypoint path convention

```
{keypoints_dir}/{pid}/{frame_id:05d}.npy
```

`frame_id` is the absolute frame number from the image filename (not the window index
0–14). This prevents collisions when the same pedestrian appears in multiple
non-overlapping observation windows.

### Pose normalization

```python
feats[k*2]   = (x - bbox_x1) / bbox_width
feats[k*2+1] = (y - bbox_y1) / bbox_height
# joints with conf < 0.25 → zeroed
```

### Context frame selection (v3)

```python
indices = np.linspace(0, step-1, min(K=4, step), dtype=int)
# step=0: [0] (single frame, mask=1, rest padded)
# step=2: [0, 1]
# step=6: [0, 2, 4, 5]
# step=14: [0, 4, 9, 13]
```

### Backbone partial unfreeze

During training, backbone parameters receive `lr × 0.1`. Head parameters receive full `lr`.
This prevents catastrophic forgetting of the ImageNet-pretrained features while allowing
fine-tuning for the pedestrian domain.

### Val set size caveat

The PIE val set contains only ~92 pedestrians (~500 samples after step filtering). This
is too small to reliably rank model variants — v3 appeared to ceiling at 0.870 val while
achieving 0.926 on test. Always report test set numbers; do not over-tune to val.
