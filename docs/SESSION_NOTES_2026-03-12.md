# Session Notes — 2026-03-12

---

## Ongoing Run Status (old architecture, visual-only baseline)

The pipeline launched 2026-03-11 (`bash run_sparse_pie_pipeline.sh`) completed step 0 and IL steps 2–4, and was in IL step 6 (~epoch 9/30) as of this session. Val accuracy was flat across all completed steps:

| Step | Best Val Acc |
|------|-------------|
| Step 0 (base, 50 epochs) | 0.8608 |
| IL Step 2 | 0.8567 |
| IL Step 4 | 0.8588 |
| IL Step 6 | 0.8618 (epoch 1, then stalled) |

**Root cause of plateau:** backbone was fully frozen and pose features were all-zeros (path bug — see below). The model had no actual pose signal and cross-attention was operating on embeddings optimized for single-frame classification, not cross-frame comparison.

This run's final result will serve as the **visual-only baseline**.

---

## Bugs Fixed

### Bug: Pose keypoints silently all-zeros throughout training

`SparseDataset._load_pose_feats()` was loading using the IL step index as the filename:
```python
kpts_path = os.path.join(self.keypoints_dir, pid, f"{t:05d}.npy")
# t = self.step = 0, 2, 4, 6... → tried "00000.npy", "00002.npy"
# Actual files: "01018.npy", "01019.npy" (absolute frame numbers)
# → file never found → returned zeros every time
```

Fix: extract absolute frame number from the image path filename:
```python
frame_id  = int(os.path.splitext(os.path.basename(frame_path))[0])
kpts_path = os.path.join(self.keypoints_dir, pid, f"{frame_id:05d}.npy")
```

**Impact:** the entire first training run (step 0 through IL step 14) trained with zero pose. Val acc 0.86 is a visual-only result. Pose hit rate after fix: ~98–100% on val set.

---

## Architecture Changes (2026-03-12)

### 1. Pose for both frames, fused before cross-attention

**Before:** pose_feats (34-dim, f_current only) concatenated at classifier input → 1314-dim input.

**After:** normalized keypoints extracted for **both** f_current and f*, projected to 1280-dim and added to backbone embeddings **before** cross-attention:

```python
self.pose_proj = nn.Linear(34, embed_dim, bias=False)  # bias=False: zero pose → zero contribution

emb_current = backbone(f_current) + pose_proj(pose_current)  # (B, 1280)
emb_fstar   = backbone(f_star)   + pose_proj(pose_fstar)    # (B, 1280)
# cross-attention now compares visual+pose jointly between frames
```

Classifier input shrinks from 1314 → 1280 (no more end-concatenation).

**Rationale:** cross-attention can now attend to joint changes ("torso lean changed", "head turned") directly in the attention weights, rather than the classifier seeing pose only after the temporal reasoning has already happened.

`bias=False` ensures `pose_proj(zeros) = 0`, so zero/missing pose adds no spurious signal — the model falls through to visual-only embedding cleanly.

### Architecture diagram

```
                    ┌─────────────────────────────────────────────────────┐
                    │              SparseTemporalPIE v2                   │
                    └─────────────────────────────────────────────────────┘

  f_current (B,3,300,300)                          f* (B,3,300,300)
         │                                                │
         ▼                                                ▼
  ┌──────────────┐                                ┌──────────────┐
  │   Backbone   │◄── shared weights (lr=1e-5) ──►│   Backbone   │
  │ EfficientPIE │                                │ EfficientPIE │
  └──────┬───────┘                                └──────┬───────┘
         │ (B, 1280)                                     │ (B, 1280)
         │                                               │
         │  ┌────────────┐                               │  ┌────────────┐
         ├──┤ pose_proj  │◄─ pose_current (B, 34)        ├──┤ pose_proj  │◄─ pose_fstar (B, 34)
         │  │ 34→1280    │   10% dropout to zeros         │  │ 34→1280    │   (shared weights)
         │  │ bias=False │                                │  │ bias=False │
         │  └─────┬──────┘                                │  └─────┬──────┘
         │        │                                       │        │
         ▼        ▼                                       ▼        ▼
       ( + )  element-wise add                          ( + )  element-wise add
         │                                                │
         ▼                                                ▼
    emb_current                                       emb_fstar
     (B, 1280)                                        (B, 1280)
         │                                                │
         │         ┌──────────────────────────┐           │
         ├────────►│    Cross-Attention        │◄─────────┤
         │    Q    │  MultiheadAttention       │   K, V   │
         │         │  8 heads, embed=1280      │          │
         │         └────────────┬─────────────┘           │
         │                      │ attn_out (B, 1280)      │
         │                      │                         │
         ▼                      ▼                         │
       ( + )  residual: emb_current + attn_out            │
         │                                                │
         ▼                                                │
    ┌─────────┐                                           │
    │LayerNorm│                                           │
    └────┬────┘                                           │
         │                                                │
         ├──────────────────────┐                         │
         │                      ▼                         │
         │               ┌────────────┐                   │
         │               │ FF Block   │                   │
         │               │ 1280→512   │                   │
         │               │ GELU       │                   │
         │               │ 512→1280   │                   │
         │               └─────┬──────┘                   │
         │                     │                          │
         ▼                     ▼                          │
       ( + )  residual                                    │
         │                                                │
         ▼                                                │
    ┌─────────┐                                           │
    │LayerNorm│                                           │
    └────┬────┘                                           │
         │                                                │
         ▼                                                │
    enriched (B, 1280)                                    │
         │                                                │
         ▼
  ┌──────────────┐
  │  Classifier  │
  │  1280 → 256  │
  │  ReLU        │
  │  Dropout     │
  │  256 → 2     │
  └──────┬───────┘
         │
         ▼
    logits (B, 2)
    [not_crossing, crossing]
```

**Data flow summary:**
1. Both frames pass through the **shared Siamese backbone** (partially unfrozen, lr=1e-5)
2. Pose keypoints (34-dim, 17 joints × xy) are projected to 1280-dim and **added** to backbone embeddings
3. Cross-attention compares visual+pose embeddings: Q = f_current, K/V = f*
4. Residual + LayerNorm + FF + LayerNorm produces the enriched representation
5. Classifier outputs crossing/not-crossing logits

**f\* selection** (computed in SparseDataset, not shown): argmax cosine distance between f_current and all prior frames [0, step-1] in the backbone embedding space.

### Architecture summary (updated)

| Component | Value |
|---|---|
| Backbone | EfficientPIE CNN (FusedMBConv + MBConv stack), outputs 1280-dim |
| pose_proj | Linear(34 → 1280, bias=False), shared for both frames |
| Cross-attention | MultiheadAttention(embed_dim=1280, num_heads=8), Q=emb_current, K/V=emb_fstar |
| Feedforward | Linear(1280→512) → GELU → Dropout → Linear(512→1280) |
| Classifier | Linear(1280→256) → ReLU → Dropout → Linear(256→2) |
| Total params | 8,933,778 |
| Trainable params | 8,933,778 (all, incl. backbone — see training strategy) |
| FLOPs | 3,641M (1,821M MACs) |
| Inference time | ~2.4 ms/sample (GPU, batch=1) |

---

## Training Strategy Changes (2026-03-12)

### 1. Partial backbone unfreeze with differential learning rates

**Before:** backbone fully frozen (0 gradient flow through 684K params).

**After:** backbone unfrozen, trained at 10× lower LR via two optimizer param groups:

```python
optimizer = optim.RMSprop([
    {'params': backbone_params, 'lr': lr * 0.1},   # 1e-5
    {'params': other_params,    'lr': lr},           # 1e-4
], weight_decay=1e-4)
```

**Rationale:** backbone embeddings were optimized for single-frame EfficientPIE classification. Cross-attention needs embeddings that encode visually meaningful temporal change. Low LR preserves pretrained features while allowing gradual adaptation. Both LRs decay together via CosineAnnealingLR (backbone: 1e-5 → 1e-7, rest: 1e-4 → 1e-7).

### 2. Pose dropout

10% of training samples have both pose vectors zeroed together:

```python
if random.random() < self.pose_dropout_p:   # pose_dropout_p=0.1
    pose_current = np.zeros(34, dtype=np.float32)
    pose_fstar   = np.zeros(34, dtype=np.float32)
```

Forces the model to learn a viable visual-only path. At inference, if pose is unavailable, zero input + `bias=False` in `pose_proj` means the model degrades gracefully to visual-only mode rather than getting a spurious learned-bias offset.

### 3. Synchronized horizontal flip

`RandomHorizontalFlip` removed from torchvision transform pipeline. Flip now applied manually in `SparseDataset.__getitem__` so images and poses are flipped together:

```python
if random.random() < self.flip_p:   # flip_p=0.5
    f_current_img = f_current_img.transpose(Image.FLIP_LEFT_RIGHT)
    f_star_img    = f_star_img.transpose(Image.FLIP_LEFT_RIGHT)
    pose_current  = flip_pose(pose_current)
    pose_fstar    = flip_pose(pose_fstar)
```

`flip_pose()` mirrors x-coordinates (`x = 1.0 - x`) and swaps all COCO left/right joint pairs (1↔2, 3↔4, 5↔6, 7↔8, 9↔10, 11↔12, 13↔14, 15↔16). Zeroed joints (low confidence) remain zero after flip. Function is verified idempotent (double flip = identity).

### Hyperparameters (next run)

| Parameter | Step 0 | IL Steps 2–14 |
|---|---|---|
| Epochs | 50 | 30 per step |
| Batch size | 32 | 32 |
| LR (backbone) | 1e-5 | 1e-5 |
| LR (other) | 1e-4 | 1e-4 |
| LR schedule | CosineAnnealing (eta_min=1e-7) | CosineAnnealing (eta_min=1e-7) |
| Weight decay | 1e-4 | 1e-4 |
| Optimizer | RMSprop | RMSprop |
| flip_p | 0.5 | 0.5 |
| pose_dropout_p | 0.1 | 0.1 |
| ColorJitter | brightness/contrast/saturation=0.5, hue=0.1 | same |

---

## Files Modified (2026-03-12)

| File | Changes |
|---|---|
| `models/SparseTemporalPIE.py` | Added `pose_proj = Linear(34→1280, bias=False)`; updated `forward(f_current, f_star, pose_current, pose_fstar)`; classifier input 1314→1280 |
| `utils/sparse_dataset.py` | Fixed `_load_pose_feats` path (absolute frame number); added pose loading for f*; added `flip_pose()` + `_FLIP_PAIRS`; added `flip_p` and `pose_dropout_p` params; pose dropout applied before flip in `__getitem__`; collate_fn updated to 5-tuple |
| `utils/train_val.py` | Updated `evaluate_sparse` and `incremental_learning_train_sparse` to unpack 5-tuple batches and pass 4 args to model |
| `train_SparseTemporalPIE.py` | Removed `RandomHorizontalFlip` from transform; added `flip_p=0.5`, `pose_dropout_p=0.1`; partial backbone unfreeze with differential LR optimizer |
| `pie_sparse_incremental_learning.py` | Same augmentation changes; removed backbone freeze line; differential LR optimizer |
| `test_SparseTemporalPIE.py` | Updated eval loop to unpack 5-tuple and pass 4 args to model |

---

## Model Complexity (updated)

| Metric | Value |
|---|---|
| Total params | 8,933,778 |
| Backbone params | 684,176 (now trainable @ 1e-5) |
| Trainable (non-backbone) | 8,249,602 @ 1e-4 |
| MACs | 1,821M |
| FLOPs | 3,641M |
| Inference time (GPU, B=1) | ~2.4 ms |

For reference: EfficientPIE baseline = 684K params, 0.279 ms inference, Acc=0.92 on PIE.

---

## Next Steps

- Let current run finish (visual-only baseline) — expected final result ~0.86–0.87 Acc
- Launch new run with all 2026-03-12 changes
- If plateau persists beyond step 6 in new run → consider increasing epochs per IL step or reducing IL step stride (e.g., every 1 frame instead of every 2)
- Target: Acc ≥ 0.90 on PIE test set
