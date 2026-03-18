# Session Notes — 2026-03-18

## Overview
Implemented SparseTemporalPIE v3, ran full IL pipeline, identified ceiling at ~0.868, diagnosed root cause, and redesigned to v4.

---

## SparseTemporalPIE v3 — Implementation & Results

### Architecture (v3)
- Multi-frame cross-attention: K=4 evenly-spaced context frames
- Pose: 68-d (34 static + 34 velocity)
- Bbox trajectory: 12-d (displacement, velocity, acceleration, size ratio)
- Context features: 5-d (speed, speed_mean, speed_valid, action, look)
- ctx_proj MLP: 17→128-d (late fusion)
- Classifier: 1280 + 128 = 1408 → 256 → 2
- Total params: ~9M

### pie_data.py changes
- `_get_intention()`: added extraction of `obd_speed`, `action`, `look` per pedestrian sequence
- `get_tracks()`: added passthrough for extra keys via loop
- `get_train_val_data()`: added passthrough for extra keys into result dict
- `filter_existing_sequences()` required no changes (already handles arbitrary keys generically)

### v3 Training Results
| Step | v2 best | v3 best | Notes |
|------|---------|---------|-------|
| 0 | 0.853 | 0.857 | Warm-up, bbox_traj=zeros |
| 2 | 0.855 | 0.868 | First real context frame |
| 4 | 0.868 | 0.862 | |
| 6 | 0.865 | 0.870 | Best v3 result |
| 8 | 0.851 | 0.869 | |
| 10 | 0.862 | 0.868 | Ceiling apparent |

### Experiments during v3 IL
1. **Original run (lr=1e-4, CosineAnnealingLR)**: peaked early (ep1-3), never recovered — classic single-decay overshoot
2. **Frozen backbone (lr=5e-5)**: stable but capped at 0.852 — too conservative
3. **Warm restarts (CosineAnnealingWarmRestarts, T_0=7, lr=1e-4)**: best results — 0.870 at step 6. LR resets every 7 epochs escape local minima. Step 8 also hit 0.869 twice (at restart boundaries ep3 and ep7)

### Root Cause Analysis
- v3 ceiling ~0.868-0.870 vs EfficientPIE 0.92
- Cross-attention on 1-4 near-identical frame embeddings ≈ scalar gate, not real attention
- Adjacent video frames are visually similar → backbone embeddings too close for attention to learn meaningful differences
- Temporal signal is better encoded in WHERE pedestrian is moving (bbox_traj) and behavior (speed/action/look), not visual frame differences
- IL mechanism fights cross-attention optimization — IL distillation + transformer training jointly is a harder optimization problem

---

## SparseTemporalPIE v4 — Architecture Redesign

### Decision
Drop cross-attention entirely. Single frame pass + context features for temporal information.

### Architecture (v4)
```
backbone(f_current) → 1280-d
+ pose_proj(pose_34d) → fused 1280-d
+ ctx_proj(bbox_traj_12d, ctx_feats_5d) → 128-d  [late fusion]
→ classifier(1408 → 256 → 2)
```
- Pose: back to 34-d static only (velocity dropped — low signal, consecutive frames similar)
- No context frames, no cross-attention, no feedforward block
- Total params: ~1.1M (vs 9M v3, vs ~684K EfficientPIE backbone alone)
- Single backbone pass per sample (same speed as EfficientPIE)
- Warm restarts kept: T_0=10, lr=1e-4

### Files Modified (v4)
- `models/SparseTemporalPIE.py`: complete rewrite, v4 architecture
- `utils/sparse_dataset.py`: simplified to 5-tuple (removed f_context, context_mask, pose_context, pose_velocity)
- `utils/train_val.py`: updated evaluate_sparse and incremental_learning_train_sparse for 4-input forward
- `train_SparseTemporalPIE.py`: updated train loop, added --restart-period arg, output-dir=weights_sparse_v4
- `pie_sparse_incremental_learning.py`: updated IL loop, added --start-step and --restart-period args
- `test_SparseTemporalPIE.py`: updated eval loop

### Rationale
- EfficientPIE achieves 0.92 with the same backbone and zero temporal fusion
- The IL distillation chain is the key mechanism, not cross-frame attention
- Adding bbox_traj + ctx_feats gives explicit motion and behavioral context that EfficientPIE lacks
- Simpler architecture = cleaner IL gradient flow

### v4 Training Status
- Step 0: in progress (cuda:1), batch_size=32, ~538 batches/epoch
- Weights output: `weights_sparse_v4/`
- Log: `training_logs/sparse_v4_step0.log`

---

## v3 IL Run Status (warm restart run, for reference)
- Steps 6, 8 done; step 10 in progress (15/30 epochs, best=0.868)
- Log: `training_logs/sparse_v3_IL_v3.log`
- Weights: `weights_sparse_v3/`
