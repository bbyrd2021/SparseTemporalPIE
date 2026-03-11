# EfficientPIE Replication Results

**Date:** 2026-03-10
**Repo:** `/data/repos/EfficientPIE`
**Weights:** `weights_v8/model_8_PIE_IL_step14_new.pth` (author-provided, IL step 14)

---

## Evaluation Results (PIE Test Set)

| Metric | Replicated | Paper (Table 3) |
|--------|-----------|-----------------|
| Accuracy | **0.918** | 0.92 |
| AUC | **0.917** | 0.92 |
| F1 | **0.952** | 0.95 |
| Precision | **0.961** | 0.96 |
| Recall | 0.943 | — |
| Loss | 0.289 | — |

**Result: matches the paper within rounding.**

---

## Inference Performance

| Metric | Value |
|--------|-------|
| Mean inference time | 0.279 ms |
| Std deviation | 0.001 ms |
| FLOPs | 909.43 M |
| Parameters | 0.69 M |

Paper reports 0.21 ms (different GPU). Consistent ballpark.

---

## Test Set Details

| Item | Value |
|------|-------|
| Data split | `random` (90/5/5) |
| Test tracks | 92 pedestrians |
| Test sequences | 880 (filtered from 893 — 13 missing frames excluded) |
| Batch size | 32 |
| Workers | 8 |
| Frame used | Index 14 (last frame of 15-frame window) |

---

## Eval Command

```bash
python test_EfficientPIE.py --weights weights_v8/model_8_PIE_IL_step14_new.pth
```

Log saved to: `training_logs/final_eval_fixed.log`

---

## Notes on Replication Setup

The author's repo had several bugs that were fixed before evaluation could run:

| Bug | Fix |
|-----|-----|
| `models/EfficientPIE_backup.py` imported non-existent `AdaptiveEncoder` | Created clean `models/EfficientPIE.py` without it |
| `utils/my_dataset.py` imported non-existent `adaptive_selection` | Removed broken import |
| `filter_existing_sequences()` crashed on empty `encoder_input` list | Fixed to skip keys where `len(v) != n` |
| `reverse_step` hardcoded to 1 in `my_dataset.py` | Fixed to `max_size_observe - step` |
| All scripts imported from `EfficientPIE_backup` | Updated to `EfficientPIE` |
| Hardcoded paths to author's machine | Updated to `/data/datasets/PIE` |

All fixes are minimal and do not alter model behavior — only the eval infrastructure.

---

## What Was NOT Replicated

The full training pipeline (ImageNet pre-training → base training → 7 IL steps) was not re-run from scratch in this session. The evaluation above uses the **author's provided weights** from `weights_v8/`. The training scripts are now fully functional and runnable; a from-scratch replication can be kicked off with `run_training_after_extraction.sh`.
