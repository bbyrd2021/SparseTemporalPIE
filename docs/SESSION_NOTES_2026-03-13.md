# Session Notes — 2026-03-13

---

## Visual-Only Baseline Run (v1) — Terminated

The first SparseTemporalPIE run (launched 2026-03-11) was terminated during IL step 10. Pose features were all-zeros throughout due to the path bug fixed on 2026-03-12, making this a visual-only baseline.

### Final results (visual-only, frozen backbone)

| Step | Best Val Acc |
|------|-------------|
| Step 0 | 0.8608 |
| IL Step 2 | 0.8567 |
| IL Step 4 | 0.8588 |
| **IL Step 6** | **0.8649** ← run best |
| IL Step 8 | 0.8588 |
| Steps 10–14 | not completed (terminated) |

**Conclusion:** val accuracy plateaued at ~0.86 with no meaningful gains across IL steps. Frozen backbone + zero pose = insufficient signal for cross-attention to learn meaningful temporal reasoning.

### Preserved artifacts

- Old weights: `weights_sparse_v1_baseline/`
- Old logs: `training_logs_sparse_v1_baseline/`
- Named baseline checkpoint: `weights_sparse_v1_baseline/baseline_visual_only_acc0.8649.pth`

---

## New Architecture Run (v2) — Launched

Killed v1 and launched v2 with all 2026-03-12 changes:

```bash
bash run_sparse_pie_pipeline.sh   # started ~11:36 2026-03-13
```

### What's different from v1

| Change | v1 (baseline) | v2 (new) |
|---|---|---|
| Pose features | All-zeros (path bug) | Real keypoints, both frames |
| Pose fusion | Concat at classifier (1314-d) | Projected + added before attention (1280-d) |
| pose_proj bias | N/A | `bias=False` (zero pose = null) |
| Backbone | Fully frozen | Partially unfrozen @ lr=1e-5 |
| Horizontal flip | Independent (image only) | Synchronized (image + pose, COCO L/R swap) |
| Pose dropout | N/A | 10% (robustness to missing pose) |

### Expected outputs

- Weights: `weights_sparse/best_sparse_step{N}.pth`
- Logs: `training_logs_sparse/step0.log`, `training_logs_sparse/il_steps.log`
- Final eval: `training_logs_sparse/evaluation.log`

### Timeline estimate

- Step 0: 50 epochs (~8–10 hours)
- IL steps 2–14: 30 epochs each × 7 steps (~3–4 hours per step, ~24 hours total)
- Final eval: ~5 min
- **Total: ~32–34 hours**
