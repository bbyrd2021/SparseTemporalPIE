# Session Notes — 2026-03-27

## Summary

Completed JAAD training pipeline for SparseTemporalPIE v3, ran test evaluation, performed cross-dataset AUC analysis, and updated documentation for publication.

---

## JAAD IL Step Completion

All 8 IL steps (0, 2, 4, 6, 8, 10, 12, 14) completed for SparseTemporalPIE v3 trained on JAAD. Best val accuracy per step:

| IL Step | Best Val Acc |
|---------|-------------|
| 0       | 0.8801       |
| 2       | 0.8966       |
| 4       | 0.9031       |
| 6       | 0.9108       |
| 8       | 0.9119       |
| 10      | 0.9168       |
| 12      | 0.9168       |
| 14      | 0.9228       |

Notable: unlike PIE, JAAD val accuracy improves monotonically across all 8 steps — no mid-chain dip. Step 14 weights used for test evaluation.

---

## JAAD Test Evaluation

Evaluated `weights_sparse_v3_jaad/best_sparse_v3_jaad_step14.pth` on JAAD test set (1,876 samples):

| Metric    | SparseTemporalPIE v3 | EfficientPIE (paper) | Delta |
|-----------|----------------------|----------------------|-------|
| Accuracy  | 0.8785               | **0.890**            | −0.012 |
| AUC       | **0.8850**           | 0.860                | **+0.025** |
| F1        | **0.6334**           | 0.620                | **+0.013** |
| Precision | 0.5970               | **0.630**            | −0.033 |

Key finding: the same accuracy-vs-AUC tradeoff observed on PIE holds on JAAD. AUC and F1 improve; accuracy and precision are below. This is a consistent cross-dataset pattern, not an artifact of a single dataset.

---

## Cross-Dataset AUC Pattern

Across both datasets, SparseTemporalPIE v3 consistently improves AUC over EfficientPIE:

| Dataset | EfficientPIE AUC | v3 AUC | Delta |
|---------|-----------------|--------|-------|
| PIE     | 0.917           | 0.947  | +0.030 |
| JAAD    | 0.860           | 0.885  | +0.025 |

This is the core paper narrative: **cross-attention produces better-calibrated risk scores across datasets**. The model may sacrifice argmax accuracy in edge cases, but its soft probability outputs are more reliable for downstream planners operating at varying thresholds.

---

## Zero-Shot JAAD Generalization (Negative Result)

Before JAAD training was complete, evaluated all three PIE-trained models on JAAD with no JAAD training. Results were near or below random:

| Model | JAAD Acc | JAAD AUC |
|-------|----------|----------|
| v3 (EfficientPIE backbone) | 0.304 | 0.487 |
| v3 (ImageNet backbone) | 0.182 | 0.474 |
| v4 | 0.286 | 0.453 |

AUC < 0.5 indicates predictions are anti-correlated with JAAD labels. Root causes:

- **Label semantics mismatch:** PIE labels encode crossing *intention* (`intention_binary`); JAAD labels encode crossing *action* (`intent`). These are different tasks encoded as the same binary.
- **Domain shift:** Different cities, cameras, and conditions.
- **Class imbalance:** JAAD is skewed toward non-crossers; PIE is balanced.

Conclusion: zero-shot JAAD transfer is not meaningful and should not be reported. Both EfficientPIE and our model require dedicated JAAD training, which is the standard protocol.

---

## Documentation Updated

- `docs/RESULTS.md`: Added Section 3.2 (JAAD SOTA comparison), Section 5.2 (JAAD IL step progression), updated Section 1 (datasets), updated Section 6 (Discussion) with cross-dataset AUC finding and JAAD limitation note.
- `docs/SESSION_NOTES_2026-03-27.md`: this file.

---

## Publication Readiness Checklist Update

### Results

- [x] PIE test evaluation — v3 EfficientPIE backbone (0.9261 acc, 0.9468 AUC)
- [x] PIE test evaluation — v3 ImageNet backbone (0.9216 acc, 0.9211 AUC)
- [x] PIE ablation — v3 vs v4 (cross-attention contribution)
- [x] PIE ablation — backbone initialization (EfficientPIE vs ImageNet)
- [x] PIE IL step progression (steps 0–14, all 8 evaluated)
- [x] JAAD test evaluation — v3 EfficientPIE backbone (0.8785 acc, 0.8850 AUC)
- [x] JAAD IL step progression (steps 0–14, all 8 complete)
- [ ] Inference time benchmark — proper protocol (warm GPU, batch=128, 100 runs, matching EfficientPIE)

### Code & Reproducibility

- [x] Training scripts — PIE step 0 + IL steps 2–14
- [x] Training scripts — JAAD step 0 + IL steps 2–14
- [x] Eval scripts — PIE (v3, v4)
- [x] Eval scripts — JAAD (v3, v4)
- [x] Keypoint extraction script (PIE + JAAD, ViTPose-B)
- [ ] README with full reproduction instructions
- [ ] Requirements / environment spec
- [ ] Model weights released (arXiv supplement or HuggingFace)

### Paper

- [ ] Abstract
- [ ] Introduction
- [ ] Related work
- [ ] Methodology (architecture diagram, equations)
- [ ] Experiments section (PIE + JAAD results tables)
- [ ] Ablation section
- [ ] Discussion / limitations
- [ ] Conclusion
- [ ] References
- [ ] Figures (architecture diagram, IL step progression plot, attention visualization)

### Submission

- [ ] arXiv preprint — unblocked, all results complete
- [ ] BMVC 2026 abstract — due May 22, 2026
- [ ] BMVC 2026 full paper — due May 29, 2026

---

## Notes

- All JAAD results use `output_type: ['intent']` (not `'intention_binary'` which is PIE-specific).
- JAAD has no OBD speed data; ego-vehicle speed features are zero-padded. This may account for part of the accuracy deficit vs. EfficientPIE.
- The only remaining experiment blocker for arXiv submission is the inference time benchmark.
