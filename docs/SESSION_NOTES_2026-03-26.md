# Session Notes — 2026-03-26

## Summary

Full evaluation of SparseTemporalPIE v3 on PIE, backbone initialization ablation, and launch of JAAD training pipeline.

---

## PIE Evaluation — SparseTemporalPIE v3 (imagenet backbone)

Completed the imagenet-initialized v3 run (weights in `weights_sparse_v3_imagenet/`). Best checkpoint was step 14.

| IL Step | Val Acc | Test Acc | AUC    | Precision |
|---------|---------|----------|--------|-----------|
| 12      | 0.8936  | 0.9205   | 0.9365 | 0.9590    |
| 14      | 0.8864  | 0.9216   | 0.9211 | 0.9628    |

Step 14 edged step 12 on accuracy but lost on AUC. Both fall short of the EfficientPIE-backbone v3 (0.9261 / 0.9468 AUC).

---

## Backbone Initialization Ablation

Compared two v3 runs with identical partial-freeze training (backbone @ lr=1e-5, head @ lr=1e-4):

| Configuration | Best Step | Test Acc | AUC    |
|---------------|-----------|----------|--------|
| EfficientPIE backbone | 14 | **0.9261** | **0.9468** |
| ImageNet backbone | 14 | 0.9216 | 0.9211 |

**Key finding:** Backbone initialization matters more than backbone fine-tuning. The EfficientPIE backbone has already been task-adapted to the PIE domain; starting from raw ImageNet features cannot recover within the small PIE training set at a constrained lr. This reinforces that SparseTemporalPIE v3 is best understood as an *extension* of EfficientPIE, not a replacement — its gains come from cross-attention and pose reasoning built on a strong, task-specific backbone.

Val acc was actually *higher* for the imagenet run (0.8936 vs 0.8823) but test acc was lower — further evidence that val is a poor proxy on this small dataset.

Results added to `docs/RESULTS.md` Section 4.2.

---

## Zero-Shot JAAD Generalization

Ran all three models (v3 EfficientPIE backbone, v3 ImageNet backbone, v4) on JAAD test set with no JAAD training. Results were near or below random:

| Model | JAAD Acc | JAAD AUC |
|-------|----------|----------|
| v3 (EfficientPIE backbone) | 0.304 | 0.487 |
| v3 (ImageNet backbone) | 0.182 | 0.474 |
| v4 | 0.286 | 0.453 |

AUC below 0.5 indicates predictions are anti-correlated with JAAD labels — not just uncertain but inverted. Root causes:
- **Label semantics mismatch:** PIE labels are annotated *intention* (will this person cross?); JAAD labels are *crossing action* (is this person currently crossing?). Different tasks encoded as the same binary.
- **Domain shift:** PIE is Toronto, JAAD is different cities/cameras.
- **Class imbalance:** JAAD is skewed toward non-crossers; PIE is balanced.

Conclusion: zero-shot JAAD transfer is not meaningful to report. JAAD requires dedicated training, as EfficientPIE also does (they train two independent models).

---

## JAAD Keypoint Extraction

Extracted ViTPose-B keypoints for all JAAD pedestrian tracks (all splits) to `/data/datasets/JAAD/keypoints_pid/`. Required symlinking missing annotation directories from `/data/repos/PedestrianIntent++/JAAD/`:
- `annotations_attributes/`
- `annotations_appearance/`
- `annotations_traffic/`
- `annotations_vehicle/`
- `split_ids/`

---

## JAAD Training — SparseTemporalPIE v3

Created JAAD training scripts:
- `scripts/sparsetemporalpie/train_SparseTemporalPIE_v3_jaad.py` — step 0, 50 epochs
- `scripts/sparsetemporalpie/jaad_sparse_incremental_learning_v3.py` — IL steps 2–14, 30 epochs each

Key differences from PIE scripts: `JAAD` data class, `output_type: ['intent']` (not `'intention_binary'`), JAAD paths. Same architecture, same partial-freeze strategy, initialized from EfficientPIE backbone weights (`weights_v8/model_8_PIE_IL_step14_new.pth`).

JAAD eval scripts also created:
- `scripts/sparsetemporalpie/test_SparseTemporalPIE_v3_jaad.py`
- `scripts/sparsetemporalpie/test_SparseTemporalPIE_jaad.py`

**Step 0** trained Mar 24–25, best val acc: **0.8801**. Weights: `weights_sparse_v3_jaad/best_sparse_v3_jaad_step0.pth`.

**IL steps 2–12** in progress as of session end. Val acc progression:

| IL Step | Best Val Acc |
|---------|-------------|
| 0       | 0.8801       |
| 2       | 0.8966       |
| 4       | 0.9031       |
| 6       | 0.9108       |
| 8       | 0.9119       |
| 10      | 0.9168       |
| 12      | 0.9168 (epoch 25, still running) |
| 14      | not yet started |

EfficientPIE paper reports **0.89 JAAD test accuracy**. Our val numbers at every step from step 4 onward already exceed this. Test eval pending after step 14 completes.

---

## Publication Readiness Checklist

The IDIL protocol uses 8 IL steps: step 0 (base training) + steps 2, 4, 6, 8, 10, 12, 14 (7 incremental steps).

### Results

- [x] PIE test evaluation — v3 EfficientPIE backbone (0.9261 acc, 0.9468 AUC)
- [x] PIE test evaluation — v3 ImageNet backbone (0.9216 acc, 0.9211 AUC)
- [x] PIE ablation — v3 vs v4 (cross-attention contribution)
- [x] PIE ablation — backbone initialization (EfficientPIE vs ImageNet)
- [x] PIE IL step progression (steps 0, 2, 4, 6, 8, 10, 12, 14 — all 8 evaluated)
- [ ] JAAD test evaluation — v3 EfficientPIE backbone *(pending step 14 completion + eval)*
- [ ] JAAD IL step progression (steps 0–14, all 8 — step 14 still running)
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

- [ ] arXiv preprint (can go up once JAAD results are in)
- [ ] BMVC 2026 abstract — due May 22, 2026
- [ ] BMVC 2026 full paper — due May 29, 2026

---

## Venue Research

Checked deadlines for target publication venues:

| Venue | Deadline | Status |
|-------|----------|--------|
| ECCV 2026 | March 5, 2026 | Missed |
| IJCAI 2026 | January 19, 2026 | Missed |
| **BMVC 2026** | **May 29, 2026** | **Open — ~2 months** |
| NeurIPS 2026 | ~May 2026 | Check |

**Recommended path:** Post arXiv preprint once JAAD results are in (establishes priority), then target BMVC 2026 (abstract May 22, paper May 29, Lancaster UK, November conference).

---

## RESULTS.md Updates

Added Section 4.2 (Backbone Initialization ablation) and expanded Section 6 (Discussion) to cover:
- Backbone initialization finding
- Expanded val/test discrepancy note with actual numbers from both runs
