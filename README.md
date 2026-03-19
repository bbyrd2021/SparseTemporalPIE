# SparseTemporalPIE

> Fork of [EfficientPIE](https://github.com/heinideyibadiaole/EfficientPIE) (IJCAI-25)
> Extended with multi-frame cross-attention, pose velocity, and explicit motion/behavioral context.
> Authors: Brandon Byrd, Abel Abebe Bzuayene — xDI Lab, NC A&T State University

---

## Overview

**EfficientPIE** predicts pedestrian crossing intention from a single image crop using an
EfficientNet-inspired backbone with no temporal modeling. It achieves 0.92 accuracy at
0.69M parameters and sub-millisecond inference.

**SparseTemporalPIE** extends EfficientPIE with three additional information streams:

- **Pose features** — ViTPose-B keypoints (34-d static + 34-d velocity) fused into the backbone embedding
- **Multi-frame cross-attention** — up to K=4 evenly-spaced context frames attend to the current frame
- **Motion and behavioral context** — bbox trajectory statistics (12-d) + ego-vehicle speed, pedestrian action/look (5-d) via late-fusion MLP

Two variants were trained and evaluated:

| Variant | Architecture | Params | Inference | Best Accuracy | Best AUC |
|---------|-------------|--------|-----------|---------------|----------|
| **v3** | Cross-attention + pose velocity + ctx MLP | 9.0M | 1.81ms | **0.9261** | **0.9468** |
| v4 | No attention, static pose + ctx MLP only | 1.1M | 1.19ms | 0.9194 | 0.9220 |
| EfficientPIE (paper) | Single frame, visual only | 0.69M | 0.21ms | 0.920 | 0.917 |

v3 establishes new state-of-the-art AUC on the PIE test set. End-to-end inference including
upstream ViTPose-B pose estimation: v3 = 5.68ms, v4 = 5.07ms (both real-time at 30fps).

See [`docs/RESULTS.md`](docs/RESULTS.md) for full SOTA comparison.

---

## Architecture (v3)

```
f_current ──► backbone ──► emb (1280-d) ◄── pose_proj(pose_current, 68-d)
                                │
f_context[0..K] ► backbone ► K context embs ◄── pose_proj(pose_context, K×68-d)
                                │
                      cross_attn(Q=emb, K/V=context, K=4)
                                │
                         attn_norm + FF(1280→512→1280) + ff_norm
                                │  (enriched, 1280-d)
bbox_traj (12-d) ──┐
ctx_feats  (5-d) ──┴──► ctx_proj MLP ──► ctx (128-d)
                                │
                    classifier(1408 → 256 → 2)
```

---

## Results (PIE Test Set)

| Metric    | EfficientPIE (paper) | v4 (ours) | v3 (ours) |
|-----------|---------------------|-----------|-----------|
| Accuracy  | 0.920               | 0.919     | **0.926** |
| AUC       | 0.917               | 0.922     | **0.947** |
| F1        | 0.952               | 0.953     | **0.957** |
| Precision | 0.960               | 0.958     | **0.957** |
| Inference | 0.21ms              | 1.19ms    | 1.81ms    |

See [`docs/RESULTS.md`](docs/RESULTS.md) for full SOTA comparison table with 14 methods.

---

## Repository Structure

```
models/
  EfficientPIE.py                        # baseline model (unchanged)
  SparseTemporalPIE.py                   # v4: single frame + ctx MLP
  SparseTemporalPIE_v3.py                # v3: multi-frame cross-attention

utils/
  pie_data.py / jaad_data.py             # dataset APIs
  my_dataset.py                          # EfficientPIE dataset loader
  sparse_dataset.py                      # v4 dataset — 5-tuple
  sparse_dataset_v3.py                   # v3 dataset — 8-tuple
  train_val.py                           # training/eval loops

scripts/
  preprocess/
    extract_frames.py                    # video → image frames (run once)
    extract_keypoints.py                 # ViTPose-B keypoint extraction (run once)
    pretrain_imagenet.py                 # ImageNet pre-training
  efficientpie/
    train_EfficientPIE.py                # PIE base training
    pie_domain_incremental_learning.py   # PIE IL steps 2→14
    test_EfficientPIE.py                 # PIE evaluation
    train_EfficientPIE_JAAD.py           # JAAD base training
    jaad_domain_incremental_learning.py  # JAAD IL steps 2→14
    test_EfficientPIE_JAAD.py            # JAAD evaluation
  sparsetemporalpie/
    train_SparseTemporalPIE.py           # v4 base training (step 0)
    pie_sparse_incremental_learning.py   # v4 IL steps 2→14
    test_SparseTemporalPIE.py            # v4 evaluation + v=0 subset
    test_SparseTemporalPIE_v3.py         # v3 evaluation + v=0 subset
  ablation/
    calibrate_change_detector.py         # ChangeDetector ablation tool

pipelines/
  run_pie_pipeline.sh                    # full EfficientPIE PIE pipeline
  run_jaad_pipeline.sh                   # full EfficientPIE JAAD pipeline
  run_sparse_pie_pipeline.sh             # full SparseTemporalPIE pipeline
  run_training_after_extraction.sh       # wait for extraction then train

weights_sparse_v3/                       # v3 IL checkpoints (steps 0–14)
weights_sparse_v4/                       # v4 IL checkpoints (steps 0–14)

docs/
  RESULTS.md                             # full results and SOTA comparison
  SPARSE_TEMPORAL_PIE.md                 # architecture and implementation guide
  SESSION_NOTES_*.md                     # development session logs
```

> All scripts are run from the repo root, e.g. `python scripts/sparsetemporalpie/train_SparseTemporalPIE.py`

---

## Installation

```bash
pip install -r requirements.txt
```

**Dataset setup:**

```bash
# Annotations (symlink from PedestrianIntent++)
ln -s /path/to/PedestrianIntent++/PIE/PIE/annotations/annotations /data/datasets/PIE/annotations
ln -s /path/to/PedestrianIntent++/PIE/PIE/annotations/annotations_attributes /data/datasets/PIE/annotations_attributes
ln -s /path/to/PedestrianIntent++/PIE/PIE/annotations/annotations_vehicle /data/datasets/PIE/annotations_vehicle
ln -s /path/to/PedestrianIntent++/JAAD/annotations /data/datasets/JAAD/annotations

# PIE clip layout
mkdir /data/datasets/PIE/PIE_clips
for i in 01 02 03 04 05 06; do
  ln -s /data/datasets/PIE/set$i /data/datasets/PIE/PIE_clips/set$i
done
```

---

## Usage

### SparseTemporalPIE (v3 — best results)

```bash
# One-time setup
python scripts/preprocess/extract_frames.py --dataset pie --data-path /data/datasets/PIE
python scripts/preprocess/extract_keypoints.py --dataset pie --data-path /data/datasets/PIE \
    --output-dir /data/datasets/PIE/keypoints_pid

# Base training (step 0)
python scripts/sparsetemporalpie/train_SparseTemporalPIE.py \
    --weights weights_v8/model_8_PIE_IL_step14_new.pth \
    --output-dir weights_sparse_v4 --epochs 50 --device cuda:0

# IL steps 2→14
python scripts/sparsetemporalpie/pie_sparse_incremental_learning.py \
    --weights weights_sparse_v4/best_sparse_step0.pth \
    --output-dir weights_sparse_v4 --restart-period 7 --device cuda:0

# Evaluate v3
python scripts/sparsetemporalpie/test_SparseTemporalPIE_v3.py \
    --weights weights_sparse_v3/best_sparse_step14.pth \
    --step 14 --device cuda:0

# Evaluate v4
python scripts/sparsetemporalpie/test_SparseTemporalPIE.py \
    --weights weights_sparse_v4/best_sparse_step2.pth \
    --step 2 --device cuda:0

# Or run the full pipeline
bash pipelines/run_sparse_pie_pipeline.sh
```

### EfficientPIE (baseline)

```bash
python scripts/efficientpie/train_EfficientPIE.py --step 0 --epochs 50 --batch_size 32 \
    --weights pre_train_weights/min_loss_pretrained_model_imagenet.pth
python scripts/efficientpie/pie_domain_incremental_learning.py --step 2 \
    --prev_weights weights_v8/best_model_PIE_step0.pth
python scripts/efficientpie/test_EfficientPIE.py \
    --weights weights_v8/best_model_PIE_IL_step14_new.pth
```

---

## Documentation

- [`docs/RESULTS.md`](docs/RESULTS.md) — full results, SOTA comparison, ablation, inference benchmarks
- [`docs/SPARSE_TEMPORAL_PIE.md`](docs/SPARSE_TEMPORAL_PIE.md) — architecture and implementation guide
- [`docs/REPLICATION_RESULTS.md`](docs/REPLICATION_RESULTS.md) — EfficientPIE replication metrics
- [`docs/SESSION_NOTES_2026-03-18.md`](docs/SESSION_NOTES_2026-03-18.md) — v3/v4 design decisions
- [`docs/SESSION_NOTES_2026-03-19.md`](docs/SESSION_NOTES_2026-03-19.md) — final test results

---

## Citation

If you use this work, please cite the original EfficientPIE paper:

```bibtex
@inproceedings{efficientpie2025,
  title     = {EfficientPIE: Real-Time Prediction on Pedestrian Crossing Intention with Sole Observation},
  booktitle = {Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2025}
}
```
