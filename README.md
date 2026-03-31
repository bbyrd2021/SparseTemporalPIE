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
| **v3** | Cross-attention + pose velocity + ctx MLP | 9.0M | 2.50ms | **0.9261** | **0.9468** |
| v4 | No attention, static pose + ctx MLP only | 1.1M | 0.46ms | 0.9194 | 0.9220 |
| EfficientPIE (paper) | Single frame, visual only | 0.69M | 0.21ms | 0.920 | 0.917 |

Inference measured at batch=128, 50-run warm-up, 100 timed CUDA event runs.
End-to-end including upstream ViTPose-B pose estimation (3.875ms): v3 = 6.38ms, v4 = 4.34ms — both real-time at 30fps.

See [`docs/RESULTS.md`](docs/RESULTS.md) for full SOTA comparison and JAAD results.

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

Context frames are selected as `np.linspace(0, step-1, min(K=4, step))` at each IL step,
so the temporal window expands from 1 frame at step 0 to the full 15-frame sequence at step 14.

---

## Results

### PIE Test Set

| Metric    | EfficientPIE (paper) | v4 (ours) | v3 (ours) |
|-----------|---------------------|-----------|-----------|
| Accuracy  | 0.920               | 0.919     | **0.926** |
| AUC       | 0.917               | 0.922     | **0.947** |
| F1        | 0.952               | 0.953     | **0.957** |
| Precision | 0.960               | 0.958     | 0.957     |
| Inference | 0.21ms              | 0.46ms    | 2.50ms    |

### JAAD Test Set

| Metric    | EfficientPIE (paper) | v3 (ours) |
|-----------|---------------------|-----------|
| Accuracy  | **0.890**           | 0.878     |
| AUC       | 0.860               | **0.885** |
| F1        | 0.620               | **0.633** |
| Precision | **0.630**           | 0.597     |

v3 consistently improves AUC across both datasets (+0.030 PIE, +0.025 JAAD), producing
better-calibrated risk scores for safety-critical downstream planners.

See [`docs/RESULTS.md`](docs/RESULTS.md) for full SOTA comparison table, ablation study, and IL step progression.

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
    train_SparseTemporalPIE.py           # v4 PIE base training (step 0)
    pie_sparse_incremental_learning.py   # v4 PIE IL steps 2→14
    train_SparseTemporalPIE_v3.py        # v3 PIE base training (step 0)
    pie_sparse_incremental_learning_v3.py # v3 PIE IL steps 2→14
    train_SparseTemporalPIE_v3_jaad.py   # v3 JAAD base training (step 0)
    jaad_sparse_incremental_learning_v3.py # v3 JAAD IL steps 2→14
    test_SparseTemporalPIE.py            # v4 evaluation (PIE + JAAD)
    test_SparseTemporalPIE_v3.py         # v3 PIE evaluation
    test_SparseTemporalPIE_v3_jaad.py    # v3 JAAD evaluation
    benchmark_inference.py              # latency benchmark (batch=128, CUDA events)
  ablation/
    calibrate_change_detector.py        # ChangeDetector ablation tool

pipelines/
  run_pie_pipeline.sh                    # full EfficientPIE PIE pipeline
  run_jaad_pipeline.sh                   # full EfficientPIE JAAD pipeline
  run_sparse_pie_pipeline.sh             # full SparseTemporalPIE v4 PIE pipeline
  run_sparse_v3_imagenet_pipeline.sh     # v3 PIE pipeline (ImageNet backbone)
  run_training_after_extraction.sh       # wait for extraction then train

weights_sparse_v3/                       # v3 PIE IL checkpoints (steps 0–14)
weights_sparse_v3_jaad/                  # v3 JAAD IL checkpoints (steps 0–14)
weights_sparse_v4/                       # v4 PIE IL checkpoints (steps 0–14)

docs/
  RESULTS.md                             # full results and SOTA comparison
  SPARSE_TEMPORAL_PIE.md                 # architecture and implementation guide
  SESSION_NOTES_*.md                     # development session logs
```

> All scripts are run from the repo root, e.g. `python scripts/sparsetemporalpie/train_SparseTemporalPIE_v3.py`

---

## Installation

```bash
pip install -r requirements.txt
```

**Dataset setup (PIE):**

```bash
# Annotations
ln -s /path/to/PedestrianIntent++/PIE/PIE/annotations/annotations /data/datasets/PIE/annotations
ln -s /path/to/PedestrianIntent++/PIE/PIE/annotations/annotations_attributes /data/datasets/PIE/annotations_attributes
ln -s /path/to/PedestrianIntent++/PIE/PIE/annotations/annotations_vehicle /data/datasets/PIE/annotations_vehicle

# PIE clip layout
mkdir /data/datasets/PIE/PIE_clips
for i in 01 02 03 04 05 06; do
  ln -s /data/datasets/PIE/set$i /data/datasets/PIE/PIE_clips/set$i
done
```

**Dataset setup (JAAD):**

```bash
ln -s /path/to/PedestrianIntent++/JAAD/annotations /data/datasets/JAAD/annotations
ln -s /path/to/PedestrianIntent++/JAAD/annotations_attributes /data/datasets/JAAD/annotations_attributes
ln -s /path/to/PedestrianIntent++/JAAD/annotations_appearance /data/datasets/JAAD/annotations_appearance
ln -s /path/to/PedestrianIntent++/JAAD/annotations_traffic /data/datasets/JAAD/annotations_traffic
ln -s /path/to/PedestrianIntent++/JAAD/annotations_vehicle /data/datasets/JAAD/annotations_vehicle
ln -s /path/to/PedestrianIntent++/JAAD/split_ids /data/datasets/JAAD/split_ids
```

---

## Usage

### SparseTemporalPIE v3 — PIE

```bash
# One-time setup
python scripts/preprocess/extract_frames.py --dataset pie --data-path /data/datasets/PIE
python scripts/preprocess/extract_keypoints.py --dataset pie --data-path /data/datasets/PIE \
    --output-dir /data/datasets/PIE/keypoints_pid

# Base training (step 0)
python scripts/sparsetemporalpie/train_SparseTemporalPIE_v3.py \
    --weights weights_v8/model_8_PIE_IL_step14_new.pth \
    --output-dir weights_sparse_v3 --epochs 50 --device cuda:0

# IL steps 2→14
python scripts/sparsetemporalpie/pie_sparse_incremental_learning_v3.py \
    --weights weights_sparse_v3/best_sparse_v3_step0.pth \
    --output-dir weights_sparse_v3 --device cuda:0

# Evaluate
python scripts/sparsetemporalpie/test_SparseTemporalPIE_v3.py \
    --weights weights_sparse_v3/best_sparse_step14.pth \
    --step 14 --device cuda:0
```

### SparseTemporalPIE v3 — JAAD

```bash
# One-time setup
python scripts/preprocess/extract_frames.py --dataset jaad --data-path /data/datasets/JAAD
python scripts/preprocess/extract_keypoints.py --dataset jaad --data-path /data/datasets/JAAD \
    --output-dir /data/datasets/JAAD/keypoints_pid

# Base training (step 0)
python scripts/sparsetemporalpie/train_SparseTemporalPIE_v3_jaad.py \
    --weights weights_v8/model_8_PIE_IL_step14_new.pth \
    --output-dir weights_sparse_v3_jaad --epochs 50 --device cuda:0

# IL steps 2→14
python scripts/sparsetemporalpie/jaad_sparse_incremental_learning_v3.py \
    --weights weights_sparse_v3_jaad/best_sparse_v3_jaad_step0.pth \
    --output-dir weights_sparse_v3_jaad --device cuda:0

# Evaluate
python scripts/sparsetemporalpie/test_SparseTemporalPIE_v3_jaad.py \
    --weights weights_sparse_v3_jaad/best_sparse_v3_jaad_step14.pth \
    --step 14 --device cuda:0
```

### Inference Benchmark

```bash
python scripts/sparsetemporalpie/benchmark_inference.py \
    --weights-v3 weights_sparse_v3/best_sparse_step14.pth \
    --weights-v4 weights_sparse_v4/best_sparse_step2.pth \
    --batch-size 128 --warmup 50 --runs 100 --device cuda:0
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

- [`docs/RESULTS.md`](docs/RESULTS.md) — full results, SOTA comparison (PIE + JAAD), ablation, inference benchmarks
- [`docs/SPARSE_TEMPORAL_PIE.md`](docs/SPARSE_TEMPORAL_PIE.md) — architecture and implementation guide
- [`docs/REPLICATION_RESULTS.md`](docs/REPLICATION_RESULTS.md) — EfficientPIE replication metrics
- [`docs/SESSION_NOTES_2026-03-26.md`](docs/SESSION_NOTES_2026-03-26.md) — backbone ablation, JAAD setup
- [`docs/SESSION_NOTES_2026-03-27.md`](docs/SESSION_NOTES_2026-03-27.md) — JAAD training results, full eval

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
