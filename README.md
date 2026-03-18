# SparseTemporalPIE

> Fork of [EfficientPIE](https://github.com/heinideyibadiaole/EfficientPIE) (IJCAI-25)
> Extended with embedding-distance keyframe selection and normalized pose features.
> Authors: Brandon Byrd, Abel Abebe Bzuayene — xDI Lab, NC A&T State University

---

## Overview

**EfficientPIE** predicts pedestrian crossing intention from a single 300×300 image crop
using an EfficientNet-inspired backbone (no RNN). It achieves state-of-the-art accuracy
at 0.69M parameters and sub-millisecond inference.

**SparseTemporalPIE** extends EfficientPIE with a temporal mechanism:
- A salient reference frame (f\*) is selected from the observation window as the frame
  with maximum cosine distance from f\_current in the frozen backbone embedding space
- f\_current and f\* are encoded by a shared frozen EfficientPIE backbone
- Cross-attention fuses the two embeddings (f\_current queries, f\* keys/values)
- Normalized ViTPose keypoints for f\_current are concatenated to the classifier input,
  providing explicit body-pose signal alongside the temporal attention context

Architecture:

```
                    ┌─ frozen backbone ──► embed_current (1280)
f_current (300×300) ┤
                    └─ (also for f* selection via cosine distance)

f* = argmax cosine_distance(embed_current, embed_t) for t in [0, step-1]
     (computed in SparseDataset using a CPU copy of the backbone)

f*       (300×300) ──► frozen backbone ──► embed_fstar (1280)

CrossAttention(Q=embed_current, K=embed_fstar, V=embed_fstar)
    │
    attn_out (1280)
    │
residual: embed_current + attn_out  → LayerNorm
    │
FeedForward: 1280 → 512 → 1280  → LayerNorm
    │
    enriched (1280)

pose_feats (34) ← normalize_pose(keypoints[step], bbox)
                  17 joints × (norm_x, norm_y); low-conf joints zeroed

concat([enriched, pose_feats])  →  1314-d
    │
Classifier: Linear(1314→256) → ReLU → Dropout → Linear(256→2)
    │
crossing probability
```

---

## Verified Results (PIE Test Set)

| Metric    | SparseTemporalPIE | EfficientPIE (replicated) | Paper (Table 3) |
|-----------|:-----------------:|:-------------------------:|:---------------:|
| Accuracy  | TBD               | 0.918                     | 0.92            |
| AUC       | TBD               | 0.917                     | 0.92            |
| F1        | TBD               | 0.952                     | 0.95            |
| Precision | TBD               | 0.961                     | 0.96            |

See [`docs/REPLICATION_RESULTS.md`](docs/REPLICATION_RESULTS.md) for full EfficientPIE details.

---

## Repository Structure

```
models/
  EfficientPIE.py                 # baseline model (unchanged)
  SparseTemporalPIE.py            # temporal cross-attention model

utils/
  pie_data.py / jaad_data.py      # dataset APIs
  my_dataset.py                   # EfficientPIE dataset loader
  sparse_dataset.py               # SparseTemporalPIE dataset (f_current, f*, pose_feats)
  change_detector.py              # ViTPose ChangeDetector (retained for ablation)
  train_val.py                    # training/eval loops (EfficientPIE + SparseTemporalPIE)

train_EfficientPIE.py             # EfficientPIE base training
pie_domain_incremental_learning.py # IL steps 2→14
test_EfficientPIE.py              # EfficientPIE evaluation

train_SparseTemporalPIE.py        # SparseTemporalPIE base training (step 0)
pie_sparse_incremental_learning.py # SparseTemporalPIE IL steps 2→14
test_SparseTemporalPIE.py         # evaluation + v=0 subset metrics

extract_frames.py                 # video → image frames (run once)
extract_keypoints.py              # ViTPose keypoint extraction (run once)
calibrate_change_detector.py      # ablation tool — not used in main pipeline

run_sparse_pie_pipeline.sh        # full SparseTemporalPIE pipeline
run_pie_pipeline.sh               # full EfficientPIE pipeline

docs/                             # design docs and session notes
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dataset setup:**

Annotations from [PedestrianIntent++](https://github.com/...):
```bash
ln -s /path/to/PedestrianIntent++/PIE/PIE/annotations/annotations /data/datasets/PIE/annotations
ln -s /path/to/PedestrianIntent++/PIE/PIE/annotations/annotations_attributes /data/datasets/PIE/annotations_attributes
ln -s /path/to/PedestrianIntent++/PIE/PIE/annotations/annotations_vehicle /data/datasets/PIE/annotations_vehicle
ln -s /path/to/PedestrianIntent++/JAAD/annotations /data/datasets/JAAD/annotations
```

PIE clip layout expected by the API:
```bash
mkdir /data/datasets/PIE/PIE_clips
for i in 01 02 03 04 05 06; do
  ln -s /data/datasets/PIE/set$i /data/datasets/PIE/PIE_clips/set$i
done
```

---

## Usage

### EfficientPIE

```bash
# 1. Extract frames (one-time)
python extract_frames.py --dataset pie --data-path /data/datasets/PIE

# 2. Base training
python train_EfficientPIE.py --step 0 --epochs 50 --batch_size 32 \
    --weights pre_train_weights/min_loss_pretrained_model_imagenet.pth

# 3. Incremental learning (steps 2→14)
python pie_domain_incremental_learning.py --step 2 --prev_weights weights_v8/best_model_PIE_step0.pth

# 4. Evaluate
python test_EfficientPIE.py --weights weights_v8/best_model_PIE_IL_step14_new.pth
```

### SparseTemporalPIE

```bash
# Prerequisites (one-time)
python extract_frames.py --dataset pie --data-path /data/datasets/PIE
python extract_keypoints.py --dataset pie --data-path /data/datasets/PIE \
    --output-dir /data/datasets/PIE/keypoints_pid

# Full pipeline (step 0 + IL steps 2–14 + eval)
bash run_sparse_pie_pipeline.sh
```

---

## Documentation

- [`docs/SPARSE_TEMPORAL_PIE.md`](docs/SPARSE_TEMPORAL_PIE.md) — full architecture and implementation guide
- [`docs/REPLICATION_RESULTS.md`](docs/REPLICATION_RESULTS.md) — verified EfficientPIE replication metrics
- [`docs/SESSION_NOTES_2026-03-10.md`](docs/SESSION_NOTES_2026-03-10.md) — EfficientPIE bug fixes and replication
- [`docs/SESSION_NOTES_2026-03-10b.md`](docs/SESSION_NOTES_2026-03-10b.md) — SparseTemporalPIE initial design
- [`docs/SESSION_NOTES_2026-03-11.md`](docs/SESSION_NOTES_2026-03-11.md) — keypoint extraction, calibration, architecture revision

---

## Citation

If you use this work, please also cite the original EfficientPIE paper:

```bibtex
@inproceedings{efficientpie2025,
  title     = {EfficientPIE: Real-Time Prediction on Pedestrian Crossing Intention with Sole Observation},
  booktitle = {Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2025}
}
```
