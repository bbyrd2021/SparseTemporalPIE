# SparseTemporalPIE

> Fork of [EfficientPIE](https://github.com/heinideyibadiaole/EfficientPIE) (IJCAI-25)
> Extended with temporal cross-attention keyframe selection via ViTPose.
> Authors: Brandon Byrd, Abel Abebe Bzuayene — xDI Lab, NC A&T State University

---

## Overview

**EfficientPIE** predicts pedestrian crossing intention from a single 300×300 image crop
using an EfficientNet-inspired backbone (no RNN). It achieves state-of-the-art accuracy
at 0.69M parameters and sub-millisecond inference.

**SparseTemporalPIE** extends EfficientPIE with a temporal mechanism:
- A ViTPose-based ChangeDetector selects the last behaviorally-salient keyframe (f*)
  from the 15-frame observation window using head orientation, body lean, and gaze signals
- f_current and f* are encoded by a shared frozen EfficientPIE backbone
- A cross-attention module with a hard gate fuses the two embeddings
- The absence flag explicitly signals when no behavioral change was detected

Architecture:

```
frozen backbone(f_current) ──► embed_current (1280)
frozen backbone(f*)        ──► embed_fstar   (1280)
                                      │
                     CrossAttention(Q=embed_current, K/V=embed_fstar)
                                      │
                       hard gate: attn_out × (1 − absence_flag)
                                      │
                       residual + LayerNorm + FeedForward
                                      │
                       concat([enriched, absence_flag])  →  1281-d
                                      │
                       Classifier (1281 → 256 → 2)  →  crossing probability
```

---

## Verified Results (PIE Test Set)

| Metric    | SparseTemporalPIE | EfficientPIE (replicated) | Paper (Table 3) |
|-----------|:-----------------:|:-------------------------:|:---------------:|
| Accuracy  | TBD               | 0.918                     | 0.92            |
| AUC       | TBD               | 0.917                     | 0.92            |
| F1        | TBD               | 0.952                     | 0.95            |
| Precision | TBD               | 0.961                     | 0.96            |

See [`docs/REPLICATION_RESULTS.md`](docs/REPLICATION_RESULTS.md) for full evaluation details.

---

## Repository Structure

```
models/
  EfficientPIE.py                 # baseline model
  SparseTemporalPIE.py            # temporal cross-attention model

utils/
  pie_data.py / jaad_data.py      # dataset APIs
  my_dataset.py                   # EfficientPIE dataset loader
  sparse_dataset.py               # SparseTemporalPIE dataset loader
  change_detector.py              # ViTPose-based keyframe selector
  train_val.py                    # training/eval loops

train_EfficientPIE.py             # EfficientPIE base training
pie_domain_incremental_learning.py # IL steps 2→14
test_EfficientPIE.py              # EfficientPIE evaluation

train_SparseTemporalPIE.py        # SparseTemporalPIE base training
pie_sparse_incremental_learning.py # SparseTemporalPIE IL steps
test_SparseTemporalPIE.py         # evaluation + v=0 subset metrics

extract_frames.py                 # video → image frames (run once)
extract_keypoints.py              # ViTPose keypoint extraction (run once)
calibrate_change_detector.py      # threshold calibration (run once)

run_sparse_pie_pipeline.sh        # full SparseTemporalPIE pipeline
run_pie_pipeline.sh               # full EfficientPIE pipeline

docs/                             # design docs, session notes, paper PDF
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

# 2. Base training (step 0)
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
    --output-dir /data/datasets/PIE/keypoints
python calibrate_change_detector.py --data-path /data/datasets/PIE \
    --keypoints-dir /data/datasets/PIE/keypoints

# Full pipeline
bash run_sparse_pie_pipeline.sh
```

---

## Documentation

- [`docs/SPARSE_TEMPORAL_PIE.md`](docs/SPARSE_TEMPORAL_PIE.md) — full architecture and implementation guide
- [`docs/REPLICATION_RESULTS.md`](docs/REPLICATION_RESULTS.md) — verified EfficientPIE replication metrics
- [`docs/SESSION_NOTES_2026-03-10.md`](docs/SESSION_NOTES_2026-03-10.md) — bug fixes and replication notes
- [`docs/SESSION_NOTES_2026-03-10b.md`](docs/SESSION_NOTES_2026-03-10b.md) — SparseTemporalPIE implementation notes

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
