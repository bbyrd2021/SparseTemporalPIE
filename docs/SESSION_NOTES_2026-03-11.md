# Session Notes — 2026-03-11

## SparseTemporalPIE: Keypoint Extraction & Calibration

---

### Keypoint Extraction — History of Bugs Fixed

**Bug 1: Full-frame extraction (wrong person)**
- Original `extract_keypoints.py` used `boxes = [[0, 0, W, H]]` — full image bbox
- ViTPose detected the most prominent person in the scene, NOT the annotated pedestrian
- Calibration AUC: **0.47** (below random) — confirmed useless signal
- Fix: manually crop image to pedestrian bbox before passing to ViTPose

**Bug 2: HuggingFace processor ignores `boxes` param for cropping**
- Passing pedestrian bbox as `boxes` to `VitPoseImageProcessor` does NOT crop the image
- The processor uses `boxes` only for post-processing coordinate remapping
- ViTPose still ran on the full frame — keypoints stuck at bottom-right corner (1850, 1035) regardless of bbox
- Fix: `image.crop((x1, y1, x2, y2))` manually, pass `[0, 0, crop_w, crop_h]` to processor, then offset output by `(x1, y1)`

**Bug 3: Path collision — window index t vs absolute frame number**
- Original path convention: `{keypoints_dir}/{pid}/{t:05d}.npy` where `t` = observation window index (0–14)
- Same pedestrian appears in multiple non-overlapping 15-frame sequences → same `(pid, t)` key for different absolute frames
- Only first sequence's keypoints survived; all other sequences loaded wrong frame's keypoints
- Hit rate before fix: **48.9%** (keypoint centroid within bbox ±20px)
- Fix: use absolute frame number from image filename → `{keypoints_dir}/{pid}/{frame_id}.npy`
- Hit rate after fix: **100.0%** (27,088/27,089)

**Bug 4: `_rolling_mean` in ChangeDetector collapsed vector mean**
- `np.mean(valid)` on a list of 2D unit vectors returned a scalar (mean of all elements)
- `np.dot(vector, scalar)` returned an array → `float()` raised `TypeError`
- Fix: `np.mean(valid, axis=0)` for element-wise mean

---

### Final Keypoint Extraction State

| Metric | Value |
|---|---|
| Output dir | `/data/datasets/PIE/keypoints_pid/` |
| Path convention | `{pid}/{frame_id}.npy` (absolute frame number) |
| Total files | 150,240 unique (pid, frame_id) pairs |
| Coordinate space | Full-frame pixels (offset applied after crop) |
| ViTPose model | `usyd-community/vitpose-base-coco-aic-mpii` (ViTPose-B) |
| Crop padding | 25% of bbox dims on each side |
| Hit rate | **100.0%** (centroid within bbox ±20px) |
| Mean confidence | 0.82–0.95 (on pedestrians with bbox_h ≥ 60px) |

---

### Visualization Samples

Saved to `docs/sample_poses/`:
- `pose_samples_gallery.png` — 12-panel gallery, 6 crossing / 6 not-crossing, diverse PIE sets, dark background

---

### Calibration

- Duration: **868 seconds (~14.5 min)**  — ~9s keypoint loading, ~14min grid search
- Best AUC: **0.5398** (marginal above random 0.5)
- Best config:
  ```json
  {
    "head_orient_delta_threshold": 0.3,
    "body_lean_delta_threshold": 0.03,
    "gaze_vector_delta_threshold": 0.3,
    "min_confidence": 0.3,
    "rolling_window": 3
  }
  ```
- Output: `change_detector_config.json`

**Analysis:** AUC of 0.54 is only marginally above random. Thresholds hit the edges of the search grid (head/gaze at max 0.3, lean at min 0.03), indicating the signal is weak and the grid search couldn't find a meaningful sweet spot. Pose-based change detection on PIE is unreliable as a binary fired/not-fired criterion. Root causes:
1. Many PIE pedestrians are small (bbox_h < 60px) — pose quality degrades even with correct cropping
2. Binary firing signal loses too much temporal information
3. Head/gaze/lean signals may not correlate strongly with crossing intent on this dataset

**Next steps to discuss:** drop ViTPose change detector in favor of embedding-distance based f* selection (no external pose dependency), or restrict to large-ped subset.

---

## Architecture Revision — ChangeDetector Replaced (2026-03-11 evening)

### Decision

AUC 0.54 confirmed the binary fired/not-fired signal is too weak to build on. Rather than re-engineer the detector, we replaced the entire selection mechanism with two complementary components that need no calibration:

1. **Embedding-distance f\* selection** — pick the frame in [0, step-1] whose frozen backbone embedding has the largest cosine distance from f\_current. No thresholds, no calibration, fully data-driven.
2. **Normalized pose features** — concatenate a 34-dim keypoint vector (17 joints × xy, normalized by bbox, low-conf zeroed) for f\_current directly to the classifier input. This replaces the scalar `absence_flag` with rich body-pose signal and avoids the information bottleneck.

### Architecture Changes

| Component | Before | After |
|---|---|---|
| f\* selection | `ChangeDetector.detect()` (threshold on head/lean/gaze deltas) | argmax cosine distance in backbone embedding space |
| Temporal signal to classifier | `absence_flag` scalar (1 if not fired) | `pose_feats` 34-dim normalized keypoint vector |
| Hard gate | `attn_out × (1 − absence_flag)` | removed — plain residual |
| Classifier input dim | 1281 (1280 + 1) | 1314 (1280 + 34) |
| Classifier head | Linear(1281→2) | Linear(1314→256) → ReLU → Dropout → Linear(256→2) |
| Calibration required | Yes (`calibrate_change_detector.py`) | No |

### Files Modified

- `models/SparseTemporalPIE.py` — classifier dim, forward signature, removed gate
- `utils/sparse_dataset.py` — complete rewrite: drop ChangeDetector, add `normalize_pose()`, add `_select_fstar()` (cosine distance via CPU backbone copy), add `_load_pose_feats()`
- `utils/train_val.py` — rename `flag` → `pose_feats` in both sparse loops
- `train_SparseTemporalPIE.py` — model created before datasets, `backbone=model.backbone` passed to SparseDataset, removed `--detector-config`
- `pie_sparse_incremental_learning.py` — same
- `test_SparseTemporalPIE.py` — same
- `run_sparse_pie_pipeline.sh` — removed calibration step, updated `--keypoints-dir` to `keypoints_pid`

### SparseDataset Backbone Copy

The frozen backbone is deep-copied to CPU in `SparseDataset.__init__` so it can be safely pickled into `DataLoader` worker processes (CUDA tensors are not fork-safe). Memory cost: ~2.7 MB per worker. The copy is used only during `_select_fstar()` to encode frames [0..step] and compute cosine distances.

### Pose Feature Path Convention

Keypoints stored as `{keypoints_dir}/{pid}/{frame_id}.npy` (absolute frame number, not window index) — see Bug 3 fix above. The `_load_pose_feats()` method in `SparseDataset` uses this convention.

### Training Launched

```bash
bash run_sparse_pie_pipeline.sh
# Step 0: 50 epochs
# Steps 2–14: 30 epochs each (7 IL steps)
# Final eval: test set Acc/AUC/F1/Precision + v=0 subset
```

Weights output: `weights_sparse/best_sparse_step{N}.pth`
Logs: `training_logs_sparse/step0.log`, `training_logs_sparse/il_steps.log`
