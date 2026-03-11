# SparseTemporalPIE — Complete Implementation Guide

> Forked from [EfficientPIE](https://github.com/heinideyibadiaole/EfficientPIE) (IJCAI-25)
> Authors: Brandon Byrd, Abel Abebe Bzuayene — xDI Lab, NC A&T State University

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Design Decisions](#3-design-decisions)
4. [File-by-File Implementation](#4-file-by-file-implementation)
   - 4.1 [models/SparseTemporalPIE.py](#41-modelssparsetemporalpiepy)
   - 4.2 [utils/change_detector.py](#42-utilschange_detectorpy)
   - 4.3 [utils/sparse_dataset.py](#43-utilssparse_datasetpy)
   - 4.4 [train_SparseTemporalPIE.py](#44-train_sparsetemporalpiepy)
   - 4.5 [pie_sparse_incremental_learning.py](#45-pie_sparse_incremental_learningpy)
   - 4.6 [test_SparseTemporalPIE.py](#46-test_sparsetemporalpiepy)
   - 4.7 [run_sparse_pie_pipeline.sh](#47-run_sparse_pie_pipelinesh)
5. [Training Protocol](#5-training-protocol)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Ablation Study Design](#7-ablation-study-design)
8. [Expected Results & Baselines](#8-expected-results--baselines)
9. [Key Implementation Notes](#9-key-implementation-notes)

---

## 1. Architecture Overview

```
Observation window: 15 frames (indices 0–14)

INFERENCE PATH
──────────────
ViTPose keypoints (every frame)
    └── ChangeDetector.detect(keypoints_sequence)
            ├── signals: head_orient_delta, body_lean_delta, gaze_vector_delta
            ├── threshold δ calibrated on PIE behavioral annotations
            └── returns: (f*_index, fired: bool)

if fired:
    f*    = frame at f*_index         │  absence_flag = 0
else:
    f*    = frame at index 0          │  absence_flag = 1

f_current (300×300 crop) ──┐
                           ├──► shared EfficientPIE backbone (FROZEN) ──► embed_current (1280)
f*        (300×300 crop) ──┘                                              embed_fstar   (1280)
                                                                                │
                                                              CrossAttention(Q=embed_current,
                                                                             K=embed_fstar,
                                                                             V=embed_fstar)
                                                                                │
                                                                         attn_out (1280)
                                                                                │
                                                              Hard gate: attn_out × (1 − absence_flag)
                                                              [gate=1 if fired, gate=0 if not fired]
                                                                                │
                                                              residual: embed_current + gated_attn_out
                                                                                │
                                                              FeedForward: 1280 → 512 → 1280
                                                                                │
                                                              concat([enriched, absence_flag])
                                                                                │
                                                                    ClassifierHead (1281 → 2)
                                                                                │
                                                                    crossing probability
```

**What f* represents semantically:**
- When fired: the pedestrian at their last moment of behavioral change — gaze shift, weight shift, head turn. Cross-attention asks "what changed between then and now?" The gate is open (1.0) and the full attention output enriches `embed_current`.
- When not fired: the gate is closed (0.0) — the attention output is suppressed entirely. The classifier reasons from `embed_current` alone, with `absence_flag=1` as the only temporal signal. This prevents frame-0 fallback embeddings from polluting the representation when no behavioral change occurred.

---

## 2. Repository Structure

Fork from EfficientPIE. New and modified files are marked.

```
EfficientPIE/  (forked)
│
├── models/
│   ├── __init__.py
│   ├── common.py                          # UNCHANGED — DropPath, SE, FusedMBConv, MBConv
│   ├── EfficientPIE.py                    # UNCHANGED — baseline model
│   └── SparseTemporalPIE.py               # NEW — our model
│
├── utils/
│   ├── __init__.py
│   ├── my_dataset.py                      # UNCHANGED — EfficientPIE dataset
│   ├── train_val.py                       # UNCHANGED — EfficientPIE train/eval loops
│   ├── change_detector.py                 # NEW — ViTPose-based keyframe selector
│   └── sparse_dataset.py                  # NEW — dataset returning (f_current, f*, flag)
│
├── train_EfficientPIE.py                  # UNCHANGED
├── pie_domain_incremental_learning.py     # UNCHANGED
├── test_EfficientPIE.py                   # UNCHANGED
│
├── train_SparseTemporalPIE.py             # NEW — base training (step=0)
├── pie_sparse_incremental_learning.py     # NEW — IL training (steps 2–14)
├── test_SparseTemporalPIE.py              # NEW — evaluation
│
├── run_sparse_pie_pipeline.sh             # NEW — full automation
├── extract_frames.py                      # EXISTING
├── requirements.txt                       # ADD: torch>=2.0, timm>=0.9
└── pre_train_weights/
    └── min_loss_pretrained_model_imagenet.pth   # EXISTING
```

---

## 3. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Backbone | EfficientPIE (shared, Siamese) | Clean comparison; any delta is purely from temporal mechanism |
| Backbone weights | Trained PIE weights, **frozen** | Accuracy delta is attributable to temporal mechanism alone |
| f* when fired | Detected salient frame | Last moment of behavioral change; cross-attention sees delta |
| f* when absent | First observed frame (index 0) | Near-zero embedding delta encodes behavioral inertia visually |
| Absence flag | Binary scalar appended to classifier input | Disambiguates genuine inertia from coincidental similarity; explicit temporal signal even when gate is closed |
| Cross-attention gate | Hard gate: `attn_out × (1 − absence_flag)` | Suppresses attention output when f* didn't fire; prevents frame-0 fallback from polluting embed_current; architecturally motivated and directly interpretable |
| Cross-attention heads | 8 | Standard for 1280-dim; balances expressiveness and compute |
| Post-attention feedforward | 1280 → 512 → 1280 | Standard transformer practice; adds non-linear mixing |
| IL strategy | Inherited from EfficientPIE (IDIL + adaptive loss) | Compatible; backbone freeze simplifies IL (only attention head drifts) |
| Perturbation | Inherited progressive perturbation | Applied to classifier output logits; no architecture changes needed |
| ViTPose pose stream | **Ablation only** — not in base model | Keeps contribution focused on temporal sampling mechanism |
| CLIP backbone | **Ablation only** — not in base model | Future work; EfficientPIE backbone is the controlled comparison |

---

## 4. File-by-File Implementation

---

### 4.1 `models/SparseTemporalPIE.py`

**Purpose:** The full model. Wraps a frozen EfficientPIE backbone as a shared encoder, adds cross-attention fusion, feedforward layer, and a new classifier head that accepts the absence flag.

```python
"""
SparseTemporalPIE
=================
Architecture:
  - Shared frozen EfficientPIE backbone (Siamese)
  - Cross-attention fusion (f_current queries, f* keys/values)
  - Small feedforward layer (1280 → 512 → 1280)
  - Classifier head (1281 → 2), taking enriched + absence_flag
"""

import torch
import torch.nn as nn
from models.EfficientPIE import EfficientPIE


class SparseTemporalPIE(nn.Module):
    def __init__(self, num_heads=8, ff_hidden=512, dropout=0.1):
        super().__init__()

        # ── Shared frozen backbone ──────────────────────────────────────
        # Load EfficientPIE and strip the classifier head.
        # We need the 1280-dim embedding, not the 2-class output.
        _base = EfficientPIE()
        self.backbone = nn.Sequential(
            _base.commonConv,
            _base.fm1,
            _base.fm2,
            _base.mb1,
            _base.mb2,
            _base.commonConv1,
            _base.avg_pool,
            _base.flatten,
            _base.dropout,
        )
        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        embed_dim = 1280  # EfficientPIE embedding dimension

        # ── Cross-attention ─────────────────────────────────────────────
        # f_current generates Q; f* generates K and V.
        # nn.MultiheadAttention expects (seq_len, batch, embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # ── Feedforward ─────────────────────────────────────────────────
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)

        # ── Classifier head ─────────────────────────────────────────────
        # Input: enriched embedding (1280) + absence_flag (1) = 1281
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def encode(self, x):
        """Pass a (B, 3, 300, 300) crop through the frozen backbone."""
        return self.backbone(x)  # (B, 1280)

    def forward(self, f_current, f_star, absence_flag):
        """
        Args:
            f_current:    (B, 3, 300, 300) — current frame crop
            f_star:       (B, 3, 300, 300) — salient keyframe or first frame
            absence_flag: (B, 1)           — 1 if f* never fired, else 0

        Returns:
            logits: (B, 2)
        """
        # Encode both frames with shared frozen backbone
        emb_current = self.encode(f_current)   # (B, 1280)
        emb_fstar   = self.encode(f_star)       # (B, 1280)

        # MultiheadAttention expects (seq_len, B, embed_dim)
        # seq_len = 1 for both query and key/value
        q = emb_current.unsqueeze(0)   # (1, B, 1280)
        k = emb_fstar.unsqueeze(0)     # (1, B, 1280)
        v = emb_fstar.unsqueeze(0)     # (1, B, 1280)

        attn_out, _ = self.cross_attn(q, k, v)  # (1, B, 1280)
        attn_out = attn_out.squeeze(0)            # (B, 1280)

        # Hard gate: suppress attention output when f* didn't fire.
        # absence_flag=1 → not fired → gate=0 → zero out attn_out.
        # absence_flag=0 → fired     → gate=1 → pass attn_out through fully.
        # This prevents frame-0 fallback embeddings from polluting emb_current
        # when no behavioral change was detected.
        gate     = 1.0 - absence_flag             # (B, 1): 1 if fired, 0 if not
        attn_out = attn_out * gate                # broadcast over 1280 dims

        enriched = self.attn_norm(emb_current + attn_out)
        enriched = self.ff_norm(enriched + self.ff(enriched))  # (B, 1280)

        # absence_flag still reaches the classifier as an explicit temporal signal
        combined = torch.cat([enriched, absence_flag], dim=1)  # (B, 1281)
        logits   = self.classifier(combined)                    # (B, 2)
        return logits


def load_backbone_weights(model, weights_path, device='cuda:0'):
    """
    Load trained EfficientPIE weights into the backbone of SparseTemporalPIE.
    Only backbone-compatible keys are loaded; classifier keys are skipped.

    Args:
        model:        SparseTemporalPIE instance
        weights_path: path to trained EfficientPIE .pth file
        device:       device string
    """
    checkpoint = torch.load(weights_path, map_location=device)
    # EfficientPIE saves state_dict directly or under 'model' key
    src = checkpoint.get('model', checkpoint)

    # Map EfficientPIE keys to backbone Sequential keys
    # EfficientPIE attr names → Sequential indices (must match __init__ order)
    key_map = {
        'commonConv': '0',
        'fm1': '1',
        'fm2': '2',
        'mb1': '3',
        'mb2': '4',
        'commonConv1': '5',
        'avg_pool': '6',
        'flatten': '7',
        'dropout': '8',
    }

    target = {}
    for src_key, src_val in src.items():
        for name, idx in key_map.items():
            if src_key.startswith(name + '.'):
                new_key = 'backbone.' + idx + src_key[len(name):]
                target[new_key] = src_val
                break

    missing, unexpected = model.load_state_dict(target, strict=False)
    loaded = len(target) - len(unexpected)
    print(f"Backbone weights loaded: {loaded}/{len(src)} keys")
    print(f"  Missing (expected — new layers): {missing}")
    print(f"  Unexpected (ignored): {unexpected}")
    return model
```

**Key notes:**
- `encode()` is exposed as a separate method so we can call it on both branches cleanly and also reuse it in the ablation that concatenates pose features.
- The hard gate (`attn_out × (1 − absence_flag)`) explicitly suppresses the cross-attention output when f* didn't fire. Without this, a frame-0 fallback embedding would flow into the residual and pollute `embed_current` even when no behavioral change was detected. The gate makes the two cases architecturally distinct rather than relying on the network to learn the distinction implicitly.
- The residual connections around both the attention and feedforward layers follow standard transformer practice and stabilize training of the new layers on top of the frozen backbone.
- `absence_flag` serves a dual role: it controls the gate AND is appended to the classifier input as an explicit temporal signal. Even with the gate closed, the classifier knows via the flag that it is reasoning in the "no change detected" regime.
- `absence_flag` is passed as a float tensor `(B, 1)` — the caller is responsible for casting.

---

### 4.2 `utils/change_detector.py`

**Purpose:** Takes a sequence of ViTPose keypoint arrays (one per frame) and returns the index of the last salient behavioral change frame, or `None` if no change exceeded the thresholds.

**ViTPose keypoint layout (COCO 17-point):**

| Index | Keypoint | Used for |
|---|---|---|
| 0 | Nose | Gaze vector origin |
| 1, 2 | Left/Right eye | Gaze vector direction |
| 5, 6 | Left/Right shoulder | Body lean |
| 11, 12 | Left/Right hip | Body lean |

```python
"""
ChangeDetector
==============
Monitors three pose geometry signals per frame:
  1. Head orientation delta  — yaw proxy from nose-to-eye midpoint vector
  2. Body lean angle         — shoulder midpoint vs. hip midpoint vertical delta
  3. Gaze vector change      — nose-to-eye-midpoint direction shift

Thresholds δ are calibrated offline against PIE behavioral annotations.
They are loaded from a config file and frozen at inference time.

Usage:
    detector = ChangeDetector(config_path='change_detector_config.json')
    f_star_idx, fired = detector.detect(keypoints_seq)
"""

import json
import numpy as np
from pathlib import Path


# Default thresholds — override with calibrated values after PIE calibration
DEFAULT_CONFIG = {
    "head_orient_delta_threshold": 0.15,   # cosine distance units
    "body_lean_delta_threshold":   0.08,   # normalized pixel units
    "gaze_vector_delta_threshold": 0.12,   # cosine distance units
    "min_confidence":              0.3,    # ViTPose keypoint confidence floor
    "rolling_window":              3,      # frames to smooth signal over
}

# COCO keypoint indices
NOSE          = 0
LEFT_EYE      = 1
RIGHT_EYE     = 2
LEFT_SHOULDER = 5
RIGHT_SHOULDER= 6
LEFT_HIP      = 11
RIGHT_HIP     = 12


class ChangeDetector:
    def __init__(self, config_path=None):
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.cfg = json.load(f)
        else:
            self.cfg = DEFAULT_CONFIG.copy()

        self.δ_head  = self.cfg['head_orient_delta_threshold']
        self.δ_lean  = self.cfg['body_lean_delta_threshold']
        self.δ_gaze  = self.cfg['gaze_vector_delta_threshold']
        self.min_conf = self.cfg['min_confidence']
        self.window   = self.cfg['rolling_window']

    # ── Signal extractors ────────────────────────────────────────────────

    def _head_orient_vector(self, kpts):
        """
        Proxy for head yaw: vector from nose to midpoint of eyes.
        kpts: (17, 3) array — [x, y, confidence]
        Returns unit vector or None if keypoints not confident.
        """
        nose     = kpts[NOSE]
        leye     = kpts[LEFT_EYE]
        reye     = kpts[RIGHT_EYE]
        if any(p[2] < self.min_conf for p in [nose, leye, reye]):
            return None
        eye_mid = (leye[:2] + reye[:2]) / 2.0
        v = eye_mid - nose[:2]
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-6 else None

    def _body_lean_angle(self, kpts, bbox_height):
        """
        Shoulder midpoint vertical position relative to hip midpoint,
        normalized by bounding box height to be scale-invariant.
        Returns scalar or None if keypoints not confident.
        """
        ls, rs = kpts[LEFT_SHOULDER], kpts[RIGHT_SHOULDER]
        lh, rh = kpts[LEFT_HIP],      kpts[RIGHT_HIP]
        if any(p[2] < self.min_conf for p in [ls, rs, lh, rh]):
            return None
        shoulder_mid_y = (ls[1] + rs[1]) / 2.0
        hip_mid_y      = (lh[1] + rh[1]) / 2.0
        if bbox_height < 1e-6:
            return None
        return (shoulder_mid_y - hip_mid_y) / bbox_height

    def _gaze_vector(self, kpts):
        """
        Same as head_orient_vector — reused as gaze proxy.
        A more accurate gaze signal would use head pose estimation,
        but nose-to-eye-midpoint is sufficient as a change signal.
        """
        return self._head_orient_vector(kpts)

    # ── Cosine distance between two unit vectors ────────────────────────

    @staticmethod
    def _cosine_dist(v1, v2):
        if v1 is None or v2 is None:
            return 0.0
        return 1.0 - float(np.dot(v1, v2))

    # ── Rolling baseline ────────────────────────────────────────────────

    def _rolling_mean(self, values, t):
        """Mean of non-None values in [max(0, t-window), t)."""
        start = max(0, t - self.window)
        valid = [v for v in values[start:t] if v is not None]
        return np.mean(valid) if valid else None

    # ── Main detection function ─────────────────────────────────────────

    def detect(self, keypoints_seq, bbox_heights=None):
        """
        Scan a sequence of keypoint arrays and return the index of the
        last frame where any signal exceeded its threshold.

        Args:
            keypoints_seq: list of (17, 3) numpy arrays, one per frame
                           or None if ViTPose failed on that frame
            bbox_heights:  list of floats (pedestrian bbox height in pixels)
                           used to normalize body lean; if None, lean is skipped

        Returns:
            (f_star_idx: int or None, fired: bool)
            f_star_idx is None when no threshold was exceeded.
        """
        n = len(keypoints_seq)
        if n == 0:
            return None, False

        # Pre-compute per-frame signals
        head_vecs  = []
        lean_vals  = []
        gaze_vecs  = []

        for t, kpts in enumerate(keypoints_seq):
            if kpts is None:
                head_vecs.append(None)
                lean_vals.append(None)
                gaze_vecs.append(None)
                continue
            h = bbox_heights[t] if bbox_heights else None
            head_vecs.append(self._head_orient_vector(kpts))
            lean_vals.append(self._body_lean_angle(kpts, h) if h else None)
            gaze_vecs.append(self._gaze_vector(kpts))

        # Scan for threshold crossings, track last firing index
        f_star_idx = None
        for t in range(1, n):
            baseline_head = self._rolling_mean(head_vecs, t)
            baseline_lean = self._rolling_mean(lean_vals, t)
            baseline_gaze = self._rolling_mean(gaze_vecs, t)

            head_delta = self._cosine_dist(head_vecs[t], baseline_head)
            gaze_delta = self._cosine_dist(gaze_vecs[t], baseline_gaze)

            lean_delta = 0.0
            if lean_vals[t] is not None and baseline_lean is not None:
                lean_delta = abs(lean_vals[t] - baseline_lean)

            if (head_delta >= self.δ_head or
                lean_delta >= self.δ_lean or
                    gaze_delta >= self.δ_gaze):
                f_star_idx = t

        fired = f_star_idx is not None
        return f_star_idx, fired

    # ── Calibration helper ───────────────────────────────────────────────

    def calibrate(self, keypoints_by_pid, labels_by_pid, output_path=None):
        """
        Grid search over threshold combinations to find values that
        best separate crossing (label=1) from not-crossing (label=0)
        pedestrian tracks by firing rate.

        Args:
            keypoints_by_pid: dict {pid: list of (17,3) arrays}
            labels_by_pid:    dict {pid: int} — 1=crossing, 0=not crossing
            output_path:      if provided, saves best config as JSON

        Returns:
            best_config: dict
        """
        import itertools
        from sklearn.metrics import roc_auc_score

        δ_head_range = np.arange(0.05, 0.35, 0.05)
        δ_lean_range = np.arange(0.03, 0.20, 0.03)
        δ_gaze_range = np.arange(0.05, 0.35, 0.05)

        best_auc  = 0.0
        best_cfg  = self.cfg.copy()

        for δh, δl, δg in itertools.product(δ_head_range, δ_lean_range, δ_gaze_range):
            self.δ_head = δh
            self.δ_lean = δl
            self.δ_gaze = δg

            y_true, y_score = [], []
            for pid, kpts_seq in keypoints_by_pid.items():
                _, fired = self.detect(kpts_seq)
                y_true.append(labels_by_pid[pid])
                y_score.append(1.0 if fired else 0.0)

            if len(set(y_true)) < 2:
                continue
            auc = roc_auc_score(y_true, y_score)
            if auc > best_auc:
                best_auc = auc
                best_cfg = {
                    **self.cfg,
                    'head_orient_delta_threshold': float(δh),
                    'body_lean_delta_threshold':   float(δl),
                    'gaze_vector_delta_threshold': float(δg),
                }

        print(f"Best calibration AUC: {best_auc:.4f}")
        print(f"Best config: {best_cfg}")

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(best_cfg, f, indent=2)

        # Restore best
        self.δ_head = best_cfg['head_orient_delta_threshold']
        self.δ_lean = best_cfg['body_lean_delta_threshold']
        self.δ_gaze = best_cfg['gaze_vector_delta_threshold']
        return best_cfg
```

**Calibration workflow** (run once before training, separate script):

```bash
python calibrate_change_detector.py \
    --data-path /data/datasets/PIE \
    --output    change_detector_config.json
```

The calibration script runs ViTPose over all annotated PIE tracks, calls `detector.calibrate()` with the `intention_binary` labels from the PIE API, and saves the best thresholds. These thresholds are then frozen for all training and inference.

---

### 4.3 `utils/sparse_dataset.py`

**Purpose:** Dataset wrapper that returns `(f_current, f_star, absence_flag, label)` tuples. Inherits the frame-reading and annotation logic from `my_dataset.py`.

```python
"""
SparseDataset
=============
Wraps the PIE/JAAD dataset to return (f_current, f*, absence_flag, label)
for each sample.

For each pedestrian track at IL step `step`:
  - f_current = frame at observation index `step`
  - f*        = last salient keyframe detected by ChangeDetector
                over frames [0, step]
              = frame at index 0 if no salient frame detected
  - absence_flag = 0 if detector fired, 1 otherwise

Keypoints are pre-computed offline and stored as .npy files to avoid
running ViTPose at training time.
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.change_detector import ChangeDetector


class SparseDataset(Dataset):
    def __init__(
        self,
        data_opts,           # same format as my_dataset.py
        preprocess_opts,     # same format as my_dataset.py
        split='train',       # 'train' or 'test'
        step=0,              # IL step index (0, 2, 4, 6, 8, 10, 12, 14)
        keypoints_dir=None,  # path to pre-computed ViTPose .npy files
        detector_config=None # path to calibrated change_detector_config.json
    ):
        self.data_opts = data_opts
        self.split = split
        self.step = step  # current observation step (f_current index)
        self.max_steps = data_opts.get('max_size_observe', 15)

        # Transform (same as EfficientPIE my_dataset.py)
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Change detector
        self.detector = ChangeDetector(config_path=detector_config)

        # Keypoints directory: {keypoints_dir}/{pid}/{frame_idx:05d}.npy
        # Each .npy is a (17, 3) array or zeros if ViTPose failed
        self.keypoints_dir = keypoints_dir

        # Load samples from PIE/JAAD API
        # (mirrors the logic in my_dataset.py)
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Returns list of dicts, each with:
          {
            'pid':         pedestrian id string
            'frame_paths': list of 15 image file paths (indices 0–14)
            'bbox_heights': list of 15 bbox heights in pixels
            'label':       int (0 or 1)
          }
        """
        # NOTE: Implement using PIE API identical to my_dataset.py.
        # The only difference is storing all 15 frame paths per sample
        # instead of just the frame at `reverse_step`.
        # See my_dataset.py for the API call pattern.
        raise NotImplementedError(
            "Populate _load_samples() using the PIE/JAAD API, "
            "following the pattern in my_dataset.py."
        )

    def _load_keypoints(self, pid, frame_idx):
        """Load pre-computed ViTPose keypoints for a specific frame."""
        if self.keypoints_dir is None:
            return None
        path = os.path.join(self.keypoints_dir, pid, f"{frame_idx:05d}.npy")
        if not os.path.exists(path):
            return None
        kpts = np.load(path)  # (17, 3)
        return kpts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pid           = sample['pid']
        frame_paths   = sample['frame_paths']  # list of 15 paths
        bbox_heights  = sample['bbox_heights']
        label         = sample['label']

        # ── Determine f* via change detector ────────────────────────────
        # Only look at frames up to current IL step
        kpts_seq = [
            self._load_keypoints(pid, t)
            for t in range(self.step + 1)
        ]
        heights_seq = bbox_heights[:self.step + 1]

        if any(k is not None for k in kpts_seq):
            f_star_idx, fired = self.detector.detect(kpts_seq, heights_seq)
        else:
            # No keypoints available — treat as no change
            f_star_idx, fired = None, False

        # ── Select frames ────────────────────────────────────────────────
        absence_flag = 0.0 if fired else 1.0

        f_current_path = frame_paths[self.step]
        f_star_path    = frame_paths[f_star_idx] if fired else frame_paths[0]

        # ── Load and transform images ────────────────────────────────────
        f_current_img = Image.open(f_current_path).convert('RGB')
        f_star_img    = Image.open(f_star_path).convert('RGB')

        f_current_tensor = self.transform(f_current_img)
        f_star_tensor    = self.transform(f_star_img)
        flag_tensor      = torch.tensor([absence_flag], dtype=torch.float32)
        label_tensor     = torch.tensor(label, dtype=torch.long)

        return f_current_tensor, f_star_tensor, flag_tensor, label_tensor
```

**Offline keypoint extraction** (run once, before any training):

```bash
python extract_keypoints.py \
    --dataset      pie \
    --data-path    /data/datasets/PIE \
    --output-dir   /data/datasets/PIE/keypoints \
    --device       cuda:0
```

`extract_keypoints.py` runs ViTPose (via `timm` or MMPose) over every annotated frame and saves `(17, 3)` `.npy` files. This is done once; `SparseDataset` reads from disk at training time with no ViTPose dependency at train time.

---

### 4.4 `train_SparseTemporalPIE.py`

**Purpose:** Base training script. Trains step=0 (only the first observed frame, f* = frame 0, absence_flag = 1 always) for 50 epochs. Mirrors `train_EfficientPIE.py` structure.

```python
"""
train_SparseTemporalPIE.py
==========================
Base training: IL step=0, 50 epochs.

At step=0:
  - f_current = frame at index 0
  - f*        = frame at index 0  (same — no history to detect from)
  - absence_flag = 1 always       (by definition: can't fire at step 0)

This is the equivalent of EfficientPIE's base training on 1/15 of samples.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.SparseTemporalPIE import SparseTemporalPIE, load_backbone_weights
from utils.sparse_dataset import SparseDataset
from utils.train_val import robust_noisy, evaluate   # reuse EfficientPIE utilities


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    # Note: backbone is frozen, but we still call model.train() so
    # dropout in the new layers behaves correctly.
    total_loss = 0.0
    criterion  = nn.CrossEntropyLoss()

    for f_current, f_star, flag, labels in loader:
        f_current = f_current.to(device)
        f_star    = f_star.to(device)
        flag      = flag.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        logits = model(f_current, f_star, flag)

        # Progressive perturbation (inherited from EfficientPIE)
        logits_perturbed = robust_noisy(logits, epoch, total_epochs=total_epochs)
        loss = criterion(logits_perturbed, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def main(args):
    device = torch.device(args.device)

    # Dataset
    data_opts = {
        'max_size_observe': 15,
        'data_path': args.data_path,
        # ... mirror EfficientPIE's data_opts
    }
    train_set = SparseDataset(
        data_opts=data_opts,
        preprocess_opts={},
        split='train',
        step=0,  # base training: step 0 only
        keypoints_dir=args.keypoints_dir,
        detector_config=args.detector_config,
    )
    val_set = SparseDataset(
        data_opts=data_opts,
        preprocess_opts={},
        split='test',
        step=0,
        keypoints_dir=args.keypoints_dir,
        detector_config=args.detector_config,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = SparseTemporalPIE(num_heads=8, ff_hidden=512).to(device)
    model = load_backbone_weights(model, args.weights, device=args.device)

    # Only optimize non-frozen parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = optim.RMSprop(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc  = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs
        )
        acc, auc, f1, prec = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss:.4f} | "
              f"acc={acc:.4f} | auc={auc:.4f} | f1={f1:.4f}")

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(args.output_dir, f'best_sparse_step0.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model: acc={best_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',       default='/data/datasets/PIE')
    parser.add_argument('--weights',         default='pre_train_weights/min_loss_pretrained_model_imagenet.pth')
    parser.add_argument('--keypoints-dir',   default='/data/datasets/PIE/keypoints')
    parser.add_argument('--detector-config', default='change_detector_config.json')
    parser.add_argument('--output-dir',      default='weights_sparse')
    parser.add_argument('--epochs',          type=int,   default=50)
    parser.add_argument('--batch-size',      type=int,   default=32)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--weight-decay',    type=float, default=1e-4)
    parser.add_argument('--device',          default='cuda:0')
    args = parser.parse_args()
    main(args)
```

**Note on learning rate:** EfficientPIE uses `lr=1e-5` because it's fine-tuning the full backbone. Our backbone is frozen, so we can use a higher lr (`1e-4`) for the new attention and classifier layers. Tune if needed.

---

### 4.5 `pie_sparse_incremental_learning.py`

**Purpose:** IL training. Runs steps 2 → 14. At each step, loads previous step's best weights as `prev_model` (frozen), trains `curr_model` with the adaptive IL loss from EfficientPIE.

```python
"""
pie_sparse_incremental_learning.py
===================================
Incremental learning for SparseTemporalPIE.

IL step progression: 0 → 2 → 4 → 6 → 8 → 10 → 12 → 14

At each step `t`:
  - f_current  = frame at index t
  - f*         = last salient frame in [0, t] (or frame 0 if none)
  - absence_flag follows from detector

Adaptive IL loss (identical to EfficientPIE eq. 14):
  if L_new > L_old:  L_a = L_new
  else:              L_a = 0.5 * L_old + L_new

The backbone is frozen throughout ALL IL steps.
Only cross-attention + feedforward + classifier parameters are updated.
prev_model is a fully frozen copy of the previous step's best model.
"""

import argparse
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.SparseTemporalPIE import SparseTemporalPIE
from utils.sparse_dataset import SparseDataset
from utils.train_val import robust_noisy, evaluate


def il_loss(logits_new, logits_old, labels, epoch, total_epochs):
    """
    Adaptive IL loss from EfficientPIE (equation 14).
    Progressive perturbation applied to logits before loss computation.
    """
    criterion = nn.CrossEntropyLoss()

    logits_new_p = robust_noisy(logits_new, epoch, total_epochs=total_epochs)
    logits_old_p = robust_noisy(logits_old, epoch, total_epochs=total_epochs)

    loss_new = criterion(logits_new_p, labels)
    loss_old = criterion(logits_old_p, labels)

    if loss_new > loss_old:
        return loss_new
    else:
        return 0.5 * loss_old + loss_new


def train_il_step(curr_model, prev_model, loader, optimizer,
                  device, epoch, total_epochs):
    curr_model.train()
    prev_model.eval()  # Always eval — fully frozen

    total_loss = 0.0
    for f_current, f_star, flag, labels in loader:
        f_current = f_current.to(device)
        f_star    = f_star.to(device)
        flag      = flag.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()

        logits_new = curr_model(f_current, f_star, flag)
        with torch.no_grad():
            logits_old = prev_model(f_current, f_star, flag)

        loss = il_loss(logits_new, logits_old.detach(), labels, epoch, total_epochs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def main(args):
    device = torch.device(args.device)
    il_steps = [2, 4, 6, 8, 10, 12, 14]
    os.makedirs(args.output_dir, exist_ok=True)

    prev_weights = args.weights  # starts as step=0 best weights

    for step in il_steps:
        print(f"\n{'='*60}")
        print(f"  IL Step {step}")
        print(f"{'='*60}")

        # ── Load prev_model (frozen) ────────────────────────────────────
        prev_model = SparseTemporalPIE().to(device)
        prev_model.load_state_dict(torch.load(prev_weights, map_location=device))
        prev_model.eval()
        for p in prev_model.parameters():
            p.requires_grad = False

        # ── Load curr_model (warm start from prev) ──────────────────────
        curr_model = SparseTemporalPIE().to(device)
        curr_model.load_state_dict(torch.load(prev_weights, map_location=device))
        # backbone already frozen in model definition; just confirm
        for p in curr_model.backbone.parameters():
            p.requires_grad = False

        # ── Dataset for this step ───────────────────────────────────────
        data_opts = {'max_size_observe': 15, 'data_path': args.data_path}
        train_set = SparseDataset(
            data_opts=data_opts, preprocess_opts={}, split='train',
            step=step, keypoints_dir=args.keypoints_dir,
            detector_config=args.detector_config,
        )
        val_set = SparseDataset(
            data_opts=data_opts, preprocess_opts={}, split='test',
            step=step, keypoints_dir=args.keypoints_dir,
            detector_config=args.detector_config,
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=4)
        val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=4)

        # ── Optimizer ───────────────────────────────────────────────────
        trainable = [p for p in curr_model.parameters() if p.requires_grad]
        optimizer = optim.RMSprop(trainable, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_acc = 0.0
        best_path = os.path.join(args.output_dir, f'best_sparse_step{step}.pth')

        for epoch in range(1, args.epochs + 1):
            train_loss = train_il_step(
                curr_model, prev_model, train_loader,
                optimizer, device, epoch, args.epochs
            )
            acc, auc, f1, prec = evaluate(curr_model, val_loader, device)
            scheduler.step()

            print(f"  Step {step:2d} | Epoch {epoch:3d}/{args.epochs} | "
                  f"loss={train_loss:.4f} | acc={acc:.4f} | f1={f1:.4f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(curr_model.state_dict(), best_path)
                print(f"    ✓ Saved: acc={best_acc:.4f}")

        prev_weights = best_path  # next step starts from this step's best
        print(f"  Step {step} complete. Best acc: {best_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',       default='/data/datasets/PIE')
    parser.add_argument('--weights',         default='weights_sparse/best_sparse_step0.pth')
    parser.add_argument('--keypoints-dir',   default='/data/datasets/PIE/keypoints')
    parser.add_argument('--detector-config', default='change_detector_config.json')
    parser.add_argument('--output-dir',      default='weights_sparse')
    parser.add_argument('--epochs',          type=int,   default=30)
    parser.add_argument('--batch-size',      type=int,   default=32)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--weight-decay',    type=float, default=1e-4)
    parser.add_argument('--device',          default='cuda:0')
    args = parser.parse_args()
    main(args)
```

---

### 4.6 `test_SparseTemporalPIE.py`

**Purpose:** Evaluation script. Reports overall metrics AND v=0 subset metrics separately.

```python
"""
test_SparseTemporalPIE.py
=========================
Reports two sets of metrics:
  1. Overall     — full test set
  2. v=0 subset  — samples where bounding box center delta < ε across
                   the observation window (stationary pedestrians)

v=0 approximation:
  Compute the Euclidean distance between the bounding box center at
  frame 0 and frame `step` (last observed). Samples where this delta
  < ε (configurable, default=5 pixels normalized by bbox width)
  are included in the v=0 subset.
"""

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score

from models.SparseTemporalPIE import SparseTemporalPIE
from utils.sparse_dataset import SparseDataset


def compute_metrics(y_true, y_pred, y_prob):
    acc   = accuracy_score(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_prob)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    return acc, auc, f1, prec


def is_stationary(sample, epsilon=5.0):
    """
    Returns True if the pedestrian's bounding box center moved less than
    epsilon pixels (normalized by bbox width) between frame 0 and the
    last observed frame.
    """
    bboxes = sample.get('bboxes')  # list of [x, y, w, h] per frame
    if bboxes is None or len(bboxes) < 2:
        return False
    cx0 = bboxes[0][0] + bboxes[0][2] / 2
    cy0 = bboxes[0][1] + bboxes[0][3] / 2
    cxN = bboxes[-1][0] + bboxes[-1][2] / 2
    cyN = bboxes[-1][1] + bboxes[-1][3] / 2
    width = bboxes[0][2]
    if width < 1e-6:
        return False
    delta = np.sqrt((cxN - cx0)**2 + (cyN - cy0)**2) / width
    return delta < epsilon


@torch.no_grad()
def evaluate_split(model, dataset, device, step, epsilon=5.0):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    model.eval()

    all_true, all_pred, all_prob, stationary_mask = [], [], [], []

    for batch_idx, (f_current, f_star, flag, labels) in enumerate(loader):
        logits = model(
            f_current.to(device),
            f_star.to(device),
            flag.to(device)
        )
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_true.extend(labels.numpy())
        all_pred.extend(preds)
        all_prob.extend(probs)

        # Compute stationary mask for this batch
        start = batch_idx * loader.batch_size
        end   = start + len(labels)
        for i in range(start, min(end, len(dataset))):
            stationary_mask.append(is_stationary(dataset.samples[i], epsilon))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)
    stationary_mask = np.array(stationary_mask, dtype=bool)

    # Overall metrics
    overall = compute_metrics(all_true, all_pred, all_prob)

    # v=0 subset metrics
    if stationary_mask.sum() > 0:
        vzero = compute_metrics(
            all_true[stationary_mask],
            all_pred[stationary_mask],
            all_prob[stationary_mask],
        )
    else:
        vzero = (None, None, None, None)
        print("  Warning: no stationary samples found — check epsilon value")

    return overall, vzero, stationary_mask.sum()


def main(args):
    device = torch.device(args.device)

    model = SparseTemporalPIE().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    data_opts = {'max_size_observe': 15, 'data_path': args.data_path}
    test_set  = SparseDataset(
        data_opts=data_opts, preprocess_opts={}, split='test',
        step=14,  # always evaluate at final IL step
        keypoints_dir=args.keypoints_dir,
        detector_config=args.detector_config,
    )

    overall, vzero, n_stationary = evaluate_split(
        model, test_set, device, step=14, epsilon=args.epsilon
    )

    print("\n" + "="*55)
    print("  SparseTemporalPIE — Evaluation Results")
    print("="*55)
    print(f"  {'Metric':<12}  {'Overall':>10}  {'v=0 Subset':>12}")
    print(f"  {'-'*40}")
    labels = ['Accuracy', 'AUC', 'F1', 'Precision']
    for i, lbl in enumerate(labels):
        ov = f"{overall[i]:.4f}" if overall[i] is not None else "  N/A  "
        vz = f"{vzero[i]:.4f}"   if vzero[i]   is not None else "  N/A  "
        print(f"  {lbl:<12}  {ov:>10}  {vz:>12}")
    print(f"\n  v=0 subset size: {n_stationary} / {len(test_set)} samples")
    print("="*55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',       default='/data/datasets/PIE')
    parser.add_argument('--weights',         default='weights_sparse/best_sparse_step14.pth')
    parser.add_argument('--keypoints-dir',   default='/data/datasets/PIE/keypoints')
    parser.add_argument('--detector-config', default='change_detector_config.json')
    parser.add_argument('--epsilon',         type=float, default=5.0)
    parser.add_argument('--device',          default='cuda:0')
    args = parser.parse_args()
    main(args)
```

---

### 4.7 `run_sparse_pie_pipeline.sh`

```bash
#!/bin/bash
# Full SparseTemporalPIE training pipeline for PIE dataset
# Prerequisites:
#   1. PIE frames extracted:    python extract_frames.py --dataset pie
#   2. Keypoints extracted:     python extract_keypoints.py --dataset pie
#   3. Detector calibrated:     python calibrate_change_detector.py

set -e

DATA_PATH="/data/datasets/PIE"
KEYPOINTS_DIR="/data/datasets/PIE/keypoints"
DETECTOR_CFG="change_detector_config.json"
WEIGHTS_DIR="weights_sparse"
LOG_DIR="training_logs_sparse"
DEVICE="cuda:0"
BACKBONE_WEIGHTS="pre_train_weights/min_loss_pretrained_model_imagenet.pth"

mkdir -p $WEIGHTS_DIR $LOG_DIR

echo "========================================"
echo "  Step 0: Base training (50 epochs)"
echo "========================================"
python train_SparseTemporalPIE.py \
    --data-path       $DATA_PATH \
    --weights         $BACKBONE_WEIGHTS \
    --keypoints-dir   $KEYPOINTS_DIR \
    --detector-config $DETECTOR_CFG \
    --output-dir      $WEIGHTS_DIR \
    --epochs          50 \
    --device          $DEVICE \
    2>&1 | tee $LOG_DIR/step0.log

echo "========================================"
echo "  Steps 2–14: Incremental learning"
echo "========================================"
python pie_sparse_incremental_learning.py \
    --data-path       $DATA_PATH \
    --weights         $WEIGHTS_DIR/best_sparse_step0.pth \
    --keypoints-dir   $KEYPOINTS_DIR \
    --detector-config $DETECTOR_CFG \
    --output-dir      $WEIGHTS_DIR \
    --epochs          30 \
    --device          $DEVICE \
    2>&1 | tee $LOG_DIR/il_steps.log

echo "========================================"
echo "  Final Evaluation"
echo "========================================"
python test_SparseTemporalPIE.py \
    --data-path       $DATA_PATH \
    --weights         $WEIGHTS_DIR/best_sparse_step14.pth \
    --keypoints-dir   $KEYPOINTS_DIR \
    --detector-config $DETECTOR_CFG \
    --device          $DEVICE \
    2>&1 | tee $LOG_DIR/evaluation.log

echo "Pipeline complete. Results in $LOG_DIR/evaluation.log"
```

---

## 5. Training Protocol

### Prerequisites (in order)

```
1. Extract frames          → python extract_frames.py --dataset pie
2. Extract keypoints       → python extract_keypoints.py --dataset pie
3. Calibrate detector      → python calibrate_change_detector.py
4. Run training pipeline   → bash run_sparse_pie_pipeline.sh
```

### Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Backbone lr | frozen | No backbone updates |
| New layers lr | 1e-4 | Higher than EfficientPIE (backbone frozen) |
| Weight decay | 1e-4 | Same as EfficientPIE |
| Optimizer | RMSProp | Same as EfficientPIE |
| Scheduler | Cosine annealing | T_max = epochs for each phase |
| Base training epochs | 50 | Same as EfficientPIE |
| IL epochs per step | 30 | Same as EfficientPIE |
| Batch size | 32 | Same as EfficientPIE |
| Attention heads | 8 | Over 1280-dim embedding |
| FF hidden dim | 512 | 1280 → 512 → 1280 |
| Dropout | 0.1 | Attention + feedforward |

### IL Step Schedule

| IL Step | Observation frames available | f* search window |
|---|---|---|
| 0  | [0]          | Must use frame 0; flag always 1 |
| 2  | [0, 1, 2]    | f* ∈ {0, 1, 2} |
| 4  | [0..4]       | f* ∈ {0..4} |
| 6  | [0..6]       | f* ∈ {0..6} |
| 8  | [0..8]       | f* ∈ {0..8} |
| 10 | [0..10]      | f* ∈ {0..10} |
| 12 | [0..12]      | f* ∈ {0..12} |
| 14 | [0..14]      | f* ∈ {0..14} (full window) |

---

## 6. Evaluation Protocol

### Metrics

| Metric | Reported for |
|---|---|
| Accuracy | Overall + v=0 subset |
| AUC-ROC | Overall + v=0 subset |
| F1 Score | Overall + v=0 subset |
| Precision | Overall + v=0 subset |

### v=0 Subset Definition

Samples where the pedestrian bounding box center displacement across the observation window is less than ε = 5.0 pixels normalized by bounding box width. Computed at evaluation time from PIE bounding box annotations — no manual labeling required.

### Cross-dataset Generalization

Evaluate the PIE-trained model on JAAD without any JAAD fine-tuning to test generalization. Report separately in the paper.

---

## 7. Ablation Study Design

Four ablation rows, each removes one component:

| Ablation | What changes | Expected effect |
|---|---|---|
| **A1** — No temporal sampling | f* = f_current always; flag always 0 | Degrades to EfficientPIE-equivalent (measures value of temporal mechanism) |
| **A2** — No absence flag | Remove flag from classifier input | Measures disambiguation value of explicit inertia signal |
| **A3** — Zeros fallback | f* = zeros when absent (original proposal) | Measures information value of first-frame anchor |
| **A4** — ViTPose features added | Concatenate 34-dim pose to classifier input | Measures pose feature contribution beyond change detection |
| **A5** — No hard gate | Remove `attn_out × gate`; let network learn suppression implicitly | Measures whether explicit gating helps vs. relying on the network to ignore attention when f* = frame 0 |

---

## 8. Expected Results & Baselines

### Comparison Table (target)

| Model | Accuracy | AUC | F1 | Precision | v=0 Acc |
|---|---|---|---|---|---|
| PIE RNN baseline | 0.79 | — | — | — | — |
| EfficientPIE | 0.92 | 0.92 | 0.95 | 0.96 | TBD |
| TCL (arXiv 2504.06292) | SOTA | — | — | — | — |
| **SparseTemporalPIE (ours)** | **>0.92** | — | — | — | **best** |

The primary claim is not necessarily beating EfficientPIE overall — it is that our v=0 subset accuracy is meaningfully higher, which is the safety-critical scenario we target.

### Inference Time Budget

| Component | Estimate |
|---|---|
| EfficientPIE backbone (×2, shared) | ~0.42ms (2× EfficientPIE's 0.21ms) |
| Cross-attention (1280-dim, 8 heads) | ~0.05ms |
| Feedforward (1280→512→1280) | ~0.02ms |
| **Total** | **~0.50ms** |

This is the floor estimate; actual overhead slightly higher. Justified for safety-critical stationary pedestrian prediction.

---

## 9. Key Implementation Notes

**On `_load_samples()` in `SparseDataset`:** This is the most critical implementation task. Follow `my_dataset.py` exactly, but store all 15 frame paths per sample instead of just one. The PIE API call pattern and label key (`intention_binary`) are identical.

**On ViTPose integration:** Use `timm` or MMPose. ViTPose-B is sufficient. The keypoint extraction is offline — `change_detector.py` reads `.npy` files at training time and has no real-time ViTPose dependency. This keeps training fast and avoids GPU memory conflicts.

**On the `evaluate()` function:** The existing `evaluate()` in `utils/train_val.py` expects `(images, labels)` batches. You will need to adapt it or write a thin wrapper that calls `model(f_current, f_star, flag)` instead of `model(images)`.

**On shared backbone weights:** Both `f_current` and `f*` pass through `self.backbone` which is the same `nn.Sequential` object — PyTorch handles this correctly. Gradients are blocked by `requires_grad=False` on all backbone parameters, not by `torch.no_grad()`, so you don't need context managers around the encode calls.

**On the absence flag at step=0:** At IL step 0 there is only one frame, so the change detector has nothing to compare. The absence flag is always 1 and f* is always frame 0 (= f_current). This means the cross-attention at step 0 sees identical queries and keys — the model learns a degenerate but valid single-frame representation identical in spirit to EfficientPIE. Subsequent IL steps incrementally teach it to use temporal context.