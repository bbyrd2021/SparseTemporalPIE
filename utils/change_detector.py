"""
ChangeDetector
==============
Monitors three pose geometry signals per frame:
  1. Head orientation delta  -- yaw proxy from nose-to-eye-midpoint vector
  2. Body lean angle         -- shoulder midpoint vs. hip midpoint vertical delta
  3. Gaze vector change      -- nose-to-eye-midpoint direction shift (same proxy)

Thresholds are calibrated offline against PIE behavioral annotations and
loaded from a JSON config file. Default values are reasonable starting points.

Usage:
    detector = ChangeDetector(config_path='change_detector_config.json')
    f_star_idx, fired = detector.detect(keypoints_seq, bbox_heights)
"""

import json
import numpy as np
from pathlib import Path


DEFAULT_CONFIG = {
    "head_orient_delta_threshold": 0.15,   # cosine distance units
    "body_lean_delta_threshold":   0.08,   # normalized pixel units
    "gaze_vector_delta_threshold": 0.12,   # cosine distance units
    "min_confidence":              0.3,    # ViTPose keypoint confidence floor
    "rolling_window":              3,      # frames to smooth signal over
}

# COCO 17-point keypoint indices
NOSE          = 0
LEFT_EYE      = 1
RIGHT_EYE     = 2
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP      = 11
RIGHT_HIP     = 12


class ChangeDetector:
    def __init__(self, config_path=None):
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.cfg = json.load(f)
        else:
            self.cfg = DEFAULT_CONFIG.copy()

        self.delta_head  = self.cfg['head_orient_delta_threshold']
        self.delta_lean  = self.cfg['body_lean_delta_threshold']
        self.delta_gaze  = self.cfg['gaze_vector_delta_threshold']
        self.min_conf    = self.cfg['min_confidence']
        self.window      = self.cfg['rolling_window']

    # -- Signal extractors ---------------------------------------------------

    def _head_orient_vector(self, kpts):
        """
        Proxy for head yaw: unit vector from nose to midpoint of eyes.
        kpts: (17, 3) array -- [x, y, confidence]
        Returns unit vector (2,) or None if keypoints not confident.
        """
        nose = kpts[NOSE]
        leye = kpts[LEFT_EYE]
        reye = kpts[RIGHT_EYE]
        if any(p[2] < self.min_conf for p in [nose, leye, reye]):
            return None
        eye_mid = (leye[:2] + reye[:2]) / 2.0
        v = eye_mid - nose[:2]
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-6 else None

    def _body_lean_angle(self, kpts, bbox_height):
        """
        Shoulder-midpoint vertical position relative to hip-midpoint,
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
        """Same proxy as head_orient_vector."""
        return self._head_orient_vector(kpts)

    @staticmethod
    def _cosine_dist(v1, v2):
        if v1 is None or v2 is None:
            return 0.0
        return 1.0 - float(np.dot(v1, v2))

    def _rolling_mean(self, values, t):
        """Mean of non-None values in [max(0, t-window), t)."""
        start = max(0, t - self.window)
        valid = [v for v in values[start:t] if v is not None]
        return np.mean(valid, axis=0) if valid else None

    # -- Main detection ------------------------------------------------------

    def detect(self, keypoints_seq, bbox_heights=None):
        """
        Scan a sequence of keypoint arrays and return the index of the
        last frame where any signal exceeded its threshold.

        Args:
            keypoints_seq: list of (17, 3) numpy arrays, one per frame,
                           or None if ViTPose failed on that frame
            bbox_heights:  list of floats (pedestrian bbox height in pixels)
                           used to normalize body lean; if None, lean is skipped

        Returns:
            (f_star_idx: int or None, fired: bool)
        """
        n = len(keypoints_seq)
        if n == 0:
            return None, False

        head_vecs = []
        lean_vals = []
        gaze_vecs = []

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

            if (head_delta >= self.delta_head or
                    lean_delta >= self.delta_lean or
                    gaze_delta >= self.delta_gaze):
                f_star_idx = t

        fired = f_star_idx is not None
        return f_star_idx, fired

    # -- Calibration ---------------------------------------------------------

    def calibrate(self, keypoints_by_pid, labels_by_pid,
                  bbox_heights_by_pid=None, output_path=None):
        """
        Grid search over threshold combinations to maximize AUC against
        crossing labels. Saves best config as JSON if output_path given.

        Args:
            keypoints_by_pid:   dict {pid: list of (17,3) arrays}
            labels_by_pid:      dict {pid: int} -- 1=crossing, 0=not crossing
            bbox_heights_by_pid: dict {pid: list of floats} -- pedestrian bbox
                                 heights per frame for body lean normalization.
                                 If None, body lean signal is skipped.
            output_path:        path to save best JSON config

        Returns:
            best_config: dict
        """
        import itertools
        from sklearn.metrics import roc_auc_score

        delta_head_range = np.arange(0.05, 0.35, 0.05)
        delta_lean_range = np.arange(0.03, 0.20, 0.03)
        delta_gaze_range = np.arange(0.05, 0.35, 0.05)

        best_auc = 0.0
        best_cfg = self.cfg.copy()

        for dh, dl, dg in itertools.product(delta_head_range, delta_lean_range, delta_gaze_range):
            self.delta_head = dh
            self.delta_lean = dl
            self.delta_gaze = dg

            y_true, y_score = [], []
            for pid, kpts_seq in keypoints_by_pid.items():
                heights = bbox_heights_by_pid[pid] if bbox_heights_by_pid else None
                _, fired = self.detect(kpts_seq, bbox_heights=heights)
                y_true.append(labels_by_pid[pid])
                y_score.append(1.0 if fired else 0.0)

            if len(set(y_true)) < 2:
                continue
            auc = roc_auc_score(y_true, y_score)
            if auc > best_auc:
                best_auc = auc
                best_cfg = {
                    **self.cfg,
                    'head_orient_delta_threshold': float(dh),
                    'body_lean_delta_threshold':   float(dl),
                    'gaze_vector_delta_threshold': float(dg),
                }

        print(f"Best calibration AUC: {best_auc:.4f}")
        print(f"Best config: {best_cfg}")

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(best_cfg, f, indent=2)

        self.delta_head = best_cfg['head_orient_delta_threshold']
        self.delta_lean = best_cfg['body_lean_delta_threshold']
        self.delta_gaze = best_cfg['gaze_vector_delta_threshold']
        return best_cfg
