"""
calibrate_change_detector.py
=============================
Calibrates ChangeDetector thresholds against PIE crossing labels.

Loads pre-computed keypoints from extract_keypoints.py and runs a grid
search over threshold combinations to maximise AUC against
intention_binary labels from the PIE API.

Keypoint path convention (matches extract_keypoints.py output):
  {keypoints_dir}/{pid}/{t:05d}.npy

Saves best thresholds to change_detector_config.json.

Usage:
    python calibrate_change_detector.py \
        --data-path /data/datasets/PIE \
        --keypoints-dir /data/datasets/PIE/keypoints_pid \
        --output change_detector_config.json
"""
import sys, os as _os; sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))

import argparse
import os
import numpy as np
from tqdm import tqdm

from utils.pie_data import PIE
from utils.change_detector import ChangeDetector


def load_kpts(pid, img_path, keypoints_dir):
    """Load keypoints for pedestrian pid at the given frame (by image path)."""
    frame_id = os.path.splitext(os.path.basename(img_path))[0]
    path = os.path.join(keypoints_dir, pid, f"{frame_id}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def main(args):
    data_opts = {
        'fstride': 1, 'sample_type': 'all', 'height_rng': [0, float('inf')],
        'squarify_ratio': 0, 'data_split_type': 'random', 'seq_type': 'intention',
        'min_track_size': 0, 'max_size_observe': 15, 'seq_overlap_rate': 0,
        'balance': False, 'crop_type': 'context', 'crop_mode': 'pad_resize',
        'encoder_input_type': [], 'decoder_input_type': ['bbox'],
        'output_type': ['intention_binary'],
    }
    data_type = {
        'encoder_input_type': [], 'decoder_input_type': ['bbox'],
        'output_type': ['intention_binary'],
    }

    print("Loading PIE annotations...")
    pie = PIE(data_path=args.data_path)
    train_seq  = pie.generate_data_trajectory_sequence('train', **data_opts)
    seq_len    = data_opts['max_size_observe']
    train_data = pie.get_train_val_data(train_seq, data_type, seq_len, 0)

    print("Building keypoint sequences for calibration...")
    keypoints_by_pid = {}
    labels_by_pid    = {}

    images  = train_data['images']   # list of seqs, each: list of 15 img paths
    bboxes  = train_data['bboxes']   # list of seqs, each: list of 15 [x1,y1,x2,y2]
    output  = train_data['output']   # list of seqs, each: list of 15 [label]
    ped_ids = train_data['ped_ids']  # list of seqs, each: list of 15 [['pid']]

    for i in tqdm(range(len(bboxes)), desc='Loading keypoints'):
        pid   = ped_ids[i][0][0]
        label = output[i][-1][0]

        kpts_seq = [load_kpts(pid, images[i][t], args.keypoints_dir)
                    for t in range(seq_len)]

        if all(k is None for k in kpts_seq):
            continue   # keypoints not yet extracted for this track — skip

        # bbox heights for body lean normalization (y2 - y1 in full-frame pixels)
        bbox_heights = [float(bboxes[i][t][3] - bboxes[i][t][1])
                        for t in range(seq_len)]

        # Use pid + index as key to handle any duplicate pids across sequences
        key = f"{pid}_{i}"
        keypoints_by_pid[key] = (kpts_seq, bbox_heights)
        labels_by_pid[key]    = label

    print(f"Calibrating on {len(keypoints_by_pid)} tracks "
          f"({sum(labels_by_pid.values())} crossing, "
          f"{len(labels_by_pid) - sum(labels_by_pid.values())} not crossing)...")

    # Wrap calibrate() to pass bbox_heights through
    detector = ChangeDetector()

    # Build plain dicts for calibrate() — it expects {key: kpts_seq}
    kpts_only    = {k: v[0] for k, v in keypoints_by_pid.items()}
    heights_only = {k: v[1] for k, v in keypoints_by_pid.items()}

    best_cfg = detector.calibrate(
        keypoints_by_pid=kpts_only,
        labels_by_pid=labels_by_pid,
        bbox_heights_by_pid=heights_only,
        output_path=args.output,
    )

    print(f"\nCalibration complete. Config saved to: {args.output}")
    print(f"Best config: {best_cfg}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',      default='/data/datasets/PIE')
    parser.add_argument('--keypoints-dir',  default='/data/datasets/PIE/keypoints_pid')
    parser.add_argument('--output',         default='change_detector_config.json')
    args = parser.parse_args()
    main(args)
