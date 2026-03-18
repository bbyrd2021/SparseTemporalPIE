"""
SparseDataset v4
================
Returns 5-tuple per sample:
  (f_current, pose_current, bbox_traj, ctx_feats, label)

Simplified from v3:
  - Removed f_context, context_mask, pose_context (no cross-attention)
  - Pose back to 34-d static only (velocity dropped)
  - bbox_traj (12-d) and ctx_feats (5-d) retained as temporal context
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# COCO 17-joint left/right pairs (0-indexed)
_FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def flip_pose(feats):
    """Mirror a 34-d normalized pose vector horizontally."""
    flipped = feats.copy()
    for k in range(17):
        if flipped[2 * k] != 0.0 or flipped[2 * k + 1] != 0.0:
            flipped[2 * k] = 1.0 - flipped[2 * k]
    for l, r in _FLIP_PAIRS:
        flipped[2 * l], flipped[2 * r] = flipped[2 * r], flipped[2 * l]
        flipped[2 * l + 1], flipped[2 * r + 1] = flipped[2 * r + 1], flipped[2 * l + 1]
    return flipped


def normalize_pose(kpts, bbox, conf_thresh=0.25):
    """Normalize 17-keypoint array to a 34-dim float32 vector."""
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    feats = np.zeros(34, dtype=np.float32)
    for k in range(17):
        x, y, c = kpts[k]
        if c >= conf_thresh:
            feats[k * 2]     = (x - x1) / bw
            feats[k * 2 + 1] = (y - y1) / bh
    return feats


class SparseDataset(Dataset):
    def __init__(
        self,
        images_seq,
        data_opts,
        step=0,
        transform=None,
        keypoints_dir=None,
        flip_p=0.0,
        pose_dropout_p=0.0,
    ):
        self.images_seq      = images_seq
        self.data_opts       = data_opts
        self.step            = step
        self.keypoints_dir   = keypoints_dir
        self.flip_p          = flip_p
        self.pose_dropout_p  = pose_dropout_p

        self.transform = transform or transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.images_seq['images'])

    # ------------------------------------------------------------------

    def _crop_image(self, img, bbox):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        half = 150
        new_x1 = max(0, int(center_x - half))
        new_y1 = max(0, int(center_y - half))
        new_x2 = min(img.width,  int(center_x + half))
        new_y2 = min(img.height, int(center_y + half))
        return img.crop([new_x1, new_y1, new_x2, new_y2])

    def _load_pose_feats(self, pid, frame_path, bbox):
        if self.keypoints_dir is None:
            return np.zeros(34, dtype=np.float32)
        frame_id  = int(os.path.splitext(os.path.basename(frame_path))[0])
        kpts_path = os.path.join(self.keypoints_dir, pid, f"{frame_id:05d}.npy")
        if not os.path.exists(kpts_path):
            return np.zeros(34, dtype=np.float32)
        kpts = np.load(kpts_path)
        return normalize_pose(kpts, bbox)

    def _compute_bbox_trajectory(self, bboxes):
        """Compute 12-d bbox trajectory statistics over [0, step]."""
        if self.step == 0:
            return np.zeros(12, dtype=np.float32)

        seq = bboxes[:self.step + 1]
        centers = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in seq], dtype=np.float32)
        widths  = np.array([max(b[2]-b[0], 1.0) for b in seq], dtype=np.float32)
        heights = np.array([max(b[3]-b[1], 1.0) for b in seq], dtype=np.float32)

        w0, h0 = widths[0], heights[0]
        norm_centers = centers.copy()
        norm_centers[:, 0] = (centers[:, 0] - centers[0, 0]) / w0
        norm_centers[:, 1] = (centers[:, 1] - centers[0, 1]) / h0

        disp_last = norm_centers[-1]
        disp_mean = norm_centers.mean(axis=0)

        if len(norm_centers) >= 2:
            vel = np.diff(norm_centers, axis=0)
            vel_mean = vel.mean(axis=0)
            vel_std  = vel.std(axis=0)
        else:
            vel_mean = np.zeros(2, dtype=np.float32)
            vel_std  = np.zeros(2, dtype=np.float32)

        if len(norm_centers) >= 3:
            vel = np.diff(norm_centers, axis=0)
            accel_mean = np.diff(vel, axis=0).mean(axis=0)
        else:
            accel_mean = np.zeros(2, dtype=np.float32)

        areas = widths * heights
        size_ratio_last = areas[-1] / areas[0]
        size_ratio_mean = areas.mean() / areas[0]

        return np.array([
            disp_last[0], disp_last[1],
            disp_mean[0], disp_mean[1],
            vel_mean[0],  vel_mean[1],
            vel_std[0],   vel_std[1],
            accel_mean[0], accel_mean[1],
            size_ratio_last, size_ratio_mean,
        ], dtype=np.float32)

    def _get_context_features(self, index):
        """Return 5-d context features: [speed, speed_mean, speed_valid, action, look]."""
        feats = np.zeros(5, dtype=np.float32)

        speeds = self.images_seq.get('obd_speed', None)
        if speeds is not None and index < len(speeds):
            seq = speeds[index]
            feats[0] = float(seq[self.step][0])
            feats[1] = np.mean([float(s[0]) for s in seq[:self.step + 1]])
            feats[2] = 1.0

        actions = self.images_seq.get('action', None)
        if actions is not None and index < len(actions):
            feats[3] = float(actions[index][self.step][0])

        looks = self.images_seq.get('look', None)
        if looks is not None and index < len(looks):
            feats[4] = float(looks[index][self.step][0])

        return feats

    # ------------------------------------------------------------------

    def __getitem__(self, index):
        frame_paths = self.images_seq['images'][index]
        bboxes      = self.images_seq['bboxes'][index]
        output      = self.images_seq['output'][index]
        pid         = self.images_seq['ped_ids'][index][0][0]

        label = output[self.step][0]

        pose_current = self._load_pose_feats(pid, frame_paths[self.step], bboxes[self.step])
        bbox_traj    = self._compute_bbox_trajectory(bboxes)
        ctx_feats    = self._get_context_features(index)

        img = Image.open(frame_paths[self.step]).convert('RGB')
        img = self._crop_image(img, bboxes[self.step])

        if self.pose_dropout_p > 0 and random.random() < self.pose_dropout_p:
            pose_current = np.zeros(34, dtype=np.float32)

        if self.flip_p > 0 and random.random() < self.flip_p:
            img          = img.transpose(Image.FLIP_LEFT_RIGHT)
            pose_current = flip_pose(pose_current)

        f_current_t  = self.transform(img)
        pose_t       = torch.tensor(pose_current, dtype=torch.float32)
        bbox_traj_t  = torch.tensor(bbox_traj,    dtype=torch.float32)
        ctx_feats_t  = torch.tensor(ctx_feats,    dtype=torch.float32)
        label_t      = torch.tensor(label,        dtype=torch.long)

        return f_current_t, pose_t, bbox_traj_t, ctx_feats_t, label_t

    @staticmethod
    def collate_fn(batch):
        f_currents, poses, bbox_trajs, ctx_feats, labels = zip(*batch)
        return (
            torch.stack(f_currents),
            torch.stack(poses),
            torch.stack(bbox_trajs),
            torch.stack(ctx_feats),
            torch.stack(labels),
        )
