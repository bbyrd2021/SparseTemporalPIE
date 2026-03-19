"""
SparseDataset v3 (reconstructed for evaluation)
================================================
Returns 8-tuple per sample:
  (f_current, f_context, context_mask, pose_current, pose_context, bbox_traj, ctx_feats, label)

  - f_context:    (K, 3, 300, 300) — K=4 evenly-spaced context frames from [0, step-1]
  - context_mask: (K,)             — 1.0=real, 0.0=padding
  - pose_current: (68,)            — 34 static + 34 velocity keypoints
  - pose_context: (K, 68)          — pose for each context frame
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
N_CONTEXT = 4


def flip_pose(feats):
    """Mirror a 34-d normalized pose vector horizontally (works for 68-d too, first 34)."""
    flipped = feats.copy()
    n = len(flipped) // 2   # 34 for 68-d, 17 for 34-d
    n_joints = n // 2       # 17
    for k in range(n_joints):
        if flipped[2 * k] != 0.0 or flipped[2 * k + 1] != 0.0:
            flipped[2 * k] = 1.0 - flipped[2 * k]
    for l, r in _FLIP_PAIRS:
        flipped[2 * l], flipped[2 * r] = flipped[2 * r], flipped[2 * l]
        flipped[2 * l + 1], flipped[2 * r + 1] = flipped[2 * r + 1], flipped[2 * l + 1]
    return flipped


def normalize_pose(kpts, bbox, conf_thresh=0.25):
    """Normalize 17-keypoint array to 34-d float32 vector."""
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


class SparseDataset_v3(Dataset):
    def __init__(
        self,
        images_seq,
        data_opts,
        step=0,
        transform=None,
        keypoints_dir=None,
        flip_p=0.0,
        pose_dropout_p=0.0,
        n_context=N_CONTEXT,
    ):
        self.images_seq     = images_seq
        self.data_opts      = data_opts
        self.step           = step
        self.keypoints_dir  = keypoints_dir
        self.flip_p         = flip_p
        self.pose_dropout_p = pose_dropout_p
        self.n_context      = n_context

        self.transform = transform or transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.images_seq['images'])

    def _crop_image(self, img, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        half = 150
        return img.crop([
            max(0, int(cx - half)), max(0, int(cy - half)),
            min(img.width,  int(cx + half)), min(img.height, int(cy + half)),
        ])

    def _load_pose_static(self, pid, frame_path, bbox):
        if self.keypoints_dir is None:
            return np.zeros(34, dtype=np.float32)
        frame_id  = int(os.path.splitext(os.path.basename(frame_path))[0])
        kpts_path = os.path.join(self.keypoints_dir, pid, f"{frame_id:05d}.npy")
        if not os.path.exists(kpts_path):
            return np.zeros(34, dtype=np.float32)
        return normalize_pose(np.load(kpts_path), bbox)

    def _load_pose_68d(self, pid, frame_paths, bboxes, idx):
        """Load 68-d pose: 34-d static + 34-d velocity (delta vs previous frame)."""
        static = self._load_pose_static(pid, frame_paths[idx], bboxes[idx])
        if idx > 0:
            prev = self._load_pose_static(pid, frame_paths[idx - 1], bboxes[idx - 1])
        else:
            prev = static
        velocity = static - prev
        return np.concatenate([static, velocity], axis=0).astype(np.float32)

    def _select_context_indices(self):
        """Return list of at most n_context evenly-spaced indices from [0, step-1]."""
        if self.step == 0:
            return [0]
        n = min(self.n_context, self.step)
        return np.linspace(0, self.step - 1, n, dtype=int).tolist()

    def _compute_bbox_trajectory(self, bboxes):
        """12-d bbox trajectory statistics over [0, step]."""
        if self.step == 0:
            return np.zeros(12, dtype=np.float32)
        seq = bboxes[:self.step + 1]
        centers = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in seq], dtype=np.float32)
        widths  = np.array([max(b[2]-b[0], 1.0) for b in seq], dtype=np.float32)
        heights = np.array([max(b[3]-b[1], 1.0) for b in seq], dtype=np.float32)
        w0, h0 = widths[0], heights[0]
        nc = centers.copy()
        nc[:, 0] = (centers[:, 0] - centers[0, 0]) / w0
        nc[:, 1] = (centers[:, 1] - centers[0, 1]) / h0
        disp_last = nc[-1]
        disp_mean = nc.mean(axis=0)
        if len(nc) >= 2:
            vel = np.diff(nc, axis=0)
            vel_mean, vel_std = vel.mean(axis=0), vel.std(axis=0)
        else:
            vel_mean = vel_std = np.zeros(2, dtype=np.float32)
        if len(nc) >= 3:
            vel = np.diff(nc, axis=0)
            accel_mean = np.diff(vel, axis=0).mean(axis=0)
        else:
            accel_mean = np.zeros(2, dtype=np.float32)
        areas = widths * heights
        return np.array([
            disp_last[0], disp_last[1], disp_mean[0], disp_mean[1],
            vel_mean[0],  vel_mean[1],  vel_std[0],   vel_std[1],
            accel_mean[0], accel_mean[1],
            areas[-1] / areas[0], areas.mean() / areas[0],
        ], dtype=np.float32)

    def _get_context_features(self, index):
        """5-d: [speed, speed_mean, speed_valid, action, look]."""
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

    def __getitem__(self, index):
        frame_paths = self.images_seq['images'][index]
        bboxes      = self.images_seq['bboxes'][index]
        output      = self.images_seq['output'][index]
        pid         = self.images_seq['ped_ids'][index][0][0]

        label = output[self.step][0]

        do_flip = self.flip_p > 0 and random.random() < self.flip_p

        # Current frame
        img_curr = Image.open(frame_paths[self.step]).convert('RGB')
        img_curr = self._crop_image(img_curr, bboxes[self.step])
        if do_flip:
            img_curr = img_curr.transpose(Image.FLIP_LEFT_RIGHT)
        f_current_t = self.transform(img_curr)

        # Current pose (68-d)
        pose_curr = self._load_pose_68d(pid, frame_paths, bboxes, self.step)
        if self.pose_dropout_p > 0 and random.random() < self.pose_dropout_p:
            pose_curr = np.zeros(68, dtype=np.float32)
        elif do_flip:
            pose_curr[:34] = flip_pose(pose_curr[:34])
            pose_curr[34:] = flip_pose(pose_curr[34:])

        # Context frames
        ctx_indices = self._select_context_indices()
        ctx_frames  = []
        ctx_poses   = []
        for ci in ctx_indices:
            img_c = Image.open(frame_paths[ci]).convert('RGB')
            img_c = self._crop_image(img_c, bboxes[ci])
            if do_flip:
                img_c = img_c.transpose(Image.FLIP_LEFT_RIGHT)
            ctx_frames.append(self.transform(img_c))

            p = self._load_pose_68d(pid, frame_paths, bboxes, ci)
            if do_flip:
                p[:34] = flip_pose(p[:34])
                p[34:] = flip_pose(p[34:])
            ctx_poses.append(torch.tensor(p, dtype=torch.float32))

        # Pad to n_context
        K = self.n_context
        real_k = len(ctx_frames)
        mask = torch.zeros(K, dtype=torch.float32)
        mask[:real_k] = 1.0

        pad_frame = torch.zeros_like(ctx_frames[0])
        pad_pose  = torch.zeros(68, dtype=torch.float32)
        while len(ctx_frames) < K:
            ctx_frames.append(pad_frame)
            ctx_poses.append(pad_pose)

        f_context_t  = torch.stack(ctx_frames)                   # (K, 3, 300, 300)
        pose_ctx_t   = torch.stack(ctx_poses)                    # (K, 68)
        pose_curr_t  = torch.tensor(pose_curr, dtype=torch.float32)
        bbox_traj_t  = torch.tensor(self._compute_bbox_trajectory(bboxes), dtype=torch.float32)
        ctx_feats_t  = torch.tensor(self._get_context_features(index), dtype=torch.float32)
        label_t      = torch.tensor(label, dtype=torch.long)

        return f_current_t, f_context_t, mask, pose_curr_t, pose_ctx_t, bbox_traj_t, ctx_feats_t, label_t

    @staticmethod
    def collate_fn(batch):
        f_curr, f_ctx, masks, p_curr, p_ctx, bbox_trajs, ctx_feats, labels = zip(*batch)
        return (
            torch.stack(f_curr),
            torch.stack(f_ctx),
            torch.stack(masks),
            torch.stack(p_curr),
            torch.stack(p_ctx),
            torch.stack(bbox_trajs),
            torch.stack(ctx_feats),
            torch.stack(labels),
        )
