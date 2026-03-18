"""
extract_keypoints.py
====================
Offline ViTPose-B keypoint extraction for PIE/JAAD datasets.

Extracts keypoints for each annotated pedestrian track, using the PIE/JAAD
bounding box annotations to guide ViTPose-B to the correct person.

Output format:
  {output_dir}/{pid}/{t:05d}.npy  — (17, 3) float32 [x, y, conf]
                                    coordinates are in full-frame pixel space
                                    t is the observation window index (0–14)

This matches the path convention expected by utils/sparse_dataset.py.

Skips files that already exist (safe to resume after interruption).

Usage:
    python extract_keypoints.py --dataset pie --data-path /data/datasets/PIE
    python extract_keypoints.py --dataset jaad --data-path /data/datasets/JAAD
"""

import argparse
import os
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor


VITPOSE_MODEL = 'usyd-community/vitpose-base-coco-aic-mpii'

DATA_OPTS = {
    'fstride': 1, 'sample_type': 'all', 'height_rng': [0, float('inf')],
    'squarify_ratio': 0, 'data_split_type': 'random', 'seq_type': 'intention',
    'min_track_size': 0, 'max_size_observe': 15, 'seq_overlap_rate': 0,
    'balance': False, 'crop_type': 'context', 'crop_mode': 'pad_resize',
    'encoder_input_type': [], 'decoder_input_type': ['bbox'],
    'output_type': ['intention_binary'],
}
DATA_TYPE = {
    'encoder_input_type': [], 'decoder_input_type': ['bbox'],
    'output_type': ['intention_binary'],
}
SEQ_LEN = 15


def frame_id_from_path(img_path):
    """Extract the frame number string from an image path.
    '/data/datasets/PIE/images/set01/video_0001/01025.png' → '01025'
    """
    return os.path.splitext(os.path.basename(img_path))[0]


def collect_pie_samples(data_path):
    """
    Use the PIE API to collect all annotated pedestrian tracks across all splits.
    Returns list of dicts: {'pid': str, 'frame_id': str, 'img_path': str, 'bbox': [x1,y1,x2,y2]}
    Deduplicates by (pid, frame_id) — the absolute frame number, not the
    window index — so the same pedestrian in multiple sequences is handled correctly.
    """
    from utils.pie_data import PIE
    pie = PIE(data_path=data_path)

    seen = set()
    samples = []

    for split in ['train', 'val', 'test']:
        seq  = pie.generate_data_trajectory_sequence(split, **DATA_OPTS)
        data = pie.get_train_val_data(seq, DATA_TYPE, SEQ_LEN, 0)

        images  = data['images']
        bboxes  = data['bboxes']
        ped_ids = data['ped_ids']

        for i in range(len(images)):
            pid = ped_ids[i][0][0]
            for t in range(SEQ_LEN):
                fid = frame_id_from_path(images[i][t])
                key = (pid, fid)
                if key in seen:
                    continue
                seen.add(key)
                samples.append({
                    'pid':      pid,
                    'frame_id': fid,
                    'img_path': images[i][t],
                    'bbox':     bboxes[i][t],
                })

    return samples


def collect_jaad_samples(data_path):
    """
    Use the JAAD API to collect all annotated pedestrian tracks across all splits.
    Returns list of dicts: {'pid': str, 'frame_id': str, 'img_path': str, 'bbox': [x1,y1,x2,y2]}
    """
    from utils.jaad_data import JAAD
    jaad = JAAD(data_path=data_path)

    seen = set()
    samples = []

    for split in ['train', 'val', 'test']:
        seq  = jaad.generate_data_trajectory_sequence(split, **DATA_OPTS)
        data = jaad.get_train_val_data(seq, DATA_TYPE, SEQ_LEN, 0)

        images  = data['images']
        bboxes  = data['bboxes']
        ped_ids = data['ped_ids']

        for i in range(len(images)):
            pid = ped_ids[i][0][0]
            for t in range(SEQ_LEN):
                fid = frame_id_from_path(images[i][t])
                key = (pid, fid)
                if key in seen:
                    continue
                seen.add(key)
                samples.append({
                    'pid':      pid,
                    'frame_id': fid,
                    'img_path': images[i][t],
                    'bbox':     bboxes[i][t],
                })

    return samples


def extract_keypoints_for_bbox(img_path, bbox, model, processor, device):
    """
    Run ViTPose-B on a pedestrian crop from img_path.

    Manually crops the image to the pedestrian bbox (with padding), runs
    ViTPose on the crop, then remaps keypoint coordinates back to
    full-frame pixel space.

    Args:
        img_path: path to full-frame image
        bbox:     [x1, y1, x2, y2] in full-frame pixel coordinates
        model, processor: loaded ViTPose model/processor
        device:   torch device

    Returns:
        (17, 3) float32 array [x, y, conf] in full-frame pixel coordinates.
        Returns zeros if inference fails or bbox is degenerate.
    """
    try:
        image = Image.open(img_path).convert('RGB')
        W, H  = image.size

        x1, y1, x2, y2 = [float(v) for v in bbox]
        # Clamp to image bounds
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(W), x2), min(float(H), y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return np.zeros((17, 3), dtype=np.float32)

        # Pad bbox by 25% on each side for context (ViTPose works better
        # with some surrounding context, not a tight crop)
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = bw * 0.25, bh * 0.25
        cx1 = max(0.0, x1 - pad_x)
        cy1 = max(0.0, y1 - pad_y)
        cx2 = min(float(W), x2 + pad_x)
        cy2 = min(float(H), y2 + pad_y)

        # Crop to pedestrian region
        crop = image.crop((int(cx1), int(cy1), int(cx2), int(cy2)))
        crop_w, crop_h = crop.size

        if crop_w < 2 or crop_h < 2:
            return np.zeros((17, 3), dtype=np.float32)

        # Run ViTPose on the crop with full-crop bbox
        crop_boxes = [[0, 0, crop_w, crop_h]]
        inputs = processor(images=crop, boxes=[crop_boxes], return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        kpts = processor.post_process_pose_estimation(
            outputs, boxes=[crop_boxes], threshold=0.0
        )
        if kpts and kpts[0]:
            result = kpts[0][0]
            xy     = result['keypoints'].cpu().numpy()  # (17, 2) in crop coords
            scores = result['scores'].cpu().numpy()     # (17,)

            # Remap from crop coordinates to full-frame coordinates
            xy[:, 0] += cx1
            xy[:, 1] += cy1

            return np.concatenate([xy, scores[:, None]], axis=1).astype(np.float32)

    except Exception:
        pass

    return np.zeros((17, 3), dtype=np.float32)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading ViTPose-B from {VITPOSE_MODEL}...")
    processor = VitPoseImageProcessor.from_pretrained(VITPOSE_MODEL)
    model     = VitPoseForPoseEstimation.from_pretrained(VITPOSE_MODEL).to(device)
    model.eval()
    print("Model loaded.")

    dataset = args.dataset.lower()
    print(f"Collecting annotated tracks from {dataset.upper()} API (all splits)...")
    if dataset == 'pie':
        samples = collect_pie_samples(args.data_path)
    elif dataset == 'jaad':
        samples = collect_jaad_samples(args.data_path)
    else:
        print(f"Unknown dataset: {args.dataset}. Use 'pie' or 'jaad'.")
        sys.exit(1)

    print(f"Total unique (pid, frame_id) pairs: {len(samples)}")
    os.makedirs(args.output_dir, exist_ok=True)

    skipped = 0
    for s in tqdm(samples, desc='Extracting keypoints'):
        out_path = os.path.join(args.output_dir, s['pid'], f"{s['frame_id']}.npy")

        if os.path.exists(out_path):
            skipped += 1
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        kpts = extract_keypoints_for_bbox(
            s['img_path'], s['bbox'], model, processor, device
        )
        np.save(out_path, kpts)

    print(f"Done. Extracted {len(samples) - skipped} new files, "
          f"skipped {skipped} existing.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',     choices=['pie', 'jaad'], default='pie')
    parser.add_argument('--data-path',   default='/data/datasets/PIE')
    parser.add_argument('--output-dir',  default='/data/datasets/PIE/keypoints_pid')
    parser.add_argument('--device',      default='cuda:0')
    args = parser.parse_args()
    main(args)
