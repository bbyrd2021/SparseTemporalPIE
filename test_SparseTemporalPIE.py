"""
test_SparseTemporalPIE.py
=========================
Reports two sets of metrics:
  1. Overall     -- full test set
  2. v=0 subset  -- stationary pedestrians (bbox center displacement < epsilon)
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score

from utils.pie_data import PIE
from utils.my_dataset import filter_existing_sequences
from utils.sparse_dataset import SparseDataset
from models.SparseTemporalPIE import SparseTemporalPIE


def compute_metrics(y_true, y_pred, y_prob):
    return (accuracy_score(y_true, y_pred),
            roc_auc_score(y_true, y_prob),
            f1_score(y_true, y_pred, zero_division=0),
            precision_score(y_true, y_pred, zero_division=0))


def is_stationary(bboxes, epsilon=5.0):
    if len(bboxes) < 2:
        return False
    def center(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)
    cx0, cy0 = center(bboxes[0])
    cxN, cyN = center(bboxes[-1])
    width = bboxes[0][2] - bboxes[0][0]
    if width < 1e-6: return False
    return np.sqrt((cxN-cx0)**2 + (cyN-cy0)**2) / width < epsilon


@torch.no_grad()
def evaluate(model, dataset, device, epsilon=5.0):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4,
                        collate_fn=dataset.collate_fn)
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    for f_current, pose_current, bbox_traj, ctx_feats, labels in loader:
        logits = model(f_current.to(device), pose_current.to(device),
                       bbox_traj.to(device), ctx_feats.to(device))
        all_prob.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        all_pred.extend(logits.argmax(dim=1).cpu().numpy())
        all_true.extend(labels.numpy())

    stationary_mask = np.array([is_stationary(dataset.images_seq['bboxes'][i], epsilon)
                                 for i in range(len(dataset))], dtype=bool)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)

    overall = compute_metrics(all_true, all_pred, all_prob)
    if stationary_mask.sum() > 0:
        vzero = compute_metrics(all_true[stationary_mask], all_pred[stationary_mask], all_prob[stationary_mask])
    else:
        vzero = (None, None, None, None)
        print("  Warning: no stationary samples found")

    return overall, vzero, int(stationary_mask.sum())


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    data_opts = {
        'fstride': 1, 'sample_type': 'all', 'height_rng': [0, float('inf')],
        'squarify_ratio': 0, 'data_split_type': 'random', 'seq_type': 'intention',
        'min_track_size': 0, 'max_size_observe': 15, 'seq_overlap_rate': 0.5,
        'balance': True, 'crop_type': 'context', 'crop_mode': 'pad_resize',
        'encoder_input_type': [], 'decoder_input_type': ['bbox'],
        'output_type': ['intention_binary'],
    }
    data_type = {
        'encoder_input_type': [], 'decoder_input_type': ['bbox'],
        'output_type': ['intention_binary'],
    }

    pie = PIE(data_path=args.data_path)
    test_seq      = pie.generate_data_trajectory_sequence('test', **data_opts)
    seq_len       = data_opts['max_size_observe']
    test_seq_data = pie.get_train_val_data(test_seq, data_type, seq_len, data_opts['seq_overlap_rate'])
    test_seq_data = filter_existing_sequences(test_seq_data, args.step, seq_len)

    model = SparseTemporalPIE().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    test_dataset = SparseDataset(test_seq_data, data_opts, step=args.step,
                                 transform=transforms.Compose([
                                     transforms.Resize([300, 300]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
                                 ]),
                                 keypoints_dir=args.keypoints_dir)

    overall, vzero, n_stationary = evaluate(model, test_dataset, device, args.epsilon)

    print("\n" + "="*55)
    print("  SparseTemporalPIE v4 -- Evaluation Results")
    print("="*55)
    print(f"  {'Metric':<12}  {'Overall':>10}  {'v=0 Subset':>12}")
    print(f"  {'-'*38}")
    for lbl, ov, vz in zip(['Accuracy','AUC','F1','Precision'], overall, vzero):
        print(f"  {lbl:<12}  {ov:.4f if ov else 'N/A':>10}  {vz:.4f if vz else 'N/A':>12}")
    print(f"\n  v=0 subset: {n_stationary} / {len(test_dataset)} samples")
    print("="*55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',     default='/data/datasets/PIE')
    parser.add_argument('--weights',       default='weights_sparse_v4/best_sparse_step14.pth')
    parser.add_argument('--keypoints-dir', default='/data/datasets/PIE/keypoints_pid')
    parser.add_argument('--step',          type=int,   default=14)
    parser.add_argument('--epsilon',       type=float, default=5.0)
    parser.add_argument('--device',        default='cuda:0')
    args = parser.parse_args()
    main(args)
