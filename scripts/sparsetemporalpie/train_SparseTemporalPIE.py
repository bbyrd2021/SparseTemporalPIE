"""
train_SparseTemporalPIE.py
==========================
Base training: IL step=0, 50 epochs. (v4 architecture)

Single frame pass — no cross-attention, no context frames.
Temporal context via bbox_traj (12-d) and ctx_feats (5-d).
"""
import sys, os as _os; sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))

import argparse
import os

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.pie_data import PIE
from utils.my_dataset import filter_existing_sequences
from utils.sparse_dataset import SparseDataset
from models.SparseTemporalPIE import SparseTemporalPIE, load_backbone_weights
from utils.train_val import evaluate_sparse


def train_one_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    from utils.train_val import robust_noisy
    model.train()
    loss_fn   = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)

    import sys
    from tqdm import tqdm
    dataloader = tqdm(dataloader, file=sys.stdout)
    for step, (f_current, pose_current, bbox_traj, ctx_feats, labels) in enumerate(dataloader):
        f_current    = f_current.to(device)
        pose_current = pose_current.to(device)
        bbox_traj    = bbox_traj.to(device)
        ctx_feats    = ctx_feats.to(device)
        labels       = labels.to(device)

        optimizer.zero_grad()
        logits   = model(f_current, pose_current, bbox_traj, ctx_feats)
        logits_p = robust_noisy(logits, epoch, total_epochs)
        loss     = loss_fn(logits_p, labels)
        loss.backward()
        optimizer.step()
        accu_loss += loss.detach()
        dataloader.desc = f"[train epoch {epoch}] loss: {accu_loss.item()/(step+1):.3f}"

    return accu_loss.item() / (step + 1)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(args)

    tb_writer = SummaryWriter()
    os.makedirs(args.output_dir, exist_ok=True)

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
    train_seq = pie.generate_data_trajectory_sequence('train', **data_opts)
    val_seq   = pie.generate_data_trajectory_sequence('val',   **data_opts)
    seq_len   = data_opts['max_size_observe']

    train_seq_data = pie.get_train_val_data(train_seq, data_type, seq_len, data_opts['seq_overlap_rate'])
    val_seq_data   = pie.get_train_val_data(val_seq,   data_type, seq_len, data_opts['seq_overlap_rate'])
    train_seq_data = filter_existing_sequences(train_seq_data, args.step, seq_len)
    val_seq_data   = filter_existing_sequences(val_seq_data,   args.step, seq_len)

    model = SparseTemporalPIE(dropout=0.1).to(device)
    model = load_backbone_weights(model, args.weights, device=args.device)

    train_transform = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dataset = SparseDataset(train_seq_data, data_opts, step=args.step,
                                  transform=train_transform,
                                  keypoints_dir=args.keypoints_dir,
                                  flip_p=0.5,
                                  pose_dropout_p=0.1)
    val_dataset   = SparseDataset(val_seq_data,   data_opts, step=args.step,
                                  transform=val_transform,
                                  keypoints_dir=args.keypoints_dir)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              collate_fn=train_dataset.collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              collate_fn=val_dataset.collate_fn)

    # Partial unfreeze backbone at 10x lower LR
    for p in model.backbone.parameters():
        p.requires_grad = True

    backbone_ids    = set(id(p) for p in model.backbone.parameters())
    other_params    = [p for p in model.parameters() if id(p) not in backbone_ids]
    backbone_params = list(model.backbone.parameters())
    print(f"Backbone params:  {sum(p.numel() for p in backbone_params):,} @ lr={args.lr*0.1:.2e}")
    print(f"Other params:     {sum(p.numel() for p in other_params):,} @ lr={args.lr:.2e}")

    optimizer = optim.RMSprop([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': other_params,    'lr': args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.restart_period, T_mult=1, eta_min=1e-7)

    best_val_acc  = 0.0
    best_path     = os.path.join(args.output_dir, f'best_sparse_step{args.step}.pth')
    min_loss      = 100.0
    min_loss_path = os.path.join(args.output_dir, f'min_loss_sparse_step{args.step}.pth')

    print("Start Training!")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        scheduler.step()
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_sparse(
            model, val_loader, device, epoch)

        tb_writer.add_scalar('train_loss', train_loss, epoch)
        tb_writer.add_scalar('val_loss',   val_loss,   epoch)
        tb_writer.add_scalar('val_acc',    val_acc,    epoch)
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f'Saved best model epoch {epoch}: acc={val_acc:.4f}')
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), min_loss_path)

    print("Finished Training!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',        default='/data/datasets/PIE')
    parser.add_argument('--weights',          default='weights_v8/model_8_PIE_IL_step14_new.pth')
    parser.add_argument('--keypoints-dir',    default='/data/datasets/PIE/keypoints_pid')
    parser.add_argument('--output-dir',       default='weights_sparse_v4')
    parser.add_argument('--step',             type=int,   default=0)
    parser.add_argument('--epochs',           type=int,   default=50)
    parser.add_argument('--batch-size',       type=int,   default=32)
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--weight-decay',     type=float, default=1e-4)
    parser.add_argument('--restart-period',   type=int,   default=10)
    parser.add_argument('--device',           default='cuda:0')
    args = parser.parse_args()
    main(args)
