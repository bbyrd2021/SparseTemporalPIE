"""
benchmark_inference.py
======================
Measures inference latency for SparseTemporalPIE v3 and v4 following the
EfficientPIE benchmark protocol: warm GPU (50 runs), then 100 timed runs,
batch=128, synthetic inputs, CUDA events for timing.

Reports: mean ms/sample ± std, total ms/batch
"""
import sys, os as _os; sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..')))

import argparse
import torch
import numpy as np

from models.SparseTemporalPIE_v3 import SparseTemporalPIE_v3
from models.SparseTemporalPIE import SparseTemporalPIE


def make_v3_batch(batch_size, device, K=4):
    return (
        torch.randn(batch_size, 3, 300, 300,   device=device),          # f_current
        torch.randn(batch_size, K, 3, 300, 300, device=device),          # f_context
        torch.ones(batch_size, K,               device=device),          # context_mask (all real)
        torch.randn(batch_size, 68,             device=device),          # pose_current
        torch.randn(batch_size, K, 68,          device=device),          # pose_context
        torch.randn(batch_size, 12,             device=device),          # bbox_traj
        torch.randn(batch_size, 5,              device=device),          # ctx_feats
    )


def make_v4_batch(batch_size, device):
    return (
        torch.randn(batch_size, 3, 300, 300, device=device),             # f_current
        torch.randn(batch_size, 34,          device=device),             # pose_current
        torch.randn(batch_size, 12,          device=device),             # bbox_traj
        torch.randn(batch_size, 5,           device=device),             # ctx_feats
    )


@torch.no_grad()
def benchmark(model, make_batch_fn, batch_size, device, warmup=50, runs=100):
    model.eval()

    # Warm up
    for _ in range(warmup):
        inputs = make_batch_fn(batch_size, device)
        _ = model(*inputs)
    torch.cuda.synchronize(device)

    # Timed runs using CUDA events
    times = []
    for _ in range(runs):
        inputs = make_batch_fn(batch_size, device)
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = model(*inputs)
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end))  # ms for full batch

    times = np.array(times)
    per_sample = times / batch_size
    return per_sample.mean(), per_sample.std(), times.mean()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}  |  Warmup: {args.warmup}  |  Runs: {args.runs}\n")

    results = {}

    # --- v3 ---
    if args.weights_v3:
        print(f"Loading v3: {args.weights_v3}")
        model_v3 = SparseTemporalPIE_v3(dropout=0.0).to(device)
        model_v3.load_state_dict(torch.load(args.weights_v3, map_location=device))
        mean_ms, std_ms, batch_ms = benchmark(
            model_v3, make_v3_batch, args.batch_size, device, args.warmup, args.runs)
        results['v3'] = (mean_ms, std_ms, batch_ms)
        print(f"  v3  per-sample: {mean_ms:.4f} ± {std_ms:.4f} ms")
        print(f"  v3  per-batch:  {batch_ms:.4f} ms  (batch={args.batch_size})")
        del model_v3
        torch.cuda.empty_cache()
    else:
        print("Skipping v3 (no --weights-v3 provided)")

    print()

    # --- v4 ---
    if args.weights_v4:
        print(f"Loading v4: {args.weights_v4}")
        model_v4 = SparseTemporalPIE(dropout=0.0).to(device)
        model_v4.load_state_dict(torch.load(args.weights_v4, map_location=device))
        mean_ms, std_ms, batch_ms = benchmark(
            model_v4, make_v4_batch, args.batch_size, device, args.warmup, args.runs)
        results['v4'] = (mean_ms, std_ms, batch_ms)
        print(f"  v4  per-sample: {mean_ms:.4f} ± {std_ms:.4f} ms")
        print(f"  v4  per-batch:  {batch_ms:.4f} ms  (batch={args.batch_size})")
        del model_v4
        torch.cuda.empty_cache()
    else:
        print("Skipping v4 (no --weights-v4 provided)")

    print()
    print("=" * 50)
    print("  Summary (mean ms/sample, batch=128)")
    print("=" * 50)
    for name, (mean_ms, std_ms, _) in results.items():
        print(f"  {name:<6}  {mean_ms:.4f} ± {std_ms:.4f} ms")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-v3',  default='weights_sparse_v3/best_sparse_step14.pth')
    parser.add_argument('--weights-v4',  default='weights_sparse_v4/best_sparse_step2.pth')
    parser.add_argument('--batch-size',  type=int, default=128)
    parser.add_argument('--warmup',      type=int, default=50)
    parser.add_argument('--runs',        type=int, default=100)
    parser.add_argument('--device',      default='cuda:0')
    args = parser.parse_args()
    main(args)
