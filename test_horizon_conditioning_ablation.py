"""
Horizon-conditioning ablation — addresses reviewer e3ix's sharpest technical concern on
the untrained-intermediate-horizon result (paper-draft.md Section 3.2.1 / test_horizon_
generalization.py):

  "The temporal projector generates outputs up to H_max=720, after which the first H
   positions are retained. Consequently, positions used by an unseen horizon such as 250
   have already received training signal through the objectives for horizons 336 and 720.
   [...] average forecasting error may vary smoothly with horizon length even when the
   conditioning representation contributes little. [...] controls using no horizon encoder,
   fixed or shuffled horizon inputs, discrete horizon embeddings, interpolated discrete
   embeddings, and deliberately incorrect conditioning values while keeping the requested
   truncation length unchanged[] would separate the effect of q_H from the nested structure
   of the output head."

This runs an ALREADY-TRAINED checkpoint (no retraining needed) at the same untrained
intermediate horizons as test_horizon_generalization.py, under four horizon-conditioning
modes, while the actual requested/truncated length (and therefore what `y` it's scored
against) is held fixed per horizon in every mode:

  - real:     the model is told the true horizon it's being asked to produce (original,
              unablated behaviour -- cond_override=None)
  - fixed:    the model is always told 336 (one arbitrary trained horizon), regardless of
              what's actually requested
  - shuffled: the model is told a horizon resampled uniformly from the trained set
              {96,192,336,720} independently each batch, decoupled from what's requested
  - nearest_wrong: the model is told the trained horizon FARTHEST from what's actually
              requested (maximally wrong conditioning, same trained-horizon vocabulary)

If "real" clearly beats fixed/shuffled/nearest_wrong at horizons far from the trained set,
that's evidence the horizon encoder's conditioning is doing real work, not just truncation
of a shared output curve. If all four modes produce near-identical error-vs-horizon curves,
that supports the reviewer's concern that Section 3.2.1's result is largely an artifact of
the nested output-head structure.

Usage:
  python test_horizon_conditioning_ablation.py --ckpt outputs_foundation/<exp>_best.pth \
      --datasets ETTh1 ETTh2 --horizons 48 150 250 500 600
"""

import os
import json
import argparse
import random
import torch

from dataset import create_dataloaders, unique_window_loader
from train import build_model, to_ci, from_ci, StreamingMetrics


TRAINED_HORIZONS = [96, 192, 336, 720]
DEFAULT_TEST_HORIZONS = [48, 150, 250, 500, 600]
MODES = ['real', 'fixed', 'shuffled', 'nearest_wrong']


def parse_args():
    p = argparse.ArgumentParser(description='NE-Time horizon-conditioning ablation')
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--datasets', type=str, nargs='+', default=['ETTh1', 'ETTh2'])
    p.add_argument('--horizons', type=int, nargs='+', default=DEFAULT_TEST_HORIZONS)
    p.add_argument('--fixed_cond', type=int, default=336,
                   help="Conditioning value used by the 'fixed' mode.")
    p.add_argument('--data_path', type=str, default='./data')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--train_stride', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output_dir', type=str, default='./outputs_foundation')
    p.add_argument('--tag', type=str, default='horizon_conditioning_ablation')
    return p.parse_args()


def nearest_wrong_horizon(h: int) -> int:
    """The trained horizon FARTHEST from h — maximally wrong conditioning."""
    return max(TRAINED_HORIZONS, key=lambda th: abs(th - h))


@torch.no_grad()
def evaluate_mode(model, test_loader, horizons, device, dataset_name, mode, rng):
    """Mirrors train_foundation.evaluate_dataset, but with per-mode cond_override."""
    if len(test_loader.dataset) == 0:
        print(f"  [SKIP] {dataset_name}: test set has 0 samples.")
        return {}

    model.eval()
    results = {}
    eval_loader = unique_window_loader(test_loader, test_loader.batch_size)

    for H in horizons:
        metrics_acc = StreamingMetrics()
        for batch in eval_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            C = x.shape[-1]

            if mode == 'real':
                cond_override = None
            elif mode == 'fixed':
                cond_override = 336
            elif mode == 'shuffled':
                cond_override = int(rng.choice(TRAINED_HORIZONS))
            elif mode == 'nearest_wrong':
                cond_override = nearest_wrong_horizon(H)
            else:
                raise ValueError(mode)

            pred, _ = model(to_ci(x), pred_len=H, cond_override=cond_override)
            pred = from_ci(pred, C)
            metrics_acc.update(pred.cpu(), y[:, :H, :].cpu())

        metrics = metrics_acc.compute()
        results[H] = metrics
        print(f"  {dataset_name:<10} mode={mode:<14} H={H:4d}  MSE={metrics['MSE']:.5f}  "
              f"MAE={metrics['MAE']:.5f}")

    return results


def main():
    args = parse_args()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu')
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt['args'])
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f}, "
          f"geometry={getattr(ckpt_args, 'geometry', 'hyperbolic')}")

    hard_max = max(ckpt_args.horizons)
    horizons = [h for h in args.horizons if h <= hard_max]
    if len(horizons) != len(args.horizons):
        print(f"WARNING: dropping horizons above the architecture's hard ceiling ({hard_max}).")

    model = build_model(ckpt_args, input_dim=1, horizon=hard_max)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)

    all_results = {}
    for name in args.datasets:
        _, _, test_loader, _ = create_dataloaders(
            dataset_name=name, root_path=args.data_path, seq_len=ckpt_args.seq_len,
            horizons=horizons, batch_size=args.batch_size, num_workers=args.num_workers,
            train_stride=args.train_stride,
        )
        all_results[name] = {}
        for mode in MODES:
            print(f"\n[{name} | mode={mode}]")
            rng = random.Random(args.seed)  # same shuffled sequence across datasets/modes
            all_results[name][mode] = {
                str(h): m for h, m in
                evaluate_mode(model, test_loader, horizons, device, name, mode, rng).items()
            }

    out_path = os.path.join(args.output_dir, f"{args.tag}_results.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("Summary (MSE): if 'real' isn't clearly best at horizons far from the trained")
    print("set {96,192,336,720}, the untrained-horizon result is likely an artifact of the")
    print("output head's nested structure rather than genuine horizon conditioning.")
    print("=" * 70)
    for name, per_mode in all_results.items():
        print(f"\n{name}:")
        header = f"  {'H':>6}" + "".join(f"{m:>16}" for m in MODES)
        print(header)
        for h in horizons:
            row = f"  {h:>6}"
            for m in MODES:
                mse = per_mode[m].get(str(h), {}).get('MSE')
                row += f"{mse:>16.5f}" if mse is not None else f"{'--':>16}"
            print(row)


if __name__ == "__main__":
    main()
