"""
Standalone post-training evaluation for a saved NE-Time foundation checkpoint.

Unlike train_foundation.py's built-in in-domain/zero-shot eval (which only
runs right after training finishes), this script loads a saved
`*_best.pth` checkpoint on its own — useful for re-running benchmarks after
pulling a checkpoint back from an HPC job, evaluating on additional datasets
you didn't originally list, or comparing several checkpoints.

Usage:
  python eval_foundation.py --ckpt outputs_foundation/foundation_medium_best.pth \
      --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange ECL Traffic

By default, evaluates on the union of the checkpoint's own pretrain +
zero-shot dataset lists (stored in the checkpoint's args), and reuses the
exact horizon list the model was trained with (required — the model's
HorizonEncoder normalises by the training-time max horizon).
"""

import os
import json
import argparse
import torch

from train import build_model, to_ci, from_ci, compute_metrics
from train_foundation import evaluate_dataset
from dataset import create_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description='NE-Time Foundation Model — standalone eval')
    p.add_argument('--ckpt', type=str, required=True,
                    help='Path to a foundation checkpoint (*_best.pth)')
    p.add_argument('--datasets', type=str, nargs='+', default=None,
                    help='Datasets to evaluate on. Default: union of the '
                         "checkpoint's pretrain_datasets + zero_shot_datasets.")
    p.add_argument('--horizons', type=int, nargs='+', default=None,
                    help='Subset of the checkpoint horizons to evaluate. '
                         'Default: all horizons the model was trained with.')
    p.add_argument('--data_path',   type=str, default='./data')
    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--train_stride', type=int, default=1)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--output_dir',  type=str, default=None,
                    help='Default: same directory as --ckpt')
    p.add_argument('--tag', type=str, default='eval',
                    help='Suffix for the output results filename')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu')
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt['args'])
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f}")
    print(f"Trained on pretrain_datasets={ckpt_args.pretrain_datasets}, "
          f"zero_shot_datasets={ckpt_args.zero_shot_datasets}")

    horizons = args.horizons or ckpt_args.horizons
    if not set(horizons).issubset(set(ckpt_args.horizons)):
        raise ValueError(
            f"--horizons {horizons} must be a subset of the checkpoint's "
            f"training horizons {ckpt_args.horizons} (HorizonEncoder "
            f"normalises by the training-time max horizon).")

    datasets = args.datasets or sorted(set(
        ckpt_args.pretrain_datasets + ckpt_args.zero_shot_datasets))

    model = build_model(ckpt_args, input_dim=1, horizon=max(ckpt_args.horizons))
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    print(f"Total parameters: {model.param_count()['TOTAL']:,}")

    results = {}
    for name in datasets:
        _, _, test_loader, _ = create_dataloaders(
            dataset_name=name, root_path=args.data_path, seq_len=ckpt_args.seq_len,
            horizons=ckpt_args.horizons,  # must match training windowing exactly
            batch_size=args.batch_size, num_workers=args.num_workers,
            train_stride=args.train_stride,
        )
        tag = 'in-domain' if name in ckpt_args.pretrain_datasets else 'zero-shot'
        print(f"\n[{tag}] {name}")
        results[name] = {
            'protocol': tag,
            'metrics': {str(h): m for h, m in
                        evaluate_dataset(model, test_loader, horizons, device, name).items()},
        }

    out_dir = args.output_dir or os.path.dirname(args.ckpt) or '.'
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_path = os.path.join(out_dir, f"{ckpt_name}_{args.tag}_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
