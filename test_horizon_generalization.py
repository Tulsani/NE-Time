"""
Multi-horizon generalization test for NE-Time.

The model was trained and evaluated (everywhere else in this project) only at the four
discrete horizons {96, 192, 336, 720}. This script deliberately tests horizons NOT in
that set, to check whether HorizonEncoder's continuous conditioning on pred_len actually
generalizes smoothly to unseen intermediate horizons, or whether the model only produces
sensible forecasts at the specific values it was trained on.

Architecturally this should work: HorizonEncoder conditions on pred_len as a continuous
value (not a lookup over the training set), and TemporalProjector's fixed 720-wide output
is sliced to any pred_len <= 720 at inference time — nothing in the forward pass restricts
pred_len to the training set. eval_foundation.py adds a conservative check requiring
horizons to be a subset of the checkpoint's trained set; this script deliberately bypasses
that to test what the architecture allows, not just what's been verified before.

Horizons > 720 are NOT true extrapolation — TemporalProjector.MAX_TEMPORAL=720 is a hard
architectural ceiling (time_expand only ever produces 720 values; slicing beyond that just
returns what's available, not a real forecast) — so this script only tests in-range,
untrained intermediate horizons, and warns if you ask for anything above 720.

Usage:
  python test_horizon_generalization.py --ckpt outputs_foundation/<exp_name>_best.pth \
      --datasets ETTh1 ETTh2 --horizons 48 150 250 500 600
"""

import os
import json
import argparse
import torch

from dataset import create_dataloaders
from train_foundation import evaluate_dataset
from train import build_model


DEFAULT_TEST_HORIZONS = [48, 150, 250, 500, 600]


def parse_args():
    p = argparse.ArgumentParser(description='NE-Time multi-horizon generalization test')
    p.add_argument('--ckpt', type=str, required=True,
                    help='Path to a pretrained foundation checkpoint (*_best.pth)')
    p.add_argument('--datasets',     type=str, nargs='+', default=['ETTh1', 'ETTh2'])
    p.add_argument('--horizons',     type=int, nargs='+', default=DEFAULT_TEST_HORIZONS,
                    help='Horizons to test — deliberately NOT restricted to the '
                         "checkpoint's trained horizon set, to test generalization.")
    p.add_argument('--data_path',    type=str, default='./data')
    p.add_argument('--batch_size',   type=int, default=32)
    p.add_argument('--train_stride', type=int, default=1)
    p.add_argument('--num_workers',  type=int, default=0)
    p.add_argument('--output_dir',   type=str, default='./outputs_foundation')
    p.add_argument('--tag',          type=str, default='horizon_generalization')
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
    print(f"Trained horizons: {ckpt_args.horizons}")

    hard_max = max(ckpt_args.horizons)  # TemporalProjector.MAX_TEMPORAL, effectively
    out_of_range = [h for h in args.horizons if h > hard_max]
    horizons = [h for h in args.horizons if h <= hard_max]
    if out_of_range:
        print(f"WARNING: {out_of_range} exceed the architecture's hard ceiling ({hard_max}) "
              f"— TemporalProjector can't truly extrapolate past this, only interpolate "
              f"within it. Dropping these from the test.")
    untested = [h for h in horizons if h not in ckpt_args.horizons]
    print(f"Testing horizons: {horizons}  (never-trained: {untested})")
    if not horizons:
        raise ValueError("No valid horizons left to test after filtering.")

    model = build_model(ckpt_args, input_dim=1, horizon=hard_max)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    print(f"Total parameters: {model.param_count()['TOTAL']:,}")

    results = {}
    for name in args.datasets:
        _, _, test_loader, _ = create_dataloaders(
            dataset_name=name, root_path=args.data_path, seq_len=ckpt_args.seq_len,
            horizons=horizons,  # deliberately the untested set, not ckpt_args.horizons
            batch_size=args.batch_size, num_workers=args.num_workers,
            train_stride=args.train_stride,
        )
        print(f"\n[{name}]")
        results[name] = {str(h): m for h, m in
                          evaluate_dataset(model, test_loader, horizons, device, name).items()}

    out_path = os.path.join(args.output_dir, f"{args.tag}_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
