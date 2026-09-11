"""
Foundation-model training for NE-Time.

Jointly pretrains ONE HyperTimeV2 (CI / univariate mode, so channel-count is
irrelevant across datasets — see train.py's --ci reshape) on a small corpus of
datasets, then evaluates:
  1. In-domain: test-set performance on each pretraining dataset.
  2. Zero-shot: test-set performance on datasets NEVER seen during pretraining
     (no fine-tuning), directly comparable to published Chronos/TimesFM/Moirai
     zero-shot numbers on the same datasets.

Usage:
  python train_foundation.py \
      --pretrain_datasets ETTh1 ETTh2 ETTm1 \
      --zero_shot_datasets ETTm2 Weather Exchange ECL Traffic \
      --size medium --epochs 50

All pretrain + zero-shot datasets share the same --seq_len (default 336),
since that is baked into the model's PatchEmbedding at construction time.
"""

import os
import time
import json
import argparse
import numpy as np
import torch

from model_no_attn_upd import HorizonWeightedLoss
from dataset import create_dataloaders, unique_window_loader
from train import (
    MODEL_CONFIGS, build_model, build_optimizer_and_scheduler,
    to_ci, from_ci, StreamingMetrics, CSVLogger, EarlyStopping,
)


class InfiniteLoader:
    """Wraps a DataLoader to yield batches forever, reshuffling on exhaustion."""

    def __init__(self, loader):
        self.loader = loader
        self._it = iter(loader)

    def next(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)


def batch_loss_and_count(model, batch, args, criterion, device):
    """Per-unique-horizon loss loop shared by train/val steps. Returns
    (avg_loss_tensor, n_samples)."""
    x        = batch['x'].to(device)
    y        = batch['y'].to(device)
    horizons = batch['horizon']

    unique_horizons = horizons.unique().tolist()
    total_loss = torch.tensor(0.0, device=device)
    n_samples  = 0

    for H in unique_horizons:
        H = int(H)
        mask = (horizons == H)
        x_h  = x[mask]
        y_h  = y[mask, :H, :]

        C    = x_h.shape[-1]
        pred = from_ci(model(to_ci(x_h), pred_len=H)[0], C)
        loss = criterion(pred, y_h, horizon=H)

        total_loss = total_loss + loss * mask.sum()
        n_samples  += mask.sum().item()

    return total_loss / max(n_samples, 1), n_samples


@torch.no_grad()
def evaluate_dataset(model, test_loader, horizons, device, dataset_name):
    """Run eval-only forward passes for each horizon on a held-out test_loader.
    Skips a horizon (with a warning) if the test set has zero usable windows."""
    if len(test_loader.dataset) == 0:
        print(f"  [SKIP] {dataset_name}: test set has 0 samples "
              f"(seq_len + max_horizon too long for this split).")
        return {}

    model.eval()
    results = {}
    eval_loader = unique_window_loader(test_loader, test_loader.batch_size)

    for H in horizons:
        metrics_acc = StreamingMetrics()
        n_batches = 0
        for batch in eval_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            C = x.shape[-1]
            pred = from_ci(model(to_ci(x), pred_len=H)[0], C)
            metrics_acc.update(pred.cpu(), y[:, :H, :].cpu())
            n_batches += 1

        if n_batches == 0:
            print(f"  [SKIP] {dataset_name} H={H}: no batches.")
            continue

        metrics = metrics_acc.compute()
        results[H] = metrics
        print(f"  {dataset_name:<10} H={H:4d}  MSE={metrics['MSE']:.5f}  "
              f"MAE={metrics['MAE']:.5f}  RMSE={metrics['RMSE']:.5f}")

    return results


def parse_args():
    p = argparse.ArgumentParser(description='NE-Time Foundation Model Training')

    p.add_argument('--pretrain_datasets',  type=str, nargs='+',
                    default=['ETTh1', 'ETTh2', 'ETTm1'])
    p.add_argument('--zero_shot_datasets', type=str, nargs='+',
                    default=['ETTm2', 'Weather', 'Exchange', 'ECL', 'Traffic'])
    p.add_argument('--data_path', type=str, default='./data')
    p.add_argument('--seq_len',   type=int, default=336)
    p.add_argument('--horizons',  type=int, nargs='+', default=[96, 192, 336, 720])
    p.add_argument('--train_stride', type=int, default=1)

    p.add_argument('--size',            type=str,   default='medium', choices=MODEL_CONFIGS.keys())
    p.add_argument('--patch_size',      type=int,   default=16)
    p.add_argument('--patch_stride',    type=int,   default=8)
    p.add_argument('--dropout',         type=float, default=0.1)
    p.add_argument('--geo_dropout',     type=float, default=0.2)
    p.add_argument('--hyp_hidden_scale', type=float, default=1.0)

    p.add_argument('--epochs',        type=int,   default=50)
    p.add_argument('--warmup_epochs', type=int,   default=3)
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--curvature_wd',  type=float, default=1e-3)
    p.add_argument('--patience',      type=int,   default=15)
    p.add_argument('--num_workers',   type=int,   default=0)

    p.add_argument('--exp_name',   type=str, default='foundation_medium')
    p.add_argument('--output_dir', type=str, default='./outputs_foundation')
    p.add_argument('--seed',       type=int, default=42)

    args = p.parse_args()
    args.ci = True  # CI is mandatory: pretrain datasets have different channel counts
    return args


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu')
    print(f"Device: {device}")

    # ── Build per-pretrain-dataset loaders ──────────────────────
    pretrain = {}
    for name in args.pretrain_datasets:
        train_loader, val_loader, test_loader, scaler = create_dataloaders(
            dataset_name=name, root_path=args.data_path, seq_len=args.seq_len,
            horizons=args.horizons, batch_size=args.batch_size,
            num_workers=args.num_workers, train_stride=args.train_stride,
        )
        if len(train_loader) == 0:
            raise RuntimeError(
                f"Pretrain dataset {name} produced 0 training batches at "
                f"seq_len={args.seq_len}, max_horizon={max(args.horizons)}. "
                f"Split is too short for this config."
            )
        pretrain[name] = dict(
            train=train_loader, val=val_loader, test=test_loader, scaler=scaler,
            train_inf=InfiniteLoader(train_loader),
        )
        print(f"[{name}] train={len(train_loader.dataset):,} "
              f"val={len(val_loader.dataset):,} test={len(test_loader.dataset):,}")

    # ── Build one shared model ───────────────────────────────────
    model = build_model(args, input_dim=1, horizon=max(args.horizons))
    model.to(device)

    params = model.param_count()
    print(f"\nTotal parameters: {params['TOTAL']:,}")

    steps_per_epoch = max(len(pretrain[n]['train']) for n in args.pretrain_datasets)
    optimizer, scheduler = build_optimizer_and_scheduler(model, args, steps_per_epoch)
    criterion = HorizonWeightedLoss(mse_weight=0.7, mae_weight=0.3)
    early_stop = EarlyStopping(patience=args.patience)
    logger = CSVLogger(os.path.join(args.output_dir, f"{args.exp_name}_log.csv"))
    ckpt_path = os.path.join(args.output_dir, f"{args.exp_name}_best.pth")

    names = args.pretrain_datasets
    rng = np.random.default_rng(args.seed)
    best_val = float('inf')

    print("\n" + "=" * 55)
    print(f"Foundation pretraining: {names} -> zero-shot on {args.zero_shot_datasets}")
    print("=" * 55)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        epoch_losses = []

        for step in range(steps_per_epoch):
            name = names[rng.integers(len(names))]  # uniform over datasets, not size-weighted
            batch = pretrain[name]['train_inf'].next()

            avg_loss, _ = batch_loss_and_count(model, batch, args, criterion, device)
            optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(avg_loss.item())

            if (step + 1) % 50 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"  [{epoch}:{step+1}/{steps_per_epoch}] loss={avg_loss.item():.5f} lr={lr:.2e}")

        train_loss = float(np.mean(epoch_losses))
        print(f"Epoch {epoch} — train loss: {train_loss:.5f} ({time.time()-t0:.1f}s)")

        # ── Validation: average per-dataset val loss, equally weighted ──
        model.eval()
        per_ds_val = {}
        with torch.no_grad():
            for name in names:
                losses = []
                for batch in pretrain[name]['val']:
                    loss, n = batch_loss_and_count(model, batch, args, criterion, device)
                    if n > 0:
                        losses.append(loss.item())
                per_ds_val[name] = float(np.mean(losses)) if losses else float('inf')

        val_loss = float(np.mean(list(per_ds_val.values())))
        print(f"Epoch {epoch} — val loss (avg over datasets): {val_loss:.5f}  "
              f"({', '.join(f'{k}={v:.5f}' for k, v in per_ds_val.items())})")

        log_row = {
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'c_global': model.c_global.c.item(),
            'c_meso':   model.c_meso.c.item(),
            'c_local':  model.c_local.c.item(),
        }
        for name in names:
            log_row[f'val_{name}'] = per_ds_val[name]
        logger.log(log_row)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'val_loss': val_loss, 'args': vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved checkpoint (val_loss={val_loss:.5f})")

        if early_stop.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    # ── Load best checkpoint for evaluation ──────────────────────
    print("\n" + "=" * 55)
    print("Loading best checkpoint for evaluation...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Best checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f}")

    # ── In-domain test ────────────────────────────────────────────
    print("\nIn-domain test results:")
    indomain_results = {}
    for name in names:
        indomain_results[name] = evaluate_dataset(
            model, pretrain[name]['test'], args.horizons, device, name)

    with open(os.path.join(args.output_dir, f"{args.exp_name}_indomain_results.json"), 'w') as f:
        json.dump({n: {str(h): m for h, m in r.items()} for n, r in indomain_results.items()},
                   f, indent=2)

    # ── Zero-shot test ────────────────────────────────────────────
    print("\nZero-shot test results:")
    zeroshot_results = {}
    for name in args.zero_shot_datasets:
        _, _, test_loader, _ = create_dataloaders(
            dataset_name=name, root_path=args.data_path, seq_len=args.seq_len,
            horizons=args.horizons, batch_size=args.batch_size,
            num_workers=args.num_workers, train_stride=args.train_stride,
        )
        zeroshot_results[name] = evaluate_dataset(
            model, test_loader, args.horizons, device, name)

    with open(os.path.join(args.output_dir, f"{args.exp_name}_zeroshot_results.json"), 'w') as f:
        json.dump({n: {str(h): m for h, m in r.items()} for n, r in zeroshot_results.items()},
                   f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
