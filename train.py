"""
Training script for HyperTime v2 / NE-Time

Modes:
  Multi-horizon (default):
    python train.py --dataset ETTm1
    One model trained jointly on all horizons. Our research mode.

  Per-horizon (benchmark comparison):
    python train.py --dataset ETTm1 --per_horizon
    Trains a separate model per horizon — same protocol as PatchTST,
    iTransformer, etc. Use this for fair benchmark comparison.
    Saves individual checkpoints + a combined results JSON.

Changes in this version (overfitting fixes):
  1. Curvature parameters (CurvatureParam) are now routed to the weight-decay
     param group instead of no_decay.  Previously they drifted freely in the
     long tail of training, encouraging overfitting geometry.
  2. HyperbolicEncoder hidden_dim is now controlled by --hyp_hidden_scale
     (default 1.0 = d_model, was 2.0 = d_model*2).  Reduces the encoder from
     ~88K to ~48K params for nano, cutting the biggest overfitting source.
  3. geo_dropout added to encoder/decoder (--geo_dropout, default 0.2).
"""

import os
import time
import math
import argparse
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, List, Optional

from model_no_attn_upd import HyperTimeV2, HorizonWeightedLoss
from dataset import create_dataloaders, download_dataset, unique_window_loader


# ─────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best       = float('inf')
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────────────────
#  Channel-independent (CI) reshape helpers
# ─────────────────────────────────────────────────────────

def to_ci(x: torch.Tensor) -> torch.Tensor:
    """[B, T, C] -> [B*C, T, 1]"""
    B, T, C = x.shape
    return x.permute(0, 2, 1).reshape(B * C, T, 1)


def from_ci(x: torch.Tensor, C: int) -> torch.Tensor:
    """[B*C, H, 1] -> [B, H, C]"""
    BC, H, _ = x.shape
    B = BC // C
    return x.reshape(B, C, H).permute(0, 2, 1)


def cosine_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    p = pred.cpu().float()
    t = target.cpu().float()
    mse  = ((p - t) ** 2).mean().item()
    mae  = (p - t).abs().mean().item()
    rmse = math.sqrt(mse)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse}


class StreamingMetrics:
    """O(1)-memory MSE/MAE/RMSE accumulator.

    compute_metrics() requires holding every batch's predictions/targets
    in memory to torch.cat them at the end, which scales with
    dataset_size * channels * horizon — the source of a host-RAM OOM on
    wide-channel zero-shot eval (Traffic, C=862). This accumulates running
    sums per batch instead, giving identical MSE/MAE (mean over all
    elements) without ever materializing the full prediction tensor.
    """

    def __init__(self):
        self.sum_sq  = 0.0
        self.sum_abs = 0.0
        self.n       = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        diff = (pred.detach().float() - target.detach().float())
        self.sum_sq  += diff.pow(2).sum().item()
        self.sum_abs += diff.abs().sum().item()
        self.n       += diff.numel()

    def compute(self) -> Dict[str, float]:
        mse = self.sum_sq / max(self.n, 1)
        mae = self.sum_abs / max(self.n, 1)
        return {"MSE": mse, "MAE": mae, "RMSE": math.sqrt(mse)}


class CSVLogger:
    def __init__(self, path: str):
        self.path = path

    def log(self, row: Dict):
        first_write = not os.path.exists(self.path)
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if first_write:
                writer.writeheader()
            writer.writerow(row)


# ─────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────

def build_optimizer_and_scheduler(model: HyperTimeV2, args, steps_per_epoch: int):
    """
    Build the AdamW optimizer (3 param groups: decay / no_decay / curvature)
    and cosine-warmup LR scheduler shared by Trainer and any multi-dataset
    training loop (e.g. train_foundation.py).

    Param groups:
      decay       — weights (LinearLayer.weight, conv kernels, …)
      no_decay    — norms, biases, affine params
      curvature   — CurvatureParam.raw_c  (small dedicated weight decay
                    to prevent geometry from drifting in the long tail)
    """
    decay_params     = []
    no_decay_params  = []
    curvature_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Check if this param lives inside a CurvatureParam module
        # by walking the module tree to find its owner
        is_curv = False
        for mod_name, mod in model.named_modules():
            if hasattr(mod, 'is_curvature') and mod.is_curvature:
                # Check if p is raw_c of this module
                for pname, pp in mod.named_parameters(recurse=False):
                    if pp is p:
                        is_curv = True
                        break
            if is_curv:
                break

        if is_curv:
            curvature_params.append(p)
        elif 'norm' in name or 'bias' in name or 'gamma' in name or 'beta' in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    curvature_wd = getattr(args, 'curvature_wd', 1e-3)

    optimizer = optim.AdamW([
        {'params': decay_params,     'weight_decay': args.weight_decay},
        {'params': no_decay_params,  'weight_decay': 0.0},
        {'params': curvature_params, 'weight_decay': curvature_wd,
         'lr': args.lr * 0.1},   # lower LR for curvature too
    ], lr=args.lr)

    total_steps  = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    scheduler    = cosine_warmup_schedule(optimizer, warmup_steps, total_steps)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    n_curv = sum(p.numel() for p in curvature_params)
    print(f"\nParam groups:")
    print(f"  decay        {n_decay:>8,}  (wd={args.weight_decay})")
    print(f"  no_decay     {n_no_decay:>8,}  (wd=0)")
    print(f"  curvature    {n_curv:>8,}  (wd={curvature_wd}, lr×0.1)")

    return optimizer, scheduler


class Trainer:

    def __init__(self, model: HyperTimeV2, args):
        self.model  = model
        self.args   = args
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps'  if torch.backends.mps.is_available() else
            'cpu')
        self.model.to(self.device)

        self.train_loader, self.val_loader, self.test_loader, self.scaler = \
            create_dataloaders(
                dataset_name  = args.dataset,
                root_path     = args.data_path,
                seq_len       = args.seq_len,
                horizons      = args.horizons,
                batch_size    = args.batch_size,
                num_workers   = args.num_workers,
                train_stride  = args.train_stride,
            )

        self.criterion = HorizonWeightedLoss(mse_weight=0.7, mae_weight=0.3)

        steps_per_epoch = len(self.train_loader)
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            model, args, steps_per_epoch)

        self.early_stop = EarlyStopping(patience=args.patience)

        os.makedirs(args.output_dir, exist_ok=True)
        self.ckpt_path = os.path.join(args.output_dir, f"{args.exp_name}_best.pth")
        self.logger    = CSVLogger(os.path.join(args.output_dir, f"{args.exp_name}_log.csv"))

        print(f"\nDevice: {self.device}")
        params = model.param_count()
        print(f"Total parameters: {params['TOTAL']:,}")
        for k, v in params.items():
            if k != 'TOTAL':
                print(f"  {k:<20} {v:>8,}")

    # ── Train one epoch ──────────────────────────────────

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        losses = []
        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            x        = batch['x'].to(self.device)
            y        = batch['y'].to(self.device)
            horizons = batch['horizon']

            unique_horizons = horizons.unique().tolist()
            total_loss = torch.tensor(0.0, device=self.device)
            n_samples  = 0

            for H in unique_horizons:
                H = int(H)
                mask = (horizons == H)
                x_h  = x[mask]
                y_h  = y[mask, :H, :]

                if getattr(self.args, "ci", False):
                    C    = x_h.shape[-1]
                    pred = from_ci(self.model(to_ci(x_h), pred_len=H)[0], C)
                else:
                    pred, _ = self.model(x_h, pred_len=H)
                loss = self.criterion(pred, y_h, horizon=H)

                total_loss = total_loss + loss * mask.sum()
                n_samples  += mask.sum().item()

            avg_loss = total_loss / n_samples
            self.optimizer.zero_grad()
            avg_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            losses.append(avg_loss.item())

            if (step + 1) % 50 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  [{epoch}:{step+1}/{len(self.train_loader)}]  "
                      f"loss={avg_loss.item():.5f}  grad={grad_norm:.3f}  lr={lr:.2e}")

        elapsed = time.time() - t0
        mean_loss = float(np.mean(losses))
        print(f"Epoch {epoch} — train loss: {mean_loss:.5f}  ({elapsed:.1f}s)")
        return mean_loss

    # ── Validate ─────────────────────────────────────────

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        losses = []

        for batch in self.val_loader:
            x        = batch['x'].to(self.device)
            y        = batch['y'].to(self.device)
            horizons = batch['horizon']

            unique_horizons = horizons.unique().tolist()
            batch_loss = 0.0
            n = 0

            for H in unique_horizons:
                H = int(H)
                mask = (horizons == H)
                x_h  = x[mask]
                y_h  = y[mask, :H, :]
                if getattr(self.args, "ci", False):
                    C    = x_h.shape[-1]
                    pred = from_ci(self.model(to_ci(x_h), pred_len=H)[0], C)
                else:
                    pred, _ = self.model(x_h, pred_len=H)
                if y_h.shape[1] < H:
                    continue
                batch_loss += self.criterion(pred, y_h).item() * mask.sum().item()
                n += mask.sum().item()

            if n > 0:
                losses.append(batch_loss / n)

        if not losses:
            return float('inf')

        val_loss = float(np.mean(losses))
        print(f"Epoch {epoch} — val   loss: {val_loss:.5f}")
        return val_loss

    # ── Test ─────────────────────────────────────────────

    @torch.no_grad()
    def test(self, horizons: List[int] = None) -> Dict[int, Dict[str, float]]:
        if horizons is None:
            horizons = self.args.horizons

        self.model.eval()
        results = {}
        eval_loader = unique_window_loader(self.test_loader, self.test_loader.batch_size)

        for H in horizons:
            metrics_acc = StreamingMetrics()
            for batch in eval_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                if getattr(self.args, "ci", False):
                    C    = x.shape[-1]
                    pred = from_ci(self.model(to_ci(x), pred_len=H)[0], C)
                else:
                    pred, _ = self.model(x, pred_len=H)
                metrics_acc.update(pred.cpu(), y[:, :H, :].cpu())

            metrics = metrics_acc.compute()
            results[H] = metrics
            print(f"  H={H:4d}  MSE={metrics['MSE']:.5f}  "
                  f"MAE={metrics['MAE']:.5f}  RMSE={metrics['RMSE']:.5f}")

        return results

    # ── Full train loop ───────────────────────────────────

    def train(self):
        print("\n" + "="*55)
        print(f"Training: {self.args.exp_name}")
        print("="*55)

        best_val = float('inf')

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss   = self.validate(epoch)

            cur_lr = self.optimizer.param_groups[0]['lr']
            log_row = {
                'epoch': epoch, 'train_loss': train_loss,
                'val_loss': val_loss, 'lr': cur_lr,
            }
            log_row['c_global'] = self.model.c_global.c.item()
            log_row['c_meso']   = self.model.c_meso.c.item()
            log_row['c_local']  = self.model.c_local.c.item()
            self.logger.log(log_row)

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'val_loss': val_loss,
                    'args': vars(self.args),
                }, self.ckpt_path)
                print(f"  ✓ Saved checkpoint (val_loss={val_loss:.5f})")

            if self.early_stop.step(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        print("\n" + "="*55)
        print("Loading best checkpoint for testing...")
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        print(f"Best checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f}")

        print("\nTest results:")
        results = self.test()

        results_path = os.path.join(self.args.output_dir, f"{self.args.exp_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"\nResults saved to {results_path}")
        return results


# ─────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'nano':   dict(d_model=64,  hyp_dim=32, cond_dim=32),
    'micro':  dict(d_model=96,  hyp_dim=48, cond_dim=48),
    'small':  dict(d_model=64,  hyp_dim=32, cond_dim=32),
    'medium': dict(d_model=128, hyp_dim=64, cond_dim=64),
    'large':  dict(d_model=256, hyp_dim=128, cond_dim=64),
}


def parse_args():
    p = argparse.ArgumentParser(description='NE-Time Training')

    # Data
    p.add_argument('--dataset',      type=str, default='ETTh1')
    p.add_argument('--data_path',    type=str, default='./data')
    p.add_argument('--seq_len',      type=int, default=336)
    p.add_argument('--horizons',     type=int, nargs='+', default=[96, 192, 336, 720])
    p.add_argument('--train_stride', type=int, default=1)

    # Model
    p.add_argument('--size',            type=str,   default='medium', choices=MODEL_CONFIGS.keys())
    p.add_argument('--patch_size',      type=int,   default=16)
    p.add_argument('--patch_stride',    type=int,   default=8)
    p.add_argument('--dropout',         type=float, default=0.1)
    p.add_argument('--geo_dropout',     type=float, default=0.2,
                   help='Dropout on tangent vectors before expmap0 / after logmap0')
    p.add_argument('--hyp_hidden_scale', type=float, default=1.0,
                   help='Hyperbolic encoder hidden_dim = d_model * this. '
                        'Was 2.0 (d_model*2). 1.0 halves encoder params, '
                        'reducing the main overfitting source.')
    p.add_argument('--proj_hidden', type=int, default=64,
                   help='TemporalProjector bottleneck width (patch_compress/time_expand '
                        'hidden dim). Was hardcoded to 64 regardless of --size. The '
                        'MultiScaleDecomposer was previously found to let the model '
                        'shortcut past the hyperbolic encoders (best val always at epoch 1 '
                        '— see FixedMADecomposer docstring in decomposition_upd.py); '
                        'TemporalProjector sits in the same kind of position (plain '
                        'Euclidean compress-then-expand, right before the output), so this '
                        'flag lets that be tested as a capacity lever too.')

    # Training
    p.add_argument('--epochs',        type=int,   default=50)
    p.add_argument('--warmup_epochs', type=int,   default=3)
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--curvature_wd',  type=float, default=1e-3,
                   help='Weight decay for CurvatureParam (separate from main wd)')
    p.add_argument('--patience',      type=int,   default=10)
    p.add_argument('--num_workers',   type=int,   default=0)

    # Mode
    p.add_argument('--per_horizon', action='store_true')
    p.add_argument('--ci',          action='store_true')

    # Misc
    p.add_argument('--exp_name',   type=str, default='netime')
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--seed',       type=int, default=42)

    return p.parse_args()


def build_model(args, input_dim: int, horizon: int) -> HyperTimeV2:
    model_cfg    = MODEL_CONFIGS[args.size]
    effective_dim = 1 if getattr(args, 'ci', False) else input_dim
    d_model      = model_cfg['d_model']
    hyp_hidden   = max(64, int(d_model * args.hyp_hidden_scale))

    return HyperTimeV2(
        input_dim     = effective_dim,
        seq_len       = args.seq_len,
        max_pred_len  = horizon,
        patch_size    = args.patch_size,
        stride        = args.patch_stride,
        dropout       = args.dropout,
        geo_dropout   = args.geo_dropout,
        hyp_hidden_dim = hyp_hidden,
        proj_hidden   = getattr(args, 'proj_hidden', 64),
        **model_cfg,
    )


def train_single(args, input_dim: int, horizons: List[int]) -> Dict:
    model = build_model(args, input_dim, max(horizons))
    trainer = Trainer(model, args)
    return trainer.train()


def train_per_horizon(args, input_dim: int) -> Dict:
    import copy
    all_results = {}

    for H in args.horizons:
        print("\n" + "="*55)
        print(f"Per-horizon training: H={H}")
        print("="*55)

        h_args          = copy.copy(args)
        h_args.horizons = [H]
        h_args.exp_name = f"{args.exp_name}_H{H}"

        model   = build_model(h_args, input_dim, H)
        trainer = Trainer(model, h_args)
        if len(trainer.val_loader.dataset) == 0:
            print(f"  [SKIP] H={H}: val set has 0 samples, skipping.")
            continue
        results = trainer.train()
        all_results[H] = results[H]

        print(f"  H={H:4d}  MSE={results[H]['MSE']:.5f}  MAE={results[H]['MAE']:.5f}")

    combined_path = os.path.join(args.output_dir, f"{args.exp_name}_per_horizon_results.json")
    with open(combined_path, 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    return all_results


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    download_dataset(args.dataset, args.data_path)

    from dataset import load_csv, DATASET_FILENAMES
    filename  = DATASET_FILENAMES.get(args.dataset, f"{args.dataset}.csv")
    data_path = os.path.join(args.data_path, filename)
    input_dim = load_csv(data_path).shape[1]
    print(f"Input dim: {input_dim}")

    if args.per_horizon and getattr(args, 'ci', False):
        print("\nMode: per-horizon + CI (SOTA benchmark comparison)")
        args.exp_name = args.exp_name + "_CI"
        results = train_per_horizon(args, input_dim)
    elif args.per_horizon:
        print("\nMode: per-horizon (benchmark comparison)")
        results = train_per_horizon(args, input_dim)
    else:
        print("\nMode: multi-horizon (joint training)")
        results = train_single(args, input_dim, args.horizons)

    print("\n" + "="*55)
    print("Final Test Results:")
    print("="*55)
    print(f"  {'H':>6}  {'MSE':>8}  {'MAE':>8}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}")
    for H, m in sorted(results.items()):
        print(f"  {H:>6}  {m['MSE']:>8.5f}  {m['MAE']:>8.5f}")


if __name__ == "__main__":
    main()
