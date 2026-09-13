"""
Imputation fine-tuning for NE-Time.

Loads a pretrained HyperTimeV2 foundation checkpoint (from train_foundation.py), freezes
it, and trains a new PatchUnfold reconstruction head via linear probing — following
MOMENT's imputation protocol: contiguous length-8 masked chunks at ratios
{12.5%, 25%, 37.5%, 50%}, MSE/MAE reported on masked positions only, on ETTh1/h2/m1/m2,
Weather, and ECL.

A separate, freshly-initialized head is trained per (dataset, mask_ratio) pair — simplest,
most defensible choice given time constraints, and consistent with how forecasting is
evaluated per-dataset in this codebase already. Note: unlike MOMENT (pretrained via masked
reconstruction, so it has a *native* pretrained head and can report a true zero-shot
imputation number), this backbone was pretrained via forecasting — there is no pretrained
reconstruction head to evaluate zero-shot, so only the linear-probing setting is reported
here. Worth stating explicitly wherever these numbers are cited.

Usage:
  python finetune_imputation.py --ckpt outputs_foundation/<exp_name>_best.pth \
      --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather ECL
"""

import os
import json
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

from dataset import (
    download_dataset, load_csv, get_split,
    ImputationDataset, collate_fn_imputation,
)
from model_no_attn_upd import HyperTimeV2, PatchUnfold
from train import MODEL_CONFIGS


MASK_RATIOS = [0.125, 0.25, 0.375, 0.5]
DEFAULT_DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'ECL']


def parse_args():
    p = argparse.ArgumentParser(description='NE-Time Imputation Fine-tuning')
    p.add_argument('--ckpt', type=str, required=True,
                    help='Path to a pretrained foundation checkpoint (*_best.pth)')
    p.add_argument('--datasets',       type=str,  nargs='+', default=DEFAULT_DATASETS)
    p.add_argument('--data_path',      type=str,  default='./data')
    p.add_argument('--mask_ratios',    type=float, nargs='+', default=MASK_RATIOS)
    p.add_argument('--mask_chunk_len', type=int,  default=8)
    p.add_argument('--batch_size',     type=int,  default=32)
    p.add_argument('--epochs',         type=int,  default=10)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--backbone_lr',    type=float, default=1e-4,
                    help='LR for backbone params when --finetune_backbone is set (typically '
                         'lower than the head LR, since the backbone starts pretrained).')
    p.add_argument('--finetune_backbone', action='store_true',
                    help='Also fine-tune the backbone (not just the new head). Default is a '
                         'strict linear probe (backbone frozen). A backbone pretrained purely '
                         'for forecasting has no learned notion of "this position is masked, '
                         'infer it from context" — masked positions reach the frozen patch_embed '
                         'as plain zeros with no mask signal at all, since patch_embed\'s input '
                         'width was fixed at pretrain time with no room for a mask channel. '
                         'Linear-probing alone leaves that gap unaddressed; unfreezing gives the '
                         'encoder a chance to adapt.')
    p.add_argument('--num_workers',    type=int,  default=0)
    p.add_argument('--exp_name',       type=str,  default='imputation')
    p.add_argument('--output_dir',     type=str,  default='./outputs_imputation')
    p.add_argument('--seed',           type=int,  default=42)
    return p.parse_args()


def load_backbone(ckpt_path: str, device: torch.device):
    """Load a pretrained HyperTimeV2 from a train_foundation.py checkpoint.

    Returns the model plus its original state_dict (for resetting between independent
    (dataset, ratio) fine-tuning runs when --finetune_backbone is set — without a reset,
    the backbone would carry over adaptation from one pair into the next, contaminating
    what's supposed to be an independent experiment per cell).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt['args'])
    model_cfg  = MODEL_CONFIGS[ckpt_args.size]
    d_model    = model_cfg['d_model']
    hyp_hidden = max(8, int(d_model * getattr(ckpt_args, 'hyp_hidden_scale', 1.0)))
    proj_hidden = getattr(ckpt_args, 'proj_hidden', 64)

    backbone = HyperTimeV2(
        input_dim=1, seq_len=ckpt_args.seq_len, max_pred_len=max(ckpt_args.horizons),
        patch_size=ckpt_args.patch_size, stride=ckpt_args.patch_stride,
        dropout=ckpt_args.dropout, geo_dropout=ckpt_args.geo_dropout,
        hyp_hidden_dim=hyp_hidden, proj_hidden=proj_hidden, **model_cfg,
    )
    backbone.load_state_dict(ckpt['model_state'])
    backbone.to(device)

    print(f"Loaded backbone from {ckpt_path}")
    print(f"  size={ckpt_args.size}  seq_len={ckpt_args.seq_len}  "
          f"patch_size={ckpt_args.patch_size}  stride={ckpt_args.patch_stride}")
    print(f"  backbone params: {sum(p.numel() for p in backbone.parameters()):,}")
    return backbone, ckpt['model_state'], ckpt_args, d_model


def reset_backbone(backbone: HyperTimeV2, original_state: dict, freeze: bool):
    """Reset to the original pretrained weights, then freeze or unfreeze in place."""
    backbone.load_state_dict(original_state)
    for param in backbone.parameters():
        param.requires_grad = not freeze
    backbone.eval() if freeze else backbone.train()


def build_head(backbone: HyperTimeV2, seq_len: int, patch_size: int, stride: int,
               d_model: int, device: torch.device) -> PatchUnfold:
    return PatchUnfold(
        seq_len=seq_len, patch_size=patch_size, stride=stride,
        num_patches=backbone.patch_embed.num_patches, d_model=d_model, out_dim=1,
    ).to(device)


def masked_mse_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """mask: 1=observed, 0=masked. Metrics computed on masked positions only."""
    inv = (1 - mask)
    denom = inv.sum().clamp(min=1)
    mse = ((pred - target) ** 2 * inv).sum() / denom
    mae = ((pred - target).abs() * inv).sum() / denom
    return mse, mae


def run_epoch(loader, backbone, head, seq_len, device, optimizer=None):
    """One pass over `loader`. Trains `head` (and `backbone`, if it has any trainable
    params) when `optimizer` is given, else eval-only."""
    train = optimizer is not None
    head.train(train)
    # Only flip the backbone's mode if it's actually being fine-tuned — a frozen backbone
    # stays in eval() throughout (set by reset_backbone), so dropout/geo_dropout stay off
    # for it regardless of the head's mode.
    if any(p.requires_grad for p in backbone.parameters()):
        backbone.train(train)
    total_mse, total_mae, n = 0.0, 0.0, 0

    for batch in loader:
        x_masked = batch['x_masked'].to(device)
        x_orig   = batch['x_orig'].to(device)
        mask     = batch['mask'].to(device)
        C = x_masked.shape[-1]

        # CI reshape: [B,T,C] -> [B*C,T,1] (per-channel univariate, matching the rest of
        # this codebase's channel-independent convention)
        x_masked_ci = x_masked.permute(0, 2, 1).reshape(-1, seq_len, 1)
        x_orig_ci   = x_orig.permute(0, 2, 1).reshape(-1, seq_len, 1)
        mask_ci     = mask.permute(0, 2, 1).reshape(-1, seq_len, 1)

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            euclidean, _, _, _ = backbone.encode_patches(x_masked_ci, mask=mask_ci)
            recon = head(euclidean)
            mse, mae = masked_mse_mae(recon, x_orig_ci, mask_ci)

        if train:
            optimizer.zero_grad()
            mse.backward()
            optimizer.step()

        total_mse += mse.item()
        total_mae += mae.item()
        n += 1

    if n == 0:
        return None
    return {'MSE': total_mse / n, 'MAE': total_mae / n}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu')
    print(f"Device: {device}")

    backbone, original_state, ckpt_args, d_model = load_backbone(args.ckpt, device)
    seq_len = ckpt_args.seq_len
    freeze_backbone = not args.finetune_backbone
    print(f"Backbone mode: {'frozen (linear probe)' if freeze_backbone else 'fine-tuned jointly with the head'}")

    results = {}
    for name in args.datasets:
        path = download_dataset(name, args.data_path)
        data = load_csv(path)
        train_ratio, val_ratio, test_ratio = get_split(name)
        print(f"\nLoaded {name}: shape={data.shape}  split={train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}")

        results[name] = {}
        for ratio in args.mask_ratios:
            train_ds = ImputationDataset(
                data, seq_len=seq_len, mask_ratio=ratio, split='train',
                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                mask_chunk_len=args.mask_chunk_len, seed=args.seed)
            test_ds = ImputationDataset(
                data, seq_len=seq_len, mask_ratio=ratio, split='test',
                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                mask_chunk_len=args.mask_chunk_len, seed=args.seed)

            if len(train_ds) == 0 or len(test_ds) == 0:
                print(f"  [SKIP] {name} ratio={ratio}: train={len(train_ds)} test={len(test_ds)} windows")
                continue

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                       collate_fn=collate_fn_imputation, num_workers=args.num_workers)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                      collate_fn=collate_fn_imputation, num_workers=args.num_workers)

            # Fresh head per (dataset, ratio) — see module docstring for why. Backbone is
            # reset to its original pretrained weights too, so fine-tuning (when enabled)
            # for one (dataset, ratio) pair can't contaminate the next.
            reset_backbone(backbone, original_state, freeze=freeze_backbone)
            head = build_head(backbone, seq_len, ckpt_args.patch_size, ckpt_args.patch_stride,
                               d_model, device)
            if freeze_backbone:
                optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
            else:
                optimizer = torch.optim.AdamW([
                    {'params': head.parameters(), 'lr': args.lr},
                    {'params': backbone.parameters(), 'lr': args.backbone_lr},
                ])

            print(f"  [{name}] mask_ratio={ratio}  train_windows={len(train_ds)}  test_windows={len(test_ds)}")
            for epoch in range(1, args.epochs + 1):
                train_metrics = run_epoch(train_loader, backbone, head, seq_len, device, optimizer)
                if epoch == args.epochs or epoch == 1:
                    print(f"    epoch {epoch}/{args.epochs}: "
                          f"train MSE={train_metrics['MSE']:.5f} MAE={train_metrics['MAE']:.5f}")

            test_metrics = run_epoch(test_loader, backbone, head, seq_len, device, optimizer=None)
            results[name][str(ratio)] = test_metrics
            print(f"    TEST  MSE={test_metrics['MSE']:.5f}  MAE={test_metrics['MAE']:.5f}")

    out_path = os.path.join(args.output_dir, f"{args.exp_name}_imputation_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
