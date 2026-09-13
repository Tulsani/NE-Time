"""
Classification for NE-Time, via MOMENT's own protocol: extract frozen embeddings from
the pretrained backbone, fit an off-the-shelf SVM on them, report accuracy. No new
training loop — the backbone is used purely as a frozen feature extractor.

Datasets: a handful of well-known univariate UCR datasets (via `aeon`, not all 91 MOMENT
uses, given time constraints) — small, fast, commonly cited in TSC benchmark papers.

UCR series are usually much shorter than our backbone's fixed 336-step input (e.g.
Chinatown is 24 steps). Short series are zero-padded to seq_len, with a mask marking the
padded region so RevIN's per-instance normalization stats are computed over the real data
only (reusing the mask-aware RevIN built for imputation) — naive zero-padding would badly
skew normalization when most of the window is padding. Patch embeddings are pooled with a
weight proportional to how much real (non-padded) data each patch actually covers, so
patches that are pure padding don't dilute the pooled embedding.

Usage:
  python finetune_classification.py --ckpt outputs_foundation/<exp_name>_best.pth \
      --datasets Chinatown ECG200 GunPoint ItalyPowerDemand Coffee TwoLeadECG
"""

import os
import json
import argparse
import numpy as np
import torch

from aeon.datasets import load_classification
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from model_no_attn_upd import HyperTimeV2
from train import MODEL_CONFIGS


DEFAULT_DATASETS = ['Chinatown', 'ECG200', 'GunPoint', 'ItalyPowerDemand', 'Coffee', 'TwoLeadECG']


def parse_args():
    p = argparse.ArgumentParser(description='NE-Time Classification (frozen embeddings + SVM)')
    p.add_argument('--ckpt', type=str, required=True,
                    help='Path to a pretrained foundation checkpoint (*_best.pth)')
    p.add_argument('--datasets',   type=str, nargs='+', default=DEFAULT_DATASETS)
    p.add_argument('--data_path',  type=str, default='./data/aeon_data',
                    help='Where aeon looks for/caches UCR data. Compute nodes on most HPC '
                         'clusters have no internet — pre-download with '
                         'download_classification_data.py on a login node first.')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--svm_C',      type=float, default=1.0)
    p.add_argument('--svm_kernel', type=str, default='rbf')
    p.add_argument('--exp_name',   type=str, default='classification')
    p.add_argument('--output_dir', type=str, default='./outputs_classification')
    p.add_argument('--seed',       type=int, default=42)
    return p.parse_args()


def load_backbone(ckpt_path: str, device: torch.device):
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
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    print(f"Loaded backbone from {ckpt_path}")
    print(f"  size={ckpt_args.size}  seq_len={ckpt_args.seq_len}  "
          f"patch_size={ckpt_args.patch_size}  stride={ckpt_args.patch_stride}")
    print(f"  frozen backbone params: {sum(p.numel() for p in backbone.parameters()):,}")
    return backbone, ckpt_args


def pad_and_mask(X: np.ndarray, seq_len: int):
    """X: [N, 1, L] (univariate). Returns (x_padded [N, seq_len, 1], mask [N, seq_len, 1])
    with mask=1 for real positions, 0 for padding. Truncates from the front if L > seq_len."""
    N, C, L = X.shape
    assert C == 1, "multivariate UCR/UEA not handled here — univariate only"
    x = np.zeros((N, seq_len, 1), dtype=np.float32)
    mask = np.zeros((N, seq_len, 1), dtype=np.float32)
    keep = min(L, seq_len)
    x[:, :keep, 0] = X[:, 0, :keep]
    mask[:, :keep, 0] = 1.0
    return x, mask


def patch_validity(mask_1d: np.ndarray, num_patches: int, patch_size: int, stride: int) -> np.ndarray:
    """mask_1d: [seq_len] (1=real, 0=padding). Returns [num_patches] with the fraction of
    real (non-padded) timesteps each patch covers — used to weight pooling so patches that
    are pure padding don't dilute the embedding."""
    frac = np.zeros(num_patches, dtype=np.float32)
    for p in range(num_patches):
        start = p * stride
        frac[p] = mask_1d[start:start + patch_size].mean()
    return frac


@torch.no_grad()
def extract_embeddings(backbone: HyperTimeV2, X: np.ndarray, seq_len: int, patch_size: int,
                        stride: int, device: torch.device, batch_size: int) -> np.ndarray:
    """X: [N, 1, L] -> [N, d_model] pooled embeddings."""
    x_padded, mask = pad_and_mask(X, seq_len)
    num_patches = backbone.patch_embed.num_patches
    weights = np.stack([patch_validity(mask[i, :, 0], num_patches, patch_size, stride)
                         for i in range(len(X))])  # [N, num_patches]

    embeddings = []
    for i in range(0, len(x_padded), batch_size):
        xb = torch.from_numpy(x_padded[i:i + batch_size]).to(device)
        mb = torch.from_numpy(mask[i:i + batch_size]).to(device)
        wb = torch.from_numpy(weights[i:i + batch_size]).to(device)  # [b, num_patches]

        euclidean, _, _, _ = backbone.encode_patches(xb, mask=mb)  # [b, num_patches, d_model]
        wb_norm = wb / wb.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pooled = (euclidean * wb_norm.unsqueeze(-1)).sum(dim=1)     # [b, d_model]
        embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


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

    backbone, ckpt_args = load_backbone(args.ckpt, device)
    seq_len = ckpt_args.seq_len

    results = {}
    for name in args.datasets:
        try:
            X_train, y_train = load_classification(name, split='train', extract_path=args.data_path)
            X_test, y_test = load_classification(name, split='test', extract_path=args.data_path)
        except Exception as e:
            print(f"[SKIP] {name}: failed to load ({e}) — pre-download with "
                  f"download_classification_data.py on a login node first if this is a "
                  f"network error.")
            continue

        print(f"\n[{name}] train={X_train.shape} test={X_test.shape} "
              f"classes={sorted(set(y_train.tolist()))}")

        train_emb = extract_embeddings(backbone, X_train, seq_len, ckpt_args.patch_size,
                                        ckpt_args.patch_stride, device, args.batch_size)
        test_emb = extract_embeddings(backbone, X_test, seq_len, ckpt_args.patch_size,
                                       ckpt_args.patch_stride, device, args.batch_size)

        # Standardize embeddings before the SVM — standard practice, SVMs are scale-sensitive.
        scaler = StandardScaler()
        train_emb = scaler.fit_transform(train_emb)
        test_emb = scaler.transform(test_emb)

        clf = SVC(C=args.svm_C, kernel=args.svm_kernel, random_state=args.seed)
        clf.fit(train_emb, y_train)
        test_acc = clf.score(test_emb, y_test)
        train_acc = clf.score(train_emb, y_train)

        results[name] = {'train_accuracy': train_acc, 'test_accuracy': test_acc,
                          'n_train': len(X_train), 'n_test': len(X_test),
                          'n_classes': len(set(y_train.tolist()))}
        print(f"  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    out_path = os.path.join(args.output_dir, f"{args.exp_name}_classification_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if results:
        avg_acc = np.mean([r['test_accuracy'] for r in results.values()])
        print(f"Average test accuracy across {len(results)} datasets: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
