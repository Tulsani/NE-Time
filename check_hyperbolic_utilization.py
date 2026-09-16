"""
Diagnostic: is the trained hyperbolic backbone actually exploiting the Poincare ball's
curvature, or do its tangent-space representations stay close enough to the origin that
expmap0 is numerically near-identity?

Motivation: the Euclidean-ablation control (GEOMETRY=euclidean) is reportedly converging
to ~the same val_loss as the hyperbolic backbone. Before treating that as "geometry doesn't
matter," check whether the hyperbolic model ever produces large-norm points on the ball in
the first place — HyperbolicEncoder's final Linear is deliberately init'd with std=0.01
("points start near origin"), and expmap0(t,c) ~= t for small ||t|| (verified locally:
ratio ||expmap0(t)||/||t|| > 0.99 for ||t|| < 0.1, only dropping meaningfully past ||t||
~ 0.5-1.0 at these curvature scales). If trained ||h_global||/||h_meso||/||h_local|| stay
small, the hyperbolic and Euclidean models are close to the same function almost everywhere
they're actually evaluated — a real architectural finding, not an ablation bug.

Usage:
  python check_hyperbolic_utilization.py --ckpt outputs_foundation/nano_wecm1_sw_lr3e4_best.pth \
      --dataset ETTh2
"""

import argparse
import torch

from dataset import create_dataloaders
from train import build_model
from hyperbolic_ops import logmap0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--dataset', type=str, default='ETTh2')
    p.add_argument('--data_path', type=str, default='./data')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_batches', type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu')

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt['args'])
    print(f"Checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f}, "
          f"geometry={getattr(ckpt_args, 'geometry', 'hyperbolic')}")

    model = build_model(ckpt_args, input_dim=1, horizon=max(ckpt_args.horizons))
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    print(f"Trained curvatures: global={model.c_global.c.item():.4f} "
          f"meso={model.c_meso.c.item():.4f} local={model.c_local.c.item():.4f} "
          f"fusion={model.c_fusion.c.item():.4f}")

    _, val_loader, _, _ = create_dataloaders(
        dataset_name=args.dataset, root_path=args.data_path, seq_len=ckpt_args.seq_len,
        horizons=ckpt_args.horizons, batch_size=args.batch_size, num_workers=0,
        train_stride=1,
    )

    from train import to_ci

    stats = {k: [] for k in ['h_global', 'h_meso', 'h_local', 'h_fused']}
    tangent_stats = {k: [] for k in ['h_global', 'h_meso', 'h_local']}

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.n_batches:
                break
            x = batch['x'].to(device)
            C = x.shape[-1]
            x_ci = to_ci(x)
            _, _, _, intermediates = model.encode_patches(x_ci, cond_len=ckpt_args.seq_len)
            for k in stats:
                h = intermediates[k]
                stats[k].append(h.norm(dim=-1).flatten())

            # Recover pre-expmap0 tangent-space norm for the three scale encoders
            # (only meaningful if this checkpoint is hyperbolic; harmless no-op check
            # for a euclidean checkpoint where h_* IS already the tangent vector).
            is_hyp = (getattr(ckpt_args, 'geometry', 'hyperbolic') == 'hyperbolic')
            for k, c_param in [('h_global', model.c_global), ('h_meso', model.c_meso),
                                ('h_local', model.c_local)]:
                h = intermediates[k]
                t = logmap0(h, c_param.c) if is_hyp else h
                tangent_stats[k].append(t.norm(dim=-1).flatten())

    print(f"\n[{args.dataset}] point norms on the ball (||h||, bounded by 1/sqrt(c)):")
    for k, vals in stats.items():
        v = torch.cat(vals)
        print(f"  {k:<10} mean={v.mean().item():.4f}  max={v.max().item():.4f}  "
              f"p95={v.quantile(0.95).item():.4f}")

    print(f"\nPre-expmap0 tangent-vector norms (||t|| fed into expmap0):")
    for k, vals in tangent_stats.items():
        v = torch.cat(vals)
        print(f"  {k:<10} mean={v.mean().item():.4f}  max={v.max().item():.4f}  "
              f"p95={v.quantile(0.95).item():.4f}")
        # ratio ||expmap0(t)|| / ||t|| at the mean norm, using this encoder's trained c
        c = {'h_global': model.c_global, 'h_meso': model.c_meso,
             'h_local': model.c_local}[k].c
        sqrt_c = c.sqrt().item()
        mean_norm = v.mean().item()
        import math
        ratio = math.tanh(sqrt_c * mean_norm) / (sqrt_c * mean_norm) if mean_norm > 1e-6 else 1.0
        print(f"    -> at mean ||t||, expmap0 nonlinearity ratio = {ratio:.4f} "
              f"(1.0 = numerically ~identity, i.e. curvature barely used)")

    print("\nInterpretation: ratio close to 1.0 across all three scales means the trained "
          "hyperbolic model rarely leaves the near-origin, ~linear regime of the Poincare "
          "ball — which would explain why an Euclidean model of the same capacity reaches "
          "similar val_loss (the two are close to the same function almost everywhere they're "
          "actually evaluated). A ratio well below 1.0 would mean the model IS exploiting "
          "curvature, and the val_loss similarity needs a different explanation (e.g. val_loss "
          "just isn't a sensitive-enough probe here — check zero-shot numbers instead).")


if __name__ == "__main__":
    main()
