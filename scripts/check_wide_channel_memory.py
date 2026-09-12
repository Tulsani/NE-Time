"""
Quick empirical check: does a single forward+backward pass on an ECL- or
Traffic-shaped batch fit in GPU memory for the `nano` model?

Earlier caution about excluding ECL (321ch) / Traffic (862ch) from the pretrain
corpus was based on the `to_ci` batch-size blowup (B*C) causing an OOM during
*eval* on the much bigger `medium` model (254K params) — compounded by a
since-fixed redundant-eval-loop bug, not a verified measurement of *training*
(forward+backward) memory for the much smaller `nano` model actually in use now.
This script measures it directly instead of re-applying that old assumption.

Usage (on a GPU node):
    python scripts/check_wide_channel_memory.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_no_attn_upd import HyperTimeV2
from train import to_ci, from_ci

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = HyperTimeV2(
    input_dim=1, seq_len=336, max_pred_len=720, patch_size=16, stride=8,
    d_model=64, hyp_dim=32, cond_dim=32, proj_hidden=32,
).to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")


def check(name: str, channels: int, batch_size: int = 32):
    x = torch.randn(batch_size, 336, channels, device=device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    pred, _ = model(to_ci(x), pred_len=720)
    pred = from_ci(pred, channels)
    loss = pred.mean()
    loss.backward()
    model.zero_grad(set_to_none=True)

    effective_batch = batch_size * channels
    if device.type == 'cuda':
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"{name:10s}  channels={channels:4d}  effective_batch(B*C)={effective_batch:6,d}  "
              f"peak_mem={peak_gb:.2f} GB")
    else:
        print(f"{name:10s}  channels={channels:4d}  effective_batch(B*C)={effective_batch:6,d}  "
              f"(no CUDA device — forward+backward completed without error, can't measure peak mem here)")


check("ECL", channels=321)
check("Traffic", channels=862)

if device.type == 'cuda':
    print(f"\nA100 80GB has ~80 GB total — compare peak_mem above against that "
          f"(plus headroom for optimizer states, activations across the full "
          f"training loop, and other datasets' loaders).")
