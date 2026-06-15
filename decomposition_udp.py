"""
Multi-Scale Decomposition (v2)

Key changes over v1:
- PatchEmbedding front-end (à la PatchTST) before decomposition
  so the model reasons over subsequences not individual timesteps
- Decomposition happens *in patch space*, not raw time space
- RevIN (Reversible Instance Normalisation) for distribution shift
- LearnableDecomposer is the default — MovingAverage as fallback
- No FFT decomposer (numerically fragile, adds little over MA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


# ─────────────────────────────────────────────────────────
#  RevIN — Reversible Instance Normalisation
#  Kim et al. 2022 — essential for non-stationary series
# ─────────────────────────────────────────────────────────

class RevIN(nn.Module):
    """
    Normalise each instance independently, store stats, denorm output.
    Handles distribution shift without data leakage.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        self._mean = None
        self._std = None

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C]  →  normalised [B, T, C]"""
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = x.std(dim=1, keepdim=True, unbiased=False).clamp(min=self.eps).detach()
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.gamma + self.beta
        return x

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C]  →  original scale [B, T, C]"""
        if self.affine:
            x = (x - self.beta) / self.gamma.clamp(min=self.eps)
        x = x * self._std + self._mean
        return x


# ─────────────────────────────────────────────────────────
#  Patch Embedding
# ─────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits time series into overlapping patches and projects each to d_model.

    Inspired by PatchTST (Nie et al. 2023).
    Patch size P, stride S → num_patches = (seq_len - P) // S + 1

    This gives the model a local receptive field while reducing sequence
    length — the hyperbolic encoder then operates over patches, not
    individual timesteps. Much more parameter-efficient.
    """

    def __init__(self, seq_len: int, patch_size: int, stride: int,
                 in_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.num_patches = (seq_len - patch_size) // stride + 1
        self.d_model = d_model

        # Project patch (patch_size * in_dim) → d_model
        self.projection = nn.Sequential(
            nn.Linear(patch_size * in_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable positional embedding over patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]  →  patches: [B, num_patches, d_model]
        """
        B, T, C = x.shape
        P, S = self.patch_size, self.stride

        # Unfold: [B, num_patches, P, C]
        x_unfolded = x.unfold(1, P, S)  # [B, num_patches, C, P] after unfold on dim 1
        # unfold on dim=1 gives [B, n_patches, C, P] — we want [B, n_patches, P*C]
        # Actually unfold(dimension, size, step): x is [B, T, C], unfold dim 1
        # → [B, n_patches, C, P]? No: [B, T, C].unfold(1, P, S) → [B, n_patches, C, P]
        # Correct reshape:
        x_unfolded = x_unfolded.permute(0, 1, 3, 2)  # [B, n_patches, P, C]
        x_flat = x_unfolded.reshape(B, -1, P * C)    # [B, n_patches, P*C]

        patches = self.projection(x_flat)             # [B, n_patches, d_model]
        patches = patches + self.pos_embed
        return self.dropout(patches)


# ─────────────────────────────────────────────────────────
#  Multi-scale decomposition over patch representations
# ─────────────────────────────────────────────────────────

class MultiScaleDecomposer(nn.Module):
    """
    Decomposes patch representations into three scales.

    Operating in patch space (not raw time) means:
    - Global window = patches covering long-range trends
    - Meso window = patches covering seasonal cycles
    - Local = residual high-frequency detail

    Uses depthwise 1D convolutions — parameter-efficient and
    gradient-stable (no recurrence).
    """

    def __init__(self, d_model: int, num_patches: int,
                 global_kernel: int = None, meso_kernel: int = None,
                 dropout: float = 0.1):
        super().__init__()

        # Default kernels: roughly 1/4 and 1/8 of patch sequence
        if global_kernel is None:
            global_kernel = max(3, num_patches // 4)
            global_kernel = global_kernel if global_kernel % 2 == 1 else global_kernel + 1
        if meso_kernel is None:
            meso_kernel = max(3, num_patches // 8)
            meso_kernel = meso_kernel if meso_kernel % 2 == 1 else meso_kernel + 1

        self.global_kernel = global_kernel
        self.meso_kernel = meso_kernel

        pad_g = global_kernel // 2
        pad_m = meso_kernel // 2

        # Depthwise separable convs — low parameter count
        self.global_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, global_kernel, padding=pad_g, groups=d_model, bias=False),
            nn.Conv1d(d_model, d_model, 1, bias=True),  # pointwise mix
            nn.LayerNorm(d_model),  # will apply after transpose
        )
        self.meso_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, meso_kernel, padding=pad_m, groups=d_model, bias=False),
            nn.Conv1d(d_model, d_model, 1, bias=True),
            nn.LayerNorm(d_model),
        )

        # Gating: learn how much of each component to pass through
        self.global_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.meso_gate  = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

        self.dropout = nn.Dropout(dropout)
        self.norm_local = nn.LayerNorm(d_model)

    def _apply_conv_block(self, block: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D]  →  [B, N, D]"""
        # Conv1d expects [B, C, L]
        xt = x.transpose(1, 2)                 # [B, D, N]
        xt = block[0](xt)                       # depthwise
        xt = block[1](xt)                       # pointwise
        out = xt.transpose(1, 2)                # [B, N, D]
        out = block[2](out)                     # LayerNorm
        return out

    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        patches: [B, num_patches, d_model]

        Returns:
            global_comp:  [B, num_patches, d_model]  — long-range trends
            meso_comp:    [B, num_patches, d_model]  — seasonal / periodic
            local_comp:   [B, num_patches, d_model]  — local residual
        """
        N = patches.shape[1]

        global_raw = self._apply_conv_block(self.global_conv, patches)
        meso_raw   = self._apply_conv_block(self.meso_conv, patches)

        # Gated components
        global_comp = self.global_gate(global_raw) * global_raw
        meso_comp   = self.meso_gate(meso_raw) * meso_raw

        # Trim to exact length (conv padding may add/remove 1)
        global_comp = global_comp[:, :N, :]
        meso_comp   = meso_comp[:, :N, :]

        # Local = residual after subtracting smoothed components
        local_comp = self.norm_local(patches - global_comp - meso_comp)

        return self.dropout(global_comp), self.dropout(meso_comp), self.dropout(local_comp)


# ─────────────────────────────────────────────────────────
#  Self-tests
# ─────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────
#  FixedMADecomposer — zero learnable parameters
# ─────────────────────────────────────────────────────────

class FixedMADecomposer(nn.Module):
    """
    Decomposes patch representations into three scales using
    FIXED moving averages — no learnable parameters.

    Why fixed: the learnable conv decomposer was fitting training
    patterns in epoch 1, giving a shortcut that bypassed the hyperbolic
    encoders. Best val always at epoch 1 confirmed this. Removing all
    learnable params here forces all learning into the geometry.

      global:  slow MA over patches  — long-range trends
      meso:    medium MA of residual — seasonal patterns
      local:   residual after both   — high-frequency detail

    Box filters applied with replicate padding; output == input length.
    """

    def __init__(self, num_patches: int,
                 global_frac: float = 0.5,
                 meso_frac:   float = 0.25):
        super().__init__()

        def odd(n):
            n = max(3, n)
            return n if n % 2 == 1 else n + 1

        self.global_k    = odd(int(num_patches * global_frac))
        self.meso_k      = odd(int(num_patches * meso_frac))
        self.num_patches = num_patches

    def _ma(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Fixed depthwise 1D box MA. x: [B, D, N] -> [B, D, N]"""
        pad   = k // 2
        x_pad = F.pad(x, (pad, pad), mode='replicate')
        D     = x.shape[1]
        w     = torch.ones(D, 1, k, device=x.device, dtype=x.dtype) / k
        out   = F.conv1d(x_pad, w, groups=D)
        return out[:, :, :self.num_patches]

    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """patches: [B, N, D] -> global, meso, local each [B, N, D]"""
        pt          = patches.transpose(1, 2)           # [B, D, N]
        global_comp = self._ma(pt, self.global_k).transpose(1, 2)
        residual    = patches - global_comp
        meso_comp   = self._ma(residual.transpose(1,2), self.meso_k).transpose(1, 2)
        local_comp  = residual - meso_comp
        return global_comp, meso_comp, local_comp

if __name__ == "__main__":
    print("=" * 55)
    print("Decomposition v2 unit tests")
    print("=" * 55)

    B, T, C = 4, 336, 7
    d_model = 64
    patch_size = 16
    stride = 8

    x = torch.randn(B, T, C)

    # RevIN
    revin = RevIN(num_features=C)
    x_norm = revin.normalise(x)
    x_rec = revin.denormalise(x_norm)
    err = (x - x_rec).abs().max().item()
    assert err < 1e-4, f"RevIN round-trip err={err}"
    print(f"[OK] RevIN round-trip  err={err:.2e}")

    # PatchEmbedding
    pe = PatchEmbedding(seq_len=T, patch_size=patch_size, stride=stride,
                        in_dim=C, d_model=d_model)
    patches = pe(x)
    n_patches_expected = (T - patch_size) // stride + 1
    assert patches.shape == (B, n_patches_expected, d_model), \
        f"PatchEmbed shape {patches.shape}, expected {(B, n_patches_expected, d_model)}"
    print(f"[OK] PatchEmbedding  {x.shape} → {patches.shape}")

    # MultiScaleDecomposer
    decomp = MultiScaleDecomposer(d_model=d_model, num_patches=patches.shape[1])
    g, m, l = decomp(patches)
    assert g.shape == patches.shape
    assert m.shape == patches.shape
    assert l.shape == patches.shape
    # Rough sanity: global should be smoother than local
    print(f"[OK] MultiScaleDecomposer  shapes={g.shape}")
    print(f"     global std={g.std():.4f}  meso std={m.std():.4f}  local std={l.std():.4f}")

    # Gradient test
    x_grad = torch.randn(B, T, C, requires_grad=True)
    x_norm = revin.normalise(x_grad)
    patches = pe(x_norm)
    g, m, l = decomp(patches)
    (g + m + l).sum().backward()
    assert x_grad.grad is not None
    print(f"[OK] Gradients flow end-to-end through decomposition")

    print("\n✓ All decomposition tests passed")