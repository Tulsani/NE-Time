"""
HyperTime v2 / NE-Time — Horizon-Aware Hyperbolic Time Series Forecaster

Changes in this revision (overfitting fixes):
  - geo_dropout forwarded to HyperbolicEncoder and HyperbolicDecoder
  - hyp_hidden_dim is now an explicit constructor arg (was hardcoded d_model*2)
    Default: d_model (was d_model*2), controlled via --hyp_hidden_scale in train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

from hyperbolic_ops import (
    CurvatureParam,
    HyperbolicEncoder,
    HyperbolicDecoder,
    tangent_space_fusion,
)
from decomposition_upd import RevIN, PatchEmbedding, MultiScaleDecomposer, FixedMADecomposer


# ─────────────────────────────────────────────────────────
#  Horizon Encoder
# ─────────────────────────────────────────────────────────

class HorizonEncoder(nn.Module):
    def __init__(self, cond_dim: int = 64, hidden_dim: int = 128, max_horizon: int = 1024):
        super().__init__()
        self.max_horizon = max_horizon
        self.cond_dim = cond_dim

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.weight_head = nn.Linear(hidden_dim, 3)
        self.cond_head   = nn.Linear(hidden_dim, cond_dim)

    def forward(self, horizon: int, batch_size: int, device: torch.device):
        h      = float(horizon)
        h_norm = h / self.max_horizon
        h_log  = math.log1p(h) / math.log1p(self.max_horizon)

        feat    = torch.tensor([[h_log, h_norm]], dtype=torch.float32, device=device)
        feat    = feat.expand(batch_size, -1)
        hidden  = self.net(feat)
        weights = F.softmax(self.weight_head(hidden), dim=-1)
        cond    = self.cond_head(hidden)
        return weights, cond


# ─────────────────────────────────────────────────────────
#  Temporal Projector
# ─────────────────────────────────────────────────────────

class TemporalProjector(nn.Module):
    MAX_TEMPORAL = 720

    def __init__(self, num_patches: int, d_model: int, cond_dim: int,
                 max_pred_len: int = 720, proj_hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.max_pred_len = max_pred_len
        self.proj_hidden  = proj_hidden

        self.patch_compress = nn.Linear(num_patches, proj_hidden, bias=True)
        self.time_expand    = nn.Linear(proj_hidden, self.MAX_TEMPORAL, bias=True)
        self.cond_proj      = nn.Linear(cond_dim, proj_hidden)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, pred_len: int) -> torch.Tensor:
        compressed = self.patch_compress(x.transpose(1, 2))
        out        = self.time_expand(compressed).transpose(1, 2)
        cond_h     = self.cond_proj(cond)
        cond_e     = cond_h.unsqueeze(1).expand(-1, compressed.shape[1], -1)
        bias       = self.time_expand(cond_e).transpose(1, 2)
        out        = out + bias
        out        = self.drop(self.norm(out))
        return out[:, :pred_len, :]


# ─────────────────────────────────────────────────────────
#  Patch Unfold — reconstruction head for imputation
# ─────────────────────────────────────────────────────────

class PatchUnfold(nn.Module):
    """Inverse of PatchEmbedding: patch-space [B, num_patches, d_model] -> reconstructed
    timestep-space [B, seq_len, out_dim].

    Unlike TemporalProjector (which compresses all patches into one global summary and
    expands it to arbitrary future timesteps, needed for forecasting beyond the input
    window), imputation reconstructs the *same* seq_len positions the patches already
    cover — so each patch is projected directly back to its own patch_size timesteps via
    a per-patch linear map, and overlapping patches (stride < patch_size) are folded back
    together by averaging. A local, structurally simpler inverse than TemporalProjector's
    global compress-then-expand, matching what the task actually needs.
    """

    def __init__(self, seq_len: int, patch_size: int, stride: int, num_patches: int,
                 d_model: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.seq_len     = seq_len
        self.patch_size  = patch_size
        self.stride      = stride
        self.num_patches = num_patches
        self.out_dim     = out_dim

        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_size * out_dim),
        )
        self.dropout = nn.Dropout(dropout)

        # How many patches cover each timestep, precomputed once — divides the summed
        # overlap-add contributions down to an average. Registered as a buffer (not a
        # parameter) so it moves with .to(device) but never gets gradients/optimizer state.
        counts = torch.zeros(seq_len)
        for p in range(num_patches):
            start = p * stride
            counts[start:start + patch_size] += 1
        self.register_buffer('counts', counts.clamp(min=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, num_patches, d_model] -> [B, seq_len, out_dim]"""
        B = x.shape[0]
        x = self.dropout(self.proj(x))                                   # [B, num_patches, patch_size*out_dim]
        x = x.view(B, self.num_patches, self.patch_size, self.out_dim)

        out = torch.zeros(B, self.seq_len, self.out_dim, device=x.device, dtype=x.dtype)
        for p in range(self.num_patches):
            start = p * self.stride
            out[:, start:start + self.patch_size, :] = \
                out[:, start:start + self.patch_size, :] + x[:, p, :, :]
        return out / self.counts.view(1, -1, 1)


# ─────────────────────────────────────────────────────────
#  Main model
# ─────────────────────────────────────────────────────────

class HyperTimeV2(nn.Module):
    """
    NE-Time / HyperTimeV2

    New constructor args vs previous version:
      geo_dropout   (float, default 0.2) — dropout on tangent vectors
                    before expmap0 in encoder, after logmap0 in decoder
      hyp_hidden_dim (int, default None → d_model) — hidden dim of the
                    hyperbolic MLP encoder/decoder.  Was hardcoded d_model*2.
                    Setting this to d_model cuts encoder params ~50% and is
                    the primary lever against overfitting.
    """

    def __init__(
        self,
        input_dim:      int   = 7,
        seq_len:        int   = 336,
        max_pred_len:   int   = 720,
        patch_size:     int   = 16,
        stride:         int   = 8,
        d_model:        int   = 128,
        hyp_dim:        int   = 64,
        cond_dim:       int   = 64,
        dropout:        float = 0.1,
        geo_dropout:    float = 0.2,      # NEW: tangent-space dropout
        hyp_hidden_dim: int   = None,     # NEW: encoder hidden dim (default=d_model)
        proj_hidden:    int   = 64,       # NEW: TemporalProjector bottleneck width (was hardcoded)
        c_global_init:  float = 0.5,
        c_meso_init:    float = 1.0,
        c_local_init:   float = 2.0,
        geometry:       str   = 'hyperbolic',  # 'hyperbolic' | 'euclidean' (ablation control)
    ):
        super().__init__()

        assert geometry in ('hyperbolic', 'euclidean'), \
            f"geometry must be 'hyperbolic' or 'euclidean', got {geometry!r}"
        self.geometry = geometry
        use_hyp = (geometry == 'hyperbolic')

        self.input_dim    = input_dim
        self.seq_len      = seq_len
        self.max_pred_len = max_pred_len
        self.d_model      = d_model
        self.hyp_dim      = hyp_dim

        # If not specified, default to d_model (was d_model*2 previously)
        if hyp_hidden_dim is None:
            hyp_hidden_dim = d_model

        # 1. Normalisation
        self.revin = RevIN(num_features=input_dim)

        # 2. Patch embedding
        self.patch_embed = PatchEmbedding(
            seq_len=seq_len, patch_size=patch_size, stride=stride,
            in_dim=input_dim, d_model=d_model, dropout=dropout,
        )
        num_patches = self.patch_embed.num_patches

        # 3. Decomposition (fixed MA — no learnable params)
        self.decomposer = FixedMADecomposer(num_patches=num_patches)

        # 4. Learnable curvatures
        self.c_global = CurvatureParam(c_global_init)
        self.c_meso   = CurvatureParam(c_meso_init)
        self.c_local  = CurvatureParam(c_local_init)
        self.c_fusion = CurvatureParam(1.0)

        # 5. Encoders — hyperbolic (default) or Euclidean-ablation control.
        # Identical MLP shapes either way (same param count); geometry=='euclidean'
        # simply skips expmap0 at the end of forward(), isolating the effect of the
        # Poincaré-ball mapping itself from the rest of the architecture (decomposer,
        # multi-scale fusion, horizon conditioning, etc., all unchanged).
        self.enc_global = HyperbolicEncoder(
            d_model, hyp_hidden_dim, hyp_dim, self.c_global,
            dropout=dropout, geo_dropout=geo_dropout, hyperbolic=use_hyp)
        self.enc_meso   = HyperbolicEncoder(
            d_model, hyp_hidden_dim, hyp_dim, self.c_meso,
            dropout=dropout, geo_dropout=geo_dropout, hyperbolic=use_hyp)
        self.enc_local  = HyperbolicEncoder(
            d_model, hyp_hidden_dim, hyp_dim, self.c_local,
            dropout=dropout, geo_dropout=geo_dropout, hyperbolic=use_hyp)

        # 6. Horizon encoder
        self.horizon_enc = HorizonEncoder(
            cond_dim=cond_dim, hidden_dim=d_model, max_horizon=max_pred_len)

        # 7. Decoder — hyperbolic or Euclidean-ablation control (see note above)
        self.decoder = HyperbolicDecoder(
            hyp_dim, hyp_hidden_dim, d_model, self.c_fusion,
            dropout=dropout, geo_dropout=geo_dropout, hyperbolic=use_hyp)

        # 8. Temporal projector
        self.temporal_proj = TemporalProjector(
            num_patches=num_patches,
            d_model=d_model,
            cond_dim=cond_dim,
            max_pred_len=max_pred_len,
            proj_hidden=proj_hidden,
            dropout=dropout,
        )

        # 9. Extra dropout before leaving hyperbolic space
        self.fusion_drop = nn.Dropout(dropout * 2)

        # 10. Output projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode_patches(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        cond_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Shared encoder path: [B,T,C] -> patch-space representation [B,num_patches,d_model].

        Factored out of forward() so other task heads (e.g. an imputation reconstruction
        head) can reuse the identical RevIN -> patch_embed -> decomposer -> hyperbolic
        encode/fuse/decode pipeline with a different output head, without duplicating it.

        `mask` (optional, [B,T,C] or [B,T,1], 1=observed/0=masked) is forwarded to RevIN so
        per-instance normalization stats aren't biased by masked-out (zeroed) positions —
        matters for imputation, irrelevant (mask=None) for forecasting.

        `cond_len` selects which trained horizon-conditioning to reuse for scale-mixing.
        For forecasting this is the actual target `pred_len`. Imputation has no real
        "horizon" — it defaults to `self.seq_len` (336, itself one of the four pretrained
        horizons), repurposing the model's learned 336-step scale-mixing preference for
        reconstruction rather than forecasting, without needing any new fusion params.

        Returns (euclidean, cond, scale_weights, intermediates) — intermediates carries
        h_global/h_meso/h_local/h_fused, matching what forward()'s `return_intermediates`
        used to expose directly.
        """
        if cond_len is None:
            cond_len = self.seq_len
        B, T, C = x.shape

        x = self.revin.normalise(x, mask=mask)
        patches = self.patch_embed(x)

        g_patches, m_patches, l_patches = self.decomposer(patches)

        h_global = self.enc_global(g_patches)
        h_meso   = self.enc_meso(m_patches)
        h_local  = self.enc_local(l_patches)

        scale_weights, cond = self.horizon_enc(cond_len, B, x.device)
        scale_weights_exp   = scale_weights.unsqueeze(1).expand(-1, patches.shape[1], -1)

        h_fused   = tangent_space_fusion(
            [h_global, h_meso, h_local], scale_weights_exp, self.c_fusion.c,
            hyperbolic=(self.geometry == 'hyperbolic'))

        h_fused   = self.fusion_drop(h_fused)
        euclidean = self.decoder(h_fused)

        intermediates = {
            'h_global': h_global.detach(), 'h_meso': h_meso.detach(),
            'h_local': h_local.detach(), 'h_fused': h_fused.detach(),
        }
        return euclidean, cond, scale_weights, intermediates

    def forward(
        self,
        x: torch.Tensor,
        pred_len: Optional[int] = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        if pred_len is None:
            pred_len = self.max_pred_len

        euclidean, cond, scale_weights, intermediates = self.encode_patches(x, cond_len=pred_len)

        temporal  = self.temporal_proj(euclidean, cond, pred_len)
        out       = self.out_proj(temporal)
        out       = self.revin.denormalise(out)

        info = {
            'scale_weights': scale_weights.detach(),
            'curvatures': {
                'global': self.c_global.c.item(),
                'meso':   self.c_meso.c.item(),
                'local':  self.c_local.c.item(),
            }
        }
        if return_intermediates:
            info.update(intermediates)

        return out, info

    def param_count(self) -> Dict[str, int]:
        def count(module):
            return sum(p.numel() for p in module.parameters())
        return {
            'revin':          count(self.revin),
            'patch_embed':    count(self.patch_embed),
            'decomposer':     count(self.decomposer),
            'curvatures':     count(self.c_global) + count(self.c_meso) +
                              count(self.c_local)  + count(self.c_fusion),
            'hyp_encoders':   count(self.enc_global) + count(self.enc_meso) +
                              count(self.enc_local),
            'horizon_enc':    count(self.horizon_enc),
            'hyp_decoder':    count(self.decoder),
            'temporal_proj':  count(self.temporal_proj),
            'out_proj':       count(self.out_proj),
            'TOTAL':          count(self),
        }


# ─────────────────────────────────────────────────────────
#  Loss
# ─────────────────────────────────────────────────────────

class HorizonWeightedLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.7, mae_weight: float = 0.3):
        super().__init__()
        self.mse_w = mse_weight
        self.mae_w = mae_weight

    def forward(self, pred, target, horizon=None):
        mse  = F.mse_loss(pred, target)
        mae  = F.l1_loss(pred, target)
        return self.mse_w * mse + self.mae_w * mae


# ─────────────────────────────────────────────────────────
#  Self-tests
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("HyperTime v2 / NE-Time model tests (overfitting-fix revision)")
    print("=" * 55)

    B, T, C = 4, 336, 7

    for hyp_hidden_scale, label in [(1.0, "d_model (new default)"),
                                     (2.0, "d_model*2 (old default)")]:
        model = HyperTimeV2(
            input_dim=C, seq_len=T, max_pred_len=720,
            patch_size=16, stride=8,
            d_model=128, hyp_dim=64, cond_dim=64,
            geo_dropout=0.2,
            hyp_hidden_dim=int(128 * hyp_hidden_scale),
        )
        params = model.param_count()
        print(f"\nhyp_hidden = {label}")
        for k, v in params.items():
            print(f"  {k:<18} {v:>8,}")

    # Forward pass
    model = HyperTimeV2(
        input_dim=C, seq_len=T, max_pred_len=720,
        d_model=128, hyp_dim=64, cond_dim=64,
        geo_dropout=0.2, hyp_hidden_dim=128,
    )
    x = torch.randn(B, T, C)
    print("\nForward pass (train mode):")
    model.train()
    for H in [96, 192, 336, 720]:
        pred, info = model(x, pred_len=H)
        assert pred.shape == (B, H, C)
        w = info['scale_weights'][0]
        print(f"  H={H:4d}  pred={pred.shape}  "
              f"w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f}]")

    # Grad flow
    x_grad = torch.randn(B, T, C, requires_grad=True)
    pred, _ = model(x_grad, pred_len=96)
    pred.sum().backward()
    assert x_grad.grad is not None
    print(f"\nGrad norm: {x_grad.grad.norm().item():.4f}  ✓")

    print("\n✓ All model tests passed")
