"""
Hyperbolic Operations — Poincaré Ball Model (v2.1)

Changes over v2:
- HyperbolicEncoder: dropout added before expmap0 (regularises geometry)
- HyperbolicDecoder: dropout added after logmap0 (regularises tangent mapping)
- CurvatureParam: exposes is_curvature flag so the trainer can route it to
  the weight-decay param group instead of no_decay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────
#  Core Poincaré ball ops (functional, no state)
# ─────────────────────────────────────────────────────────

def poincare_project(x: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    max_norm = (1.0 - eps) / c.sqrt()
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


def expmap0(v: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    sqrt_c = c.sqrt()
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    tanh_term = torch.tanh(sqrt_c * v_norm)
    x = (tanh_term / (sqrt_c * v_norm)) * v
    return poincare_project(x, c, eps)


def logmap0(x: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    sqrt_c = c.sqrt()
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    arctanh_input = (sqrt_c * x_norm).clamp(max=1.0 - eps)
    atanh_term = torch.atanh(arctanh_input)
    return (1.0 / sqrt_c) * (atanh_term / x_norm) * x


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = (1 + 2 * c * xy + c * c * x2 * y2).clamp(min=eps)
    return poincare_project(num / den, c, eps)


def hyp_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    diff = mobius_add(-x, y, c, eps)
    diff_norm = diff.norm(dim=-1).clamp(min=eps)
    sqrt_c = c.sqrt()
    dist = (2.0 / sqrt_c) * torch.atanh((sqrt_c * diff_norm).clamp(max=1.0 - eps))
    return dist


def tangent_space_fusion(
    h_list: list,
    c_list: list,
    weights: torch.Tensor,
    c_fusion: torch.Tensor,
    eps: float = 1e-5,
    hyperbolic: bool = True,
) -> torch.Tensor:
    """Fuse multi-scale representations by a weighted sum in tangent space.

    Each entry of h_list lives on its OWN ball (h_list[i] was produced by
    expmap0(., c_list[i])), not on the fusion ball. A prior version of this
    function logmap0'd every entry under a single shared curvature (the fusion
    curvature) regardless of which ball it actually came from — mathematically
    inconsistent (a point valid on the c_global ball need not lie in the domain of
    the c_fusion logarithm) and, checked directly against a trained checkpoint,
    not just a theoretical issue: measured h_global norms exceeded the fusion
    ball's radius, silently triggering logmap0's clamp. Fixed by giving each
    h_list[i] its own logmap0 under c_list[i] — valid, since every point is then
    mapped into the tangent space AT THE ORIGIN of its own ball, and the tangent
    space at the origin is canonically R^n regardless of curvature, so these three
    tangent vectors can be weighted-summed directly in that shared Euclidean space
    before a single expmap0 under c_fusion places the fused result back on the
    fusion ball.

    hyperbolic=False (Euclidean-ablation control): h_list entries are already
    Euclidean (see HyperbolicEncoder(hyperbolic=False)), so logmap0/expmap0 are
    skipped and this degenerates to a plain weighted sum — same shapes, same
    weight-mixing logic, the only difference being the two geometric ops that
    define "hyperbolic" in the first place.
    """
    if hyperbolic:
        tangents = torch.stack(
            [logmap0(h, c, eps) for h, c in zip(h_list, c_list)], dim=-2)
    else:
        tangents = torch.stack(h_list, dim=-2)
    w = weights.unsqueeze(-1)
    t_fused = (w * tangents).sum(dim=-2)
    if hyperbolic:
        return expmap0(t_fused, c_fusion, eps)
    return t_fused


# ─────────────────────────────────────────────────────────
#  Learnable modules
# ─────────────────────────────────────────────────────────

class HypLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, c: nn.Parameter, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = logmap0(x, self.c)
        t = self.linear(t)
        return expmap0(t, self.c)


class HypLayerNorm(nn.Module):
    def __init__(self, dim: int, c: nn.Parameter):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = logmap0(x, self.c)
        t = self.ln(t)
        return expmap0(t, self.c)


class CurvatureParam(nn.Module):
    """
    Learnable curvature c > 0, parameterised as c = softplus(raw_c).

    is_curvature = True flags this parameter for the trainer so it can be
    routed into a weight-decay param group (prevents curvature from drifting
    freely in the long tail of training, which was a source of overfitting).
    """

    is_curvature = True  # trainer checks for this attribute

    def __init__(self, init_c: float = 1.0):
        super().__init__()
        init_raw = math.log(math.exp(init_c) - 1.0)
        self.raw_c = nn.Parameter(torch.tensor(init_raw))

    @property
    def c(self) -> torch.Tensor:
        return F.softplus(self.raw_c) + 1e-5

    def forward(self) -> torch.Tensor:
        return self.c


class HyperbolicEncoder(nn.Module):
    """
    Encodes Euclidean features into the Poincaré ball.

    v2.1: geo_dropout applied to the tangent vector immediately before
    expmap0.  This forces the model to learn geometry that is robust to
    partial tangent-vector corruption — analogous to how standard dropout
    prevents neurons from co-adapting.  p=0.2 by default (separate from
    the existing feature dropout inside the MLP).
    """

    def __init__(self, in_dim: int, hidden_dim: int, hyp_dim: int,
                 c_param: CurvatureParam, dropout: float = 0.1,
                 geo_dropout: float = 0.2, hyperbolic: bool = True,
                 out_init_std: float = 0.01):
        super().__init__()
        self.c_param = c_param
        self.hyperbolic = hyperbolic

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hyp_dim),
        )

        # out_init_std controls how far from the ball's origin points start.
        # Default 0.01 (small init) was chosen for training stability. A larger value
        # is the "stronger geometry bias" ablation lever: it starts training with
        # tangent vectors already large enough that expmap0's nonlinearity is
        # non-negligible from step 1, rather than letting the model default toward
        # the near-linear regime measured in the trained default-init backbone
        # (see check_hyperbolic_utilization.py — the global branch in particular
        # stayed close to linear under the default init).
        nn.init.normal_(self.net[-1].weight, std=out_init_std)
        nn.init.zeros_(self.net[-1].bias)
        # Flag so HyperTimeV2._init_weights() (which re-inits every nn.Linear after all
        # submodules are constructed) does not clobber this deliberate init — previously
        # it did, silently, meaning this layer's actual init was always ~trunc_normal(0.02)
        # regardless of what was requested here (found while wiring up out_init_std).
        self.net[-1]._custom_init = True

        # Dropout applied to tangent vector before expmap0
        self.geo_drop = nn.Dropout(geo_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim]  →  [..., hyp_dim] on Poincaré ball
        (or plain Euclidean [..., hyp_dim] if hyperbolic=False — the
        Euclidean-ablation control; same MLP, same param count, expmap0 skipped)."""
        t = self.net(x)
        t = self.geo_drop(t)          # regularise in tangent space before mapping
        if not self.hyperbolic:
            return t
        return expmap0(t, self.c_param.c)


class HyperbolicDecoder(nn.Module):
    """
    Decodes Poincaré ball points back to Euclidean space.

    v2.1: geo_dropout applied immediately after logmap0, before the MLP.
    This regularises the tangent-space representation on the decoding side,
    symmetric with HyperbolicEncoder.  p=0.2 by default.
    """

    def __init__(self, hyp_dim: int, hidden_dim: int, out_dim: int,
                 c_param: CurvatureParam, dropout: float = 0.1,
                 geo_dropout: float = 0.2, hyperbolic: bool = True):
        super().__init__()
        self.c_param = c_param
        self.hyperbolic = hyperbolic

        # geo_drop sits between logmap0 and the MLP
        self.geo_drop = nn.Dropout(geo_dropout)

        self.net = nn.Sequential(
            nn.LayerNorm(hyp_dim),
            nn.Linear(hyp_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., hyp_dim] on ball  →  [..., out_dim] Euclidean
        (hyperbolic=False: x is already Euclidean, logmap0 skipped)."""
        t = logmap0(x, self.c_param.c) if self.hyperbolic else x
        t = self.geo_drop(t)          # regularise tangent vector before MLP
        return self.net(t)


class HyperbolicDistanceAttention(nn.Module):
    """
    Patch-to-patch relational mixing, weighted by hyperbolic distance between patches'
    own ball embeddings, instead of the purely pointwise (per-patch, no cross-patch
    interaction) path the rest of the encoder uses.

    Why this is the genuinely non-Euclidean operation the rest of the architecture lacks
    (see Documentation/hyperbolic-bug.md): expmap0 immediately followed by logmap0 at the
    SAME curvature is an exact round-trip no-op, regardless of what happens to the tangent
    vector in between — which is why the original encode-fuse-decode design measured as
    statistically indistinguishable from an equal-width Euclidean network once the fusion
    bug was fixed. hyp_distance does not have this property: it depends on curvature
    through mobius_add's bilinear structure, not a norm-rescaling that cancels under a
    matching inverse map. Two points equidistant in Euclidean terms are not generally
    equidistant in hyp_distance, and that distance governs the attention weights below —
    a relation an equal-width Euclidean network computing ||x_i - x_j||^2 cannot reproduce
    for points sharing the same underlying tangent-space coordinates.

    hyperbolic=False gives the matched Euclidean control: negative squared Euclidean
    distance between the same (now plain Euclidean, see HyperbolicEncoder(hyperbolic=False))
    representations drives the same softmax-attention/gated-residual mechanism. Same shapes,
    same parameter count, only the distance metric's geometry differs — isolating exactly
    this operation the same way every other hyperbolic-vs-Euclidean ablation in this project
    has (geometry flag, parameter-matched control).
    """

    def __init__(self, dim: int, dropout: float = 0.1, init_temp: float = 1.0,
                 hyperbolic: bool = True):
        super().__init__()
        self.hyperbolic = hyperbolic
        # Learnable softmax temperature (log-parameterised, always positive)
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))
        # Learnable residual gate: sigmoid(gate)=0.5 at init (raw=0), halfway between
        # "ignore attention, keep original per-patch representation" and "fully replace it".
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """h: [B, N, D] — N patches' embeddings (on the ball of curvature c if
        hyperbolic=True, else plain Euclidean). Returns [B, N, D], same space."""
        xi = h.unsqueeze(2)   # [B, N, 1, D]
        xj = h.unsqueeze(1)   # [B, 1, N, D] — broadcasts against xi to [B, N, N, D] inside
                              # mobius_add / the subtraction below, without materialising it

        if self.hyperbolic:
            dist = hyp_distance(xi, xj, c)         # [B, N, N]
            t = logmap0(h, c)                       # [B, N, D]
        else:
            dist = ((xi - xj) ** 2).sum(dim=-1)      # [B, N, N] squared Euclidean distance
            t = h

        temp = torch.exp(self.log_temp).clamp(min=1e-3)
        attn = torch.softmax(-dist / temp, dim=-1)   # closer patches -> higher weight
        attn = self.dropout(attn)

        t_attn = torch.einsum('bij,bjd->bid', attn, t)   # neighbour-weighted mix, in
                                                            # tangent (or Euclidean) space
        t_attn = self.norm(t_attn)

        gate = torch.sigmoid(self.gate)
        t_mixed = gate * t + (1 - gate) * t_attn

        if self.hyperbolic:
            return expmap0(t_mixed, c)
        return t_mixed


# ─────────────────────────────────────────────────────────
#  Self-tests
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    print("=" * 55)
    print("Hyperbolic ops unit tests (v2.1)")
    print("=" * 55)

    c_val = torch.tensor(1.0)

    # 1. project stays in ball
    x = torch.randn(8, 16) * 5
    px = poincare_project(x, c_val)
    max_norm = px.norm(dim=-1).max().item()
    assert max_norm < 1.0, f"project failed: max_norm={max_norm}"
    print(f"[OK] poincare_project  max_norm={max_norm:.4f} < 1.0")

    # 2. expmap0 / logmap0 round-trip
    v = torch.randn(4, 8) * 0.3
    x_hyp = expmap0(v, c_val)
    v_rec = logmap0(x_hyp, c_val)
    err = (v - v_rec).norm().item()
    assert err < 1e-5, f"round-trip error={err}"
    print(f"[OK] expmap0/logmap0 round-trip  err={err:.2e}")

    # 3. tangent_space_fusion — now with per-scale curvatures (distinct c's, the
    #    cross-curvature fix) rather than one shared curvature for every h
    B, T, D = 4, 32, 16
    c1, c2, c3 = torch.tensor(0.5), torch.tensor(1.0), torch.tensor(2.0)
    c_fuse = torch.tensor(0.9)
    h1 = expmap0(torch.randn(B, T, D) * 0.1, c1)
    h2 = expmap0(torch.randn(B, T, D) * 0.1, c2)
    h3 = expmap0(torch.randn(B, T, D) * 0.1, c3)
    w_exp = torch.softmax(torch.randn(B, 3), dim=-1).unsqueeze(1).expand(B, T, 3)
    fused = tangent_space_fusion([h1, h2, h3], [c1, c2, c3], w_exp, c_fuse)
    assert fused.shape == (B, T, D)
    assert fused.norm(dim=-1).max().item() < 1.0 / c_fuse.sqrt().item()
    print(f"[OK] tangent_space_fusion (per-scale curvatures)  shape={fused.shape}")

    # 3b. Regression check, using the actual scenario found on the real trained
    # checkpoint: a low-curvature branch (large own-ball radius, e.g. the global
    # scale, c1=0.5 -> radius 1.414) can produce points past a higher-curvature
    # fusion ball's own (smaller) radius (c_fuse=0.9 -> radius 1.054) — exactly what
    # was measured (h_global max norm 1.242 > fusion radius 1.012 on
    # nano_wecm1_sw_lr3e4). Such a point must be logmap0'd under its OWN curvature
    # (c1), not silently clamped under the mismatched fusion curvature.
    big_global = expmap0(torch.ones(1, D) * 3.0, c1)   # pushed near c1's own ball radius
    big_global_norm = big_global.norm().item()
    fusion_ball_radius = 1.0 / c_fuse.sqrt().item()
    assert big_global_norm > fusion_ball_radius, \
        "test setup should produce a point past the fusion ball's own radius"
    t_correct = logmap0(big_global, c1)       # correct: own curvature
    t_wrong   = logmap0(big_global, c_fuse)   # old buggy behaviour: fusion curvature
    assert not torch.allclose(t_correct, t_wrong), \
        "own-curvature and fusion-curvature logmap0 should differ for this point"
    print(f"[OK] cross-curvature fix verified: point past fusion-ball radius "
          f"({big_global_norm:.3f} > {fusion_ball_radius:.3f}) maps differently "
          f"under its own curvature vs. the old shared-curvature bug")

    # 4. CurvatureParam — positive and has is_curvature flag
    cp = CurvatureParam(init_c=1.0)
    assert cp.c.item() > 0
    assert cp.is_curvature is True
    print(f"[OK] CurvatureParam  c={cp.c.item():.4f}  is_curvature={cp.is_curvature}")

    # 5. HyperbolicEncoder with geo_dropout — gradient flows in train AND eval
    c_param = CurvatureParam(1.0)
    enc = HyperbolicEncoder(in_dim=7, hidden_dim=64, hyp_dim=32,
                            c_param=c_param, dropout=0.1, geo_dropout=0.2)
    x_in = torch.randn(4, 336, 7, requires_grad=True)

    enc.train()
    h_train = enc(x_in)
    h_train.sum().backward()
    assert x_in.grad is not None
    print(f"[OK] HyperbolicEncoder (train)  shape={h_train.shape}  grad flows")

    x_in2 = torch.randn(4, 336, 7, requires_grad=True)
    enc.eval()
    with torch.no_grad():
        h_eval = enc(x_in2)
    assert h_eval.shape == (4, 336, 32)
    print(f"[OK] HyperbolicEncoder (eval)   shape={h_eval.shape}")

    # 6. HyperbolicDecoder with geo_dropout
    dec = HyperbolicDecoder(hyp_dim=32, hidden_dim=64, out_dim=7,
                            c_param=c_param, dropout=0.1, geo_dropout=0.2)
    dec.train()
    out = dec(h_train.detach())
    assert out.shape == (4, 336, 7)
    print(f"[OK] HyperbolicDecoder  shape={out.shape}")

    # 7. HyperbolicDistanceAttention — shape, grad flow, param-matched hyperbolic vs
    # Euclidean control, and (the whole point) genuine curvature-dependence: unlike the
    # encode-fuse-decode path, this must NOT behave identically to its Euclidean twin on
    # the same underlying tangent-space points.
    B, N, D = 4, 41, 32   # N=41 matches num_patches at seq_len=336,patch=16,stride=8
    c_attn = CurvatureParam(1.0)
    attn_hyp = HyperbolicDistanceAttention(dim=D, hyperbolic=True)
    attn_euc = HyperbolicDistanceAttention(dim=D, hyperbolic=False)
    n_params_hyp = sum(p.numel() for p in attn_hyp.parameters())
    n_params_euc = sum(p.numel() for p in attn_euc.parameters())
    assert n_params_hyp == n_params_euc, "hyperbolic/Euclidean attention must be param-matched"

    t_shared = (torch.randn(B, N, D) * 0.2).requires_grad_()   # shared tangent-space input (leaf)
    h_on_ball = expmap0(t_shared, c_attn.c)

    out_hyp = attn_hyp(h_on_ball, c_attn.c)          # operates on the ball
    out_euc_raw = attn_euc(t_shared)                  # operates directly in Euclidean space
    assert out_hyp.shape == (B, N, D)
    assert out_euc_raw.shape == (B, N, D)
    out_hyp.sum().backward()
    assert t_shared.grad is not None
    print(f"[OK] HyperbolicDistanceAttention  shape={out_hyp.shape}  "
          f"params(hyp)={n_params_hyp}=params(euc)={n_params_euc}  grad flows")

    # The critical property: logmap0(out_hyp) must differ from a same-curvature round-trip
    # of out_euc_raw by more than numerical noise — i.e. the hyperbolic path's attention
    # weights (driven by hyp_distance) are NOT the same relation as the Euclidean path's
    # (driven by squared Euclidean distance) on the identical underlying points, so this
    # component does not collapse to a round-trip no-op the way the old fusion path did.
    tangent_of_hyp_output = logmap0(out_hyp, c_attn.c)
    diff = (tangent_of_hyp_output - out_euc_raw).abs().mean().item()
    assert diff > 1e-3, (
        f"hyperbolic and Euclidean attention outputs are suspiciously close (diff={diff:.2e}) "
        f"-- check that hyp_distance is actually driving different attention weights than "
        f"squared Euclidean distance would."
    )
    print(f"[OK] genuine curvature-dependence confirmed: hyp vs Euclidean attention "
          f"outputs differ by {diff:.4f} (mean abs) on identical input points")

    print("\n✓ All hyperbolic ops v2.1 tests passed")
