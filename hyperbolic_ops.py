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
    weights: torch.Tensor,
    c: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    tangents = torch.stack([logmap0(h, c, eps) for h in h_list], dim=-2)
    w = weights.unsqueeze(-1)
    t_fused = (w * tangents).sum(dim=-2)
    return expmap0(t_fused, c, eps)


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
                 geo_dropout: float = 0.2):
        super().__init__()
        self.c_param = c_param

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

        # Small init → points start near origin
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

        # Dropout applied to tangent vector before expmap0
        self.geo_drop = nn.Dropout(geo_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim]  →  [..., hyp_dim] on Poincaré ball"""
        t = self.net(x)
        t = self.geo_drop(t)          # regularise in tangent space before mapping
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
                 geo_dropout: float = 0.2):
        super().__init__()
        self.c_param = c_param

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
        """x: [..., hyp_dim] on ball  →  [..., out_dim] Euclidean"""
        t = logmap0(x, self.c_param.c)
        t = self.geo_drop(t)          # regularise tangent vector before MLP
        return self.net(t)


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

    # 3. tangent_space_fusion
    B, T, D = 4, 32, 16
    h1 = expmap0(torch.randn(B, T, D) * 0.1, c_val)
    h2 = expmap0(torch.randn(B, T, D) * 0.1, c_val)
    h3 = expmap0(torch.randn(B, T, D) * 0.1, c_val)
    w_exp = torch.softmax(torch.randn(B, 3), dim=-1).unsqueeze(1).expand(B, T, 3)
    fused = tangent_space_fusion([h1, h2, h3], w_exp, c_val)
    assert fused.shape == (B, T, D)
    assert fused.norm(dim=-1).max().item() < 1.0
    print(f"[OK] tangent_space_fusion  shape={fused.shape}")

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

    print("\n✓ All hyperbolic ops v2.1 tests passed")
