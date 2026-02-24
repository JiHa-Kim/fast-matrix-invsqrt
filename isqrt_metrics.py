from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class QualityStats:
    residual_fro: float
    residual_spec: float
    sym_x: float
    sym_w: float
    mv_err: float
    hard_dir_err: float


def _bnorm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Frobenius/vector norm over the last two dims (works for (..., n, 1) and (..., n, k))
    return torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp_min(eps)


def _ensure_eye(Af: torch.Tensor, eye_mat: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Returns an eye tensor that is broadcast-compatible with Af (shape (..., n, n)).
    Accepts:
      - None
      - shape (n, n)
      - shape (..., n, n) matching Af.shape
    """
    n = Af.shape[-1]
    if eye_mat is None:
        return torch.eye(n, device=Af.device, dtype=Af.dtype)

    if eye_mat.device != Af.device or eye_mat.dtype != Af.dtype:
        eye_mat = eye_mat.to(device=Af.device, dtype=Af.dtype)

    if eye_mat.shape == (n, n):
        return eye_mat
    if eye_mat.shape == Af.shape:
        return eye_mat

    # Fallback: ignore incompatible shapes
    return torch.eye(n, device=Af.device, dtype=Af.dtype)


def _agg_max(x: torch.Tensor) -> float:
    # Conservative aggregate over batch: max
    return float(x.detach().float().reshape(-1).max().item())


def _agg_nan_if_empty(v: torch.Tensor) -> float:
    if v.numel() == 0:
        return float("nan")
    return _agg_max(v)


@torch.no_grad()
def exact_inverse_proot(
    A: torch.Tensor, p_val: int = 2, eps: float = 1e-20
) -> torch.Tensor:
    # Supports batching: A shape (..., n, n)
    eigvals, V = torch.linalg.eigh(A.double())
    eigvals = eigvals.clamp_min(eps)
    D = torch.diag_embed(eigvals ** (-1.0 / p_val))
    X = V @ D @ V.mT
    return X.to(dtype=A.dtype)


@torch.no_grad()
def exact_inverse_sqrt(A: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return exact_inverse_proot(A, p_val=2, eps=eps)


@torch.no_grad()
def iroot_relative_error(
    Xhat: torch.Tensor, A: torch.Tensor, p_val: int = 2
) -> torch.Tensor:
    # Returns per-batch relative Fro error (shape: batch)
    Xref = exact_inverse_proot(A, p_val=p_val)
    denom = torch.linalg.matrix_norm(Xref, ord="fro").clamp_min(1e-12)
    num = torch.linalg.matrix_norm(Xhat - Xref, ord="fro")
    return num / denom


@torch.no_grad()
def isqrt_relative_error(Xhat: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return iroot_relative_error(Xhat, A, p_val=2)


@torch.no_grad()
def compute_quality_stats(
    X: torch.Tensor,
    A: torch.Tensor,
    power_iters: int,
    mv_samples: int,
    hard_probe_iters: int = 0,
    eye_mat: Optional[torch.Tensor] = None,
    p_val: int = 2,
) -> QualityStats:
    """
    Computes conservative (worst-case over batch) quality stats.

    Assumptions:
      - A is SPD-ish (at least for "exact" reference and solve-based probes).
      - X is an approximate A^{-1/2}.
    """
    n = A.shape[-1]
    batch = A.shape[:-2]

    Xf = X.float()
    Af = A.float()
    eye = _ensure_eye(Af, eye_mat)

    # Core residual: R = I - X^p A
    if p_val == 2:
        W = Xf @ Af @ Xf
    elif p_val == 3:
        W = Xf @ Xf @ Xf @ Af
    elif p_val == 4:
        X2 = Xf @ Xf
        W = X2 @ X2 @ Af
    else:
        W = Xf
        for _ in range(p_val - 1):
            W = W @ Xf
        W = W @ Af

    R = eye - W  # broadcast works if eye is (n,n)

    # Residual Fro per matrix, normalized by sqrt(n)
    # matrix_norm returns shape: batch
    residual_fro_per = torch.linalg.matrix_norm(R, ord="fro") / math.sqrt(float(n))
    residual_fro = _agg_max(residual_fro_per)

    # Symmetry measures per matrix (relative Fro)
    x_denom = torch.linalg.matrix_norm(Xf, ord="fro").clamp_min(1e-12)
    w_denom = torch.linalg.matrix_norm(W, ord="fro").clamp_min(1e-12)
    sym_x_per = torch.linalg.matrix_norm(Xf - Xf.mT, ord="fro") / x_denom
    sym_w_per = torch.linalg.matrix_norm(W - W.mT, ord="fro") / w_denom
    sym_x = _agg_max(sym_x_per)
    sym_w = _agg_max(sym_w_per)

    # Random MV probe: median over samples per matrix, then max over batch
    if mv_samples > 0:
        k = int(mv_samples)
        V = torch.randn(*batch, n, k, device=Af.device, dtype=Af.dtype)
        RV = R @ V  # (..., n, k)

        mv_num = torch.linalg.vector_norm(RV, dim=-2)  # (..., k)
        mv_den = torch.linalg.vector_norm(V, dim=-2).clamp_min(1e-12)  # (..., k)
        ratios = mv_num / mv_den  # (..., k)

        mv_med_per = ratios.median(dim=-1).values  # batch
        mv_err = _agg_max(mv_med_per)
    else:
        mv_err = float("nan")

    # Spectral-ish residual norm estimate via power iteration on R
    # Returns worst-case over batch of ||R||_2 estimate.
    if power_iters > 0:
        it = int(power_iters)
        v = torch.randn(*batch, n, 1, device=Af.device, dtype=Af.dtype)
        v = v / _bnorm(v)

        for _ in range(it):
            v = R @ v
            v = v / _bnorm(v)

        # Estimate ||R|| via ||R v|| / ||v||
        Rv = R @ v
        spec_per = torch.linalg.vector_norm(
            Rv, dim=(-2, -1)
        ) / torch.linalg.vector_norm(v, dim=(-2, -1)).clamp_min(1e-12)
        residual_spec = _agg_max(spec_per)
    else:
        residual_spec = float("nan")

    # "Hard direction" probe: iterate u <- A^{-1} u (push toward smallest-eig direction),
    # then measure ||R u|| / ||u||. Use per-batch norms, then max over batch.
    if hard_probe_iters > 0:
        it = int(hard_probe_iters)
        u = torch.randn(*batch, n, 1, device=Af.device, dtype=Af.dtype)
        u = u / _bnorm(u)

        for _ in range(it):
            u = torch.linalg.solve(Af, u)  # batched solve
            u = u / _bnorm(u)

        Ru = R @ u
        hard_per = torch.linalg.vector_norm(
            Ru, dim=(-2, -1)
        ) / torch.linalg.vector_norm(u, dim=(-2, -1)).clamp_min(1e-12)
        hard_dir_err = _agg_max(hard_per)
    else:
        hard_dir_err = float("nan")

    return QualityStats(
        residual_fro=float(residual_fro),
        residual_spec=float(residual_spec),
        sym_x=float(sym_x),
        sym_w=float(sym_w),
        mv_err=float(mv_err),
        hard_dir_err=float(hard_dir_err),
    )
