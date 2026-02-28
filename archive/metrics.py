from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .utils import _validate_p_val, _check_square


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

    if len(eye_mat.shape) >= 2 and eye_mat.shape[-2:] == (n, n):
        try:
            torch.broadcast_shapes(eye_mat.shape, Af.shape)
            return eye_mat
        except RuntimeError:
            pass

    raise ValueError(
        f"eye_mat has incompatible shape {tuple(eye_mat.shape)} for A shape "
        f"{tuple(Af.shape)}; expected broadcastable shape ending in {(n, n)}"
    )


def _agg_max(x: torch.Tensor) -> float:
    # Conservative aggregate over batch: max
    return float(x.detach().float().reshape(-1).max().item())


def _agg_nan_if_empty(v: torch.Tensor) -> float:
    if v.numel() == 0:
        return float("nan")
    return _agg_max(v)


@torch.no_grad()
def exact_inverse_proot(
    A: torch.Tensor, p_val: int = 2, eps: float = 1e-20, assume_spd: bool = True
) -> torch.Tensor:
    """Exact inverse p-th root.

    For assume_spd=True (default), uses eigendecomposition with eigenvalue clamping.
    For assume_spd=False, only p_val=1 is supported (direct inverse).
    """
    _validate_p_val(p_val)
    _check_square(A)
    assume_spd = bool(assume_spd)
    if not assume_spd:
        if p_val != 1:
            raise ValueError(
                "exact_inverse_proot with assume_spd=False currently supports only p_val=1"
            )
        return torch.linalg.inv(A.double()).to(dtype=A.dtype)
    # Supports batching: A shape (..., n, n)
    eigvals, V = torch.linalg.eigh(A.double())
    eigvals = eigvals.clamp_min(eps)
    D = torch.diag_embed(eigvals ** (-1.0 / p_val))
    X = V @ D @ V.mH
    return X.to(dtype=A.dtype)


@torch.no_grad()
def exact_inverse_sqrt(A: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return exact_inverse_proot(A, p_val=2, eps=eps, assume_spd=True)


@torch.no_grad()
def iroot_relative_error(
    Xhat: torch.Tensor, A: torch.Tensor, p_val: int = 2, assume_spd: bool = True
) -> torch.Tensor:
    _validate_p_val(p_val)
    _check_square(Xhat)
    _check_square(A)
    if Xhat.shape != A.shape:
        raise ValueError(
            f"Xhat and A must have compatible shapes, got {Xhat.shape} and {A.shape}"
        )
    # Returns per-batch relative Fro error (shape: batch)
    Xref = exact_inverse_proot(A, p_val=p_val, assume_spd=assume_spd)
    denom = torch.linalg.matrix_norm(Xref, ord="fro").clamp_min(1e-12)
    num = torch.linalg.matrix_norm(Xhat - Xref, ord="fro")
    return num / denom


@torch.no_grad()
def isqrt_relative_error(Xhat: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return iroot_relative_error(Xhat, A, p_val=2, assume_spd=True)


@torch.no_grad()
def compute_quality_stats(
    X: torch.Tensor,
    A: torch.Tensor,
    power_iters: int,
    mv_samples: int,
    hard_probe_iters: int = 0,
    eye_mat: Optional[torch.Tensor] = None,
    p_val: int = 2,
    assume_spd: bool = True,
) -> QualityStats:
    """
    Computes conservative (worst-case over batch) quality stats.

    Assumptions:
      - assume_spd=True: A is SPD-ish and X is an SPD inverse-root iterate.
      - assume_spd=False: no symmetry assumptions; diagnostics use general residuals.

    Note: For assume_spd=True and p_val=2, residual is R = I - X A X.
    Otherwise residual is the general form R = I - X^p A.
    """
    if A.is_complex() or X.is_complex():
        raise ValueError("compute_quality_stats does not support complex tensors.")

    _validate_p_val(p_val)
    assume_spd = bool(assume_spd)
    _check_square(X)
    _check_square(A)
    if X.shape != A.shape:
        raise ValueError(f"X.shape {X.shape} must perfectly match A.shape {A.shape}")
    n = A.shape[-1]
    batch = A.shape[:-2]

    Xf = X.float()
    Af = A.float()
    eye = _ensure_eye(Af, eye_mat)

    # Core residual.
    # For SPD p=2 this uses the standard isqrt diagnostic R = I - X A X.
    # Otherwise use the general p-root diagnostic R = I - X^p A.
    if assume_spd and p_val == 2:
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

    # Estimate ||R||_2 via power iteration on R^T R.
    # For sufficient iterations, this is a standard spectral norm estimator.
    if power_iters > 0:
        it = int(power_iters)
        v = torch.randn(*batch, n, 1, device=Af.device, dtype=Af.dtype)
        v = v / _bnorm(v)

        Rt = R.mT
        for _ in range(it):
            v = Rt @ (R @ v)
            v = v / _bnorm(v)

        # Estimate ||R|| via ||R v|| / ||v|| (which converges to sqrt(eigmax(R^T R)))
        Rv = R @ v
        spec_per = torch.linalg.vector_norm(
            Rv, dim=(-2, -1)
        ) / torch.linalg.vector_norm(v, dim=(-2, -1)).clamp_min(1e-12)
        residual_spec = _agg_max(spec_per)
    else:
        residual_spec = float("nan")

    # "Hard direction" probe is meaningful under SPD assumptions.
    if hard_probe_iters > 0 and assume_spd:
        it = int(hard_probe_iters)
        Ad = A.double()
        Rd = R.double()
        u = torch.randn(*batch, n, 1, device=Ad.device, dtype=Ad.dtype)
        u = u / _bnorm(u)

        for _ in range(it):
            u = torch.linalg.solve(Ad, u)  # batched solve
            u = u / _bnorm(u)

        Ru = Rd @ u
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
