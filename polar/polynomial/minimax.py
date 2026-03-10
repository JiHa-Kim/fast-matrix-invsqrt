from __future__ import annotations

import dataclasses
import functools
from typing import Tuple

import numpy as np
import torch
from numpy.polynomial import Chebyshev, Polynomial

try:
    import mpmath as mp
except Exception:
    mp = None

from polar.ops import symmetrize

Tensor = torch.Tensor


@dataclasses.dataclass(frozen=True)
class PolyInvSqrtCoeffs:
    degree: int
    ell: float
    interval_lo: float
    interval_hi: float
    coeffs: Tuple[float, ...]
    max_rel_err: float
    pred_sigma_min: float
    pred_sigma_max: float


def _cheb_basis_matrix(xs: np.ndarray, degree: int) -> np.ndarray:
    basis = np.empty((xs.shape[0], degree + 1), dtype=np.float64)
    basis[:, 0] = 1.0
    if degree == 0:
        return basis
    basis[:, 1] = xs
    for k in range(2, degree + 1):
        basis[:, k] = 2.0 * xs * basis[:, k - 1] - basis[:, k - 2]
    return basis


def _scalar_interval_bounds(coeffs: np.ndarray, ell: float) -> tuple[float, float, float]:
    sigmas = np.linspace(float(max(ell, 1e-6)), 1.0, 4097, dtype=np.float64)
    xs = sigmas * sigmas
    lo = float(ell * ell)
    hi = 1.0
    mid = 0.5 * (lo + hi)
    radius = 0.5 * (hi - lo)
    ts = (xs - mid) / radius
    T = _cheb_basis_matrix(ts, coeffs.shape[0] - 1)
    inv_sqrt = T @ coeffs
    sigma_out = sigmas * inv_sqrt
    rel = np.abs(xs * inv_sqrt * inv_sqrt - 1.0)
    return float(np.min(sigma_out)), float(np.max(sigma_out)), float(np.max(rel))


@functools.lru_cache(maxsize=512)
def _poly_coeffs_cached(degree: int, ell_key: float, dps: int) -> PolyInvSqrtCoeffs:
    if mp is None:
        raise RuntimeError("mpmath is required for polynomial coefficients")

    degree = int(degree)
    ell = float(min(max(ell_key, 1e-6), 1.0))
    lo = float(ell * ell)
    hi = 1.0
    mp.mp.dps = int(dps)

    power_hi_to_lo, _err = mp.chebyfit(lambda x: x ** (-mp.mpf("0.5")), [lo, hi], degree + 1, error=True)
    power_lo_to_hi = np.array([float(v) for v in reversed(power_hi_to_lo)], dtype=np.float64)
    cheb = Polynomial(power_lo_to_hi).convert(kind=Chebyshev, domain=[lo, hi], window=[-1.0, 1.0])
    coeffs = np.array(cheb.coef, dtype=np.float64)

    pred_sigma_min, pred_sigma_max, max_rel_err = _scalar_interval_bounds(coeffs, ell)
    return PolyInvSqrtCoeffs(
        degree=degree,
        ell=ell,
        interval_lo=lo,
        interval_hi=hi,
        coeffs=tuple(float(v) for v in coeffs),
        max_rel_err=float(max_rel_err),
        pred_sigma_min=float(pred_sigma_min),
        pred_sigma_max=float(pred_sigma_max),
    )


def poly_inv_sqrt_coeffs_from_ell(
    degree: int,
    ell: float,
    dps: int = 100,
) -> PolyInvSqrtCoeffs:
    ell_key = float(f"{float(ell):.12e}")
    return _poly_coeffs_cached(int(degree), ell_key, int(dps))


@torch.no_grad()
def chebyshev_clenshaw_matrix(
    A: Tensor,
    coeffs: Tuple[float, ...],
    interval_lo: float,
    interval_hi: float,
    out_dtype: torch.dtype,
) -> Tensor:
    n = A.shape[0]
    I = torch.eye(n, device=A.device, dtype=out_dtype)
    lo = float(interval_lo)
    hi = float(interval_hi)
    mid = 0.5 * (lo + hi)
    radius = 0.5 * (hi - lo)
    if radius <= 0.0:
        raise ValueError("interval must have positive width")

    work = symmetrize(A.to(dtype=out_dtype))
    T = symmetrize((work - mid * I) / radius)
    zeros = torch.zeros_like(T)
    b_kplus1 = zeros
    b_kplus2 = zeros

    for ck in reversed(coeffs[1:]):
        b_k = symmetrize(2.0 * (T @ b_kplus1) - b_kplus2 + float(ck) * I)
        b_kplus2 = b_kplus1
        b_kplus1 = b_k

    return symmetrize(T @ b_kplus1 - b_kplus2 + float(coeffs[0]) * I)


@torch.no_grad()
def poly_step_matrix_only(
    S: Tensor,
    coeffs: PolyInvSqrtCoeffs,
    matmul_dtype: torch.dtype,
) -> tuple[Tensor, float]:
    if (not np.isfinite(coeffs.max_rel_err)) or coeffs.max_rel_err > 0.25:
        raise RuntimeError(
            f"polynomial inverse-sqrt fit is unstable for ell={coeffs.ell:.3e}, degree={coeffs.degree}"
        )
    Q = chebyshev_clenshaw_matrix(
        S,
        coeffs.coeffs,
        interval_lo=coeffs.interval_lo,
        interval_hi=coeffs.interval_hi,
        out_dtype=matmul_dtype,
    )
    if not torch.isfinite(Q).all():
        raise RuntimeError("non-finite polynomial Clenshaw evaluation")
    return Q, 0.0


@torch.no_grad()
def newton_schulz_inv_sqrt_matrix_only(
    S: Tensor,
    steps: int,
    matmul_dtype: torch.dtype,
) -> tuple[Tensor, float]:
    n = S.shape[0]
    I = torch.eye(n, device=S.device, dtype=matmul_dtype)
    X = I.clone()
    A = symmetrize(S.to(dtype=matmul_dtype))
    for _ in range(int(steps)):
        X2 = symmetrize(X @ X)
        X = symmetrize(0.5 * (X @ (3.0 * I - A @ X2)))
        if not torch.isfinite(X).all():
            raise RuntimeError("non-finite Newton-Schulz inverse-sqrt iterate")
    return X, 0.0
