from __future__ import annotations

import dataclasses
import functools
import math
from typing import Tuple

import torch

from polar.ops import chol_with_jitter_fp64, symmetrize

try:
    import mpmath as mp
except Exception:
    mp = None

Tensor = torch.Tensor


def bf16_target(mode: str) -> float:
    u = 2.0**-8
    if mode == "aggressive":
        return float(1.0 + u)
    if mode == "robust":
        return float((1.0 + u) / (1.0 - u))
    raise ValueError(mode)


def dwh_coeffs_from_ell(ell: float) -> Tuple[float, float, float]:
    ell = float(min(max(ell, 1e-300), 1.0))
    ell2 = ell * ell
    d = (4.0 * (1.0 - ell2) / (ell2 * ell2)) ** (1.0 / 3.0)
    a = math.sqrt(1.0 + d) + 0.5 * math.sqrt(
        8.0 - 4.0 * d + 8.0 * (2.0 - ell2) / (ell2 * math.sqrt(1.0 + d))
    )
    b = 0.25 * (a - 1.0) * (a - 1.0)
    c = a + b - 1.0
    return float(a), float(b), float(c)


def dwh_ell_next(ell: float) -> float:
    a, b, c = dwh_coeffs_from_ell(ell)
    return float(ell * (a + b * ell * ell) / (1.0 + c * ell * ell))


@torch.no_grad()
def dwh_step_chunked(
    X: Tensor,
    S: Tensor,
    ell: float,
    rhs_chunk_rows: int,
    jitter_rel: float,
    out_dtype: torch.dtype,
) -> Tuple[Tensor, float]:
    # Mathematical Form: X_{k+1} = X_k (a I + b S) (I + c S)^{-1}
    # This is equivalent to X_k [ (b/c) I + (a - b/c) (I + c S)^{-1} ]
    a, b, c = dwh_coeffs_from_ell(ell)
    n = S.shape[0]
    I = torch.eye(n, device=S.device, dtype=torch.float64)
    M = symmetrize(I + float(c) * S)
    L, shift = chol_with_jitter_fp64(M, jitter_rel=jitter_rel)
    
    # We precompute the n x n update matrix Q.
    # GEMM (X @ Q) is much faster than TRSM (cholesky_solve).
    invM = torch.cholesky_inverse(L)
    alpha = float(b / c)
    beta = float(a - b / c)
    Q = alpha * I + beta * invM

    X_next = torch.empty_like(X, dtype=out_dtype)
    for i in range(0, X.shape[0], rhs_chunk_rows):
        end = min(i + rhs_chunk_rows, X.shape[0])
        Xi = X[i:end].to(torch.float64)
        Zi = Xi @ Q
        X_next[i:end] = Zi.to(dtype=out_dtype)

    return X_next, float(shift)


@dataclasses.dataclass(frozen=True)
class ZoloCoeffs:
    r: int
    ell: float
    c_odd: Tuple[float, ...]
    c_even: Tuple[float, ...]
    mhat: float


@functools.lru_cache(maxsize=512)
def _zolo_coeffs_cached(r: int, ell_key: float, dps: int) -> ZoloCoeffs:
    if mp is None:
        raise RuntimeError("mpmath is required for Zolo coefficients")

    mp.mp.dps = int(dps)
    ell = mp.mpf(ell_key)
    kp = mp.sqrt(1 - ell * ell)
    m = kp * kp
    Kp = mp.ellipk(m)

    c_all = []
    for i in range(1, 2 * r + 1):
        u = mp.mpf(i) * Kp / mp.mpf(2 * r + 1)
        sn = mp.ellipfun("sn", u, m)
        cn = mp.ellipfun("cn", u, m)
        ci = ell * ell * (sn / cn) ** 2
        c_all.append(ci)

    c_odd = [c_all[2 * j] for j in range(r)]
    c_even = [c_all[2 * j + 1] for j in range(r)]

    mhat = mp.mpf(1)
    for j in range(r):
        mhat *= (1 + c_odd[j]) / (1 + c_even[j])

    return ZoloCoeffs(
        r=int(r),
        ell=float(ell_key),
        c_odd=tuple(float(v) for v in c_odd),
        c_even=tuple(float(v) for v in c_even),
        mhat=float(mhat),
    )


def zolo_coeffs_from_ell(r: int, ell: float, dps: int = 100) -> ZoloCoeffs:
    ell = float(min(max(ell, 1e-18), 1.0 - 1e-18))
    ell_key = float(f"{ell:.18e}")
    return _zolo_coeffs_cached(int(r), ell_key, int(dps))


def zolo_scalar_value(sigma: float, coeffs: ZoloCoeffs) -> float:
    x = float(sigma)
    x2 = x * x
    val = float(coeffs.mhat) * x
    for ce, co in zip(coeffs.c_even, coeffs.c_odd):
        val *= (x2 + ce) / (x2 + co)
    return float(val)


def zolo_ell_next(ell: float, coeffs: ZoloCoeffs) -> float:
    return float(max(min(zolo_scalar_value(ell, coeffs), 1.0), 1e-300))


@torch.no_grad()
def zolo_product_step_chunked(
    X: Tensor,
    S: Tensor,
    coeffs: ZoloCoeffs,
    rhs_chunk_rows: int,
    jitter_rel: float,
    out_dtype: torch.dtype,
) -> Tuple[Tensor, float]:
    # Mathematical Form: X_{k+1} = mhat * X_k * product( (S + co I)^{-1} (S + ce I) )
    # This is equivalent to X_k * [ mhat * product( I + (ce - co) (S + co I)^{-1} ) ]
    # The term in brackets is a pure n x n matrix. By computing it first, 
    # we reduce the number of passes over the large matrix X from r to 1.
    n = S.shape[0]
    I = torch.eye(n, device=S.device, dtype=torch.float64)
    max_shift = 0.0

    # Initialize n x n update matrix Q
    Q = torch.eye(n, device=S.device, dtype=torch.float64)

    for ce, co in zip(coeffs.c_even, coeffs.c_odd):
        Z = symmetrize(S + float(co) * I)
        L, shift = chol_with_jitter_fp64(Z, jitter_rel=jitter_rel)
        max_shift = max(max_shift, float(shift))
        delta = float(ce - co)

        # Update Q: Q_next = Q @ (I + delta * Z^{-1}) = Q + delta * (Q @ Z^{-1})
        invZ = torch.cholesky_inverse(L)
        Q = Q + delta * (Q @ invZ)

    Q = float(coeffs.mhat) * Q

    # One pass over X using GEMM
    X_next = torch.empty_like(X, dtype=out_dtype)
    for i in range(0, X.shape[0], rhs_chunk_rows):
        end = min(i + rhs_chunk_rows, X.shape[0])
        Xi = X[i:end].to(torch.float64)
        Zi = Xi @ Q
        X_next[i:end] = Zi.to(dtype=out_dtype)

    return X_next, float(max_shift)
