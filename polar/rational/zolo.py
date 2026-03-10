from __future__ import annotations

import dataclasses
import functools
from typing import Tuple

import torch

from polar.ops import chol_with_jitter_fp64, symmetrize

try:
    import mpmath as mp
except Exception:
    mp = None

Tensor = torch.Tensor


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
def zolo_step_matrix_only(
    S: Tensor,
    coeffs: ZoloCoeffs,
    jitter_rel: float,
) -> Tuple[Tensor, float]:
    n = S.shape[0]
    I = torch.eye(n, device=S.device, dtype=torch.float64)
    max_shift = 0.0
    Q = torch.eye(n, device=S.device, dtype=torch.float64)

    for ce, co in zip(coeffs.c_even, coeffs.c_odd):
        Z = symmetrize(S + float(co) * I)
        L, shift = chol_with_jitter_fp64(Z, jitter_rel=jitter_rel)
        max_shift = max(max_shift, float(shift))
        delta = float(ce - co)
        invZ = torch.cholesky_inverse(L)
        Q = Q + delta * (Q @ invZ)

    Q = float(coeffs.mhat) * Q
    return Q, float(max_shift)

