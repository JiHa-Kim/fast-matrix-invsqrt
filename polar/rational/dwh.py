from __future__ import annotations

import math
from typing import Tuple

import torch

from polar.ops import chol_with_jitter_fp64, symmetrize

Tensor = torch.Tensor


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
def dwh_step_matrix_only(
    S: Tensor,
    ell: float,
    jitter_rel: float,
) -> Tuple[Tensor, float]:
    a, b, c = dwh_coeffs_from_ell(ell)
    n = S.shape[0]
    I = torch.eye(n, device=S.device, dtype=torch.float64)
    M = symmetrize(I + float(c) * S)
    L, shift = chol_with_jitter_fp64(M, jitter_rel=jitter_rel)

    invM = torch.cholesky_inverse(L)
    alpha = float(b / c)
    beta = float(a - b / c)
    Q = alpha * I + beta * invM
    return Q, float(shift)


@torch.no_grad()
def dwh_step_chunked(
    X: Tensor,
    S: Tensor,
    ell: float,
    rhs_chunk_rows: int,
    jitter_rel: float,
    out_dtype: torch.dtype,
) -> Tuple[Tensor, float]:
    Q, shift = dwh_step_matrix_only(S, ell, jitter_rel)

    X_next = torch.empty_like(X, dtype=out_dtype)
    for i in range(0, X.shape[0], rhs_chunk_rows):
        end = min(i + rhs_chunk_rows, X.shape[0])
        Xi = X[i:end].to(torch.float64)
        Zi = Xi @ Q
        X_next[i:end] = Zi.to(dtype=out_dtype)

    return X_next, float(shift)
