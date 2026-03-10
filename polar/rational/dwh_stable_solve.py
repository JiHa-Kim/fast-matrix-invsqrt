from __future__ import annotations

import math
from typing import Tuple

import torch

from polar.ops import symmetrize
from polar.rational.dwh import dwh_coeffs_from_ell

Tensor = torch.Tensor

@torch.no_grad()
def dwh_step_matrix_only_stable_solve(
    S: Tensor,
    ell: float,
    jitter_rel: float,
) -> Tuple[Tensor, float]:
    """
    DWH step using torch.linalg.solve_ex for stability in fp32.
    """
    a, b, c = dwh_coeffs_from_ell(ell)
    n = S.shape[0]
    dtype = S.dtype
    device = S.device
    
    # We want to use the input dtype (e.g. fp32) if possible
    I = torch.eye(n, device=device, dtype=dtype)
    M = symmetrize(I + float(c) * S)
    
    # Q = alpha * I + beta * M^{-1}
    # We can compute M^{-1} using solve_ex
    invM, info = torch.linalg.solve_ex(M, I)
    
    shift = 0.0
    if (info != 0).any():
        scale = float((torch.trace(M).abs() / max(n, 1)).item())
        base = max(float(jitter_rel) * max(scale, 1.0), 1e-30)
        delta = base
        for _ in range(8):
            Mt = M + delta * I
            invM, info = torch.linalg.solve_ex(Mt, I)
            if (info == 0).all():
                shift = delta
                break
            delta *= 2.0
        else:
            raise RuntimeError("solve_ex failed even after jitter escalation")

    alpha = float(b / c)
    beta = float(a - b / c)
    Q = alpha * I + beta * invM
    return Q, float(shift)


@torch.no_grad()
def dwh_step_chunked_stable_solve(
    X: Tensor,
    S: Tensor,
    ell: float,
    rhs_chunk_rows: int,
    jitter_rel: float,
    out_dtype: torch.dtype,
) -> Tuple[Tensor, float]:
    Q, shift = dwh_step_matrix_only_stable_solve(S, ell, jitter_rel)

    X_next = torch.empty_like(X, dtype=out_dtype)
    for i in range(0, X.shape[0], rhs_chunk_rows):
        end = min(i + rhs_chunk_rows, X.shape[0])
        Xi = X[i:end].to(dtype=Q.dtype)
        Zi = Xi @ Q
        X_next[i:end] = Zi.to(dtype=out_dtype)

    return X_next, float(shift)
