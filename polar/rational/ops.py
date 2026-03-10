from __future__ import annotations

import math
from typing import Tuple

import torch

from polar.ops import (
    symmetrize,
    safe_exp,
    acosh_exp,
    chol_with_jitter_fp64,
)

Tensor = torch.Tensor

@torch.no_grad()
def cert_bound_trace_logdet_stable(S: Tensor, jitter_rel: float) -> Tuple[float, float]:
    """
    More robust certificate calculation that falls back to eigvalsh if Cholesky fails.
    Useful for lower precision matrices.
    """
    S_work = symmetrize(S.to(torch.float64))
    n = S_work.shape[0]

    try:
        L, shift = chol_with_jitter_fp64(S_work, jitter_rel=jitter_rel)
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum().item()
    except RuntimeError:
        # Fallback to eigvalsh if Cholesky fails even after jitter
        evals = torch.linalg.eigvalsh(S_work)
        # Ensure all eigenvalues are positive for logdet calculation
        evals = torch.clamp(evals, min=1e-300)
        logdet = torch.log(evals).sum().item()
        shift = 0.0 # We didn't use shift for eigvalsh

    a = max(float((torch.trace(S_work) / n).item()), 1e-300)
    g = safe_exp(logdet / n)
    r = max(a / max(g, 1e-300), 1.0)

    logu = 0.5 * n * math.log(r)
    eta_ub = acosh_exp(logu)
    return float(safe_exp(eta_ub)), float(shift)


@torch.no_grad()
def apply_right_small_chunked_fast(
    X: Tensor, U: Tensor, rhs_chunk_rows: int, out_dtype: torch.dtype
) -> Tensor:
    """
    Lower-precision version of apply_right_small_chunked.
    """
    m, n = X.shape
    X_next = torch.empty((m, n), device=X.device, dtype=out_dtype)
    
    # Use the higher precision of X or U for the intermediate multiplication
    work_dtype = torch.promote_types(X.dtype, U.dtype)
    if work_dtype == torch.float16 or work_dtype == torch.bfloat16:
        # Avoid half precision for the actual matmul if possible for stability
        work_dtype = torch.float32
        
    U_work = U.to(dtype=work_dtype)

    for i in range(0, m, rhs_chunk_rows):
        Xi = X[i : i + rhs_chunk_rows].to(dtype=work_dtype)
        Zi = Xi @ U_work
        X_next[i : i + rhs_chunk_rows] = Zi.to(dtype=out_dtype)

    return X_next
