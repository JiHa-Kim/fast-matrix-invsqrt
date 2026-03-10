from __future__ import annotations

import math
from typing import Tuple

import torch

from polar.ops import symmetrize
from polar.rational.dwh import dwh_coeffs_from_ell

Tensor = torch.Tensor

# 2000.0 is an extremely safe sweet spot for FP32.
# It preserves ~4 digits of accuracy during the worst-case factorization
# while retaining large geometric step sizes of DWH.
MAX_C_FP32 = 5000.0


def dwh_coeffs_from_ell_capped(ell: float, max_c: float = MAX_C_FP32) -> Tuple[float, float, float]:
    """
    Finds the optimal DWH coefficients but statically caps the aggressiveness
    so that the condition number `c` never exceeds the hardware precision bounds.
    """
    ell_cap = 1.0
    low, high = 1e-15, 1.0
    for _ in range(60):
        mid = (low + high) / 2.0
        _, _, c = dwh_coeffs_from_ell(mid)
        if c > max_c:
            low = mid
        else:
            high = mid
            ell_cap = mid
            
    ell_safe = float(max(float(ell), ell_cap))
    return dwh_coeffs_from_ell(ell_safe)


def dwh_ell_next_capped(ell: float, max_c: float = MAX_C_FP32) -> float:
    a, b, c = dwh_coeffs_from_ell_capped(ell, max_c)
    return float(ell * (a + b * ell * ell) / (1.0 + c * ell * ell))


@torch.no_grad()
def chol_with_jitter_native(A: Tensor, jitter_rel: float, max_tries: int = 16) -> Tuple[Tensor, float]:
    """
    Executes a native precision Cholesky decomposition.
    Because we analytically bounded the condition number, we can execute this in pure fp32!
    """
    A = symmetrize(A)
    n = A.shape[0]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    scale = float((torch.trace(A).abs() / max(n, 1)).item())
    
    eps = 1e-7 if A.dtype == torch.float32 else 1e-15
    if A.dtype in (torch.bfloat16, torch.float16):
        eps = 1e-3
        
    base = max(float(jitter_rel) * max(scale, 1.0), eps * scale, eps)
    
    delta = 0.0
    for _ in range(max_tries):
        At = A if delta == 0.0 else (A + delta * I)
        L, info = torch.linalg.cholesky_ex(At)
        if int(info.item()) == 0:
            return L, float(delta)
        delta = base if delta == 0.0 else 2.0 * delta
        
    print(f"DEBUG: Native Cholesky failed! A.isnan().any()={A.isnan().any().item()}, A.isinf().any()={A.isinf().any().item()}")
    evals = torch.linalg.eigvalsh(A.to(torch.float64))
    print(f"DEBUG: A eigvals min={evals[0].item():g} max={evals[-1].item():g}")
    raise RuntimeError(f"Native Cholesky failed after {max_tries} tries. dtype={A.dtype}, scale={scale:g}, last delta={delta:g}")


@torch.no_grad()
def dwh_step_matrix_only_opt(
    S: Tensor,
    ell: float,
    jitter_rel: float,
    max_c: float = MAX_C_FP32,
) -> Tuple[Tensor, float]:
    a, b, c = dwh_coeffs_from_ell_capped(ell, max_c)
    n = S.shape[0]
    dtype = S.dtype
    device = S.device
    
    I = torch.eye(n, device=device, dtype=dtype)
    M = symmetrize(I + float(c) * S)
    
    # Executed completely in native fp32 thanks to the condition cap!
    L, shift = chol_with_jitter_native(M, jitter_rel)
    invM = torch.cholesky_inverse(L)
    
    alpha = float(b / c)
    beta = float(a - b / c)
    Q = alpha * I + beta * symmetrize(invM)
    return symmetrize(Q), float(shift)


@torch.no_grad()
def dwh_step_chunked_opt(
    X: Tensor,
    S: Tensor,
    ell: float,
    rhs_chunk_rows: int,
    jitter_rel: float,
    out_dtype: torch.dtype,
    max_c: float = MAX_C_FP32,
) -> Tuple[Tensor, float]:
    Q, shift = dwh_step_matrix_only_opt(S, ell, jitter_rel, max_c)
    X_next = torch.empty_like(X, dtype=out_dtype)
    for i in range(0, X.shape[0], rhs_chunk_rows):
        end = min(i + rhs_chunk_rows, X.shape[0])
        Xi = X[i:end].to(dtype=Q.dtype)
        Zi = Xi @ Q
        X_next[i:end] = Zi.to(dtype=out_dtype)
    return X_next, float(shift)
