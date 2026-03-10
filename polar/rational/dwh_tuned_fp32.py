from __future__ import annotations

import math
from typing import Tuple

import torch

from polar.ops import symmetrize
from polar.rational.dwh import dwh_coeffs_from_ell

Tensor = torch.Tensor

# For FP32, a condition number of 1e5 is the limit for precision.
# We cap 'c' to 20,000 to ensure the matrix addition I + c*S 
# doesn't truncate the +I shift significantly.
SAFE_MAX_C_FP32 = 20000.0

def get_tuned_dwh_coeffs_fp32(ell: float) -> Tuple[float, float, float]:
    """
    Computes DWH coefficients but ensures they are numerically safe for FP32.
    Includes a relaxation buffer to prevent divergence.
    """
    # Relaxation: Use a slightly larger ell than the theoretical lower bound.
    # This makes the rational function less aggressive, providing a safety margin.
    ell_relaxed = float(ell) * 1.1
    
    low, high = 1e-10, 1.0
    ell_safe = ell_relaxed
    
    # Check if we even need to back off
    _, _, c_test = dwh_coeffs_from_ell(max(ell, 1e-10))
    if c_test > SAFE_MAX_C_FP32:
        for _ in range(30):
            mid = (low + high) / 2.0
            _, _, c = dwh_coeffs_from_ell(mid)
            if c > SAFE_MAX_C_FP32:
                low = mid
            else:
                high = mid
        ell_safe = high
    
    return dwh_coeffs_from_ell(max(float(ell), ell_safe))

@torch.no_grad()
def dwh_step_tuned_fp32(
    X: Tensor,
    S: Tensor,
    ell: float,
    rhs_chunk_rows: int,
    jitter_rel: float,
    out_dtype: torch.dtype,
) -> Tuple[Tensor, float]:
    """
    Directly updates X using tuned FP32 coefficients and solve_ex.
    Direct update is often more stable than accumulating Q.
    """
    a, b, c = get_tuned_dwh_coeffs_fp32(ell)
    n = S.shape[0]
    device = S.device
    dtype = S.dtype
    
    I = torch.eye(n, device=device, dtype=dtype)
    M = symmetrize(I + float(c) * S)
    
    # Use solve_ex for robustness
    invM, info = torch.linalg.solve_ex(M, I)
    
    shift = 0.0
    if (info != 0).any():
        # Jitter escalation if solve_ex still fails
        scale = float((torch.trace(M).abs() / max(n, 1)).item())
        base = max(float(jitter_rel) * max(scale, 1.0), 1e-7 * scale)
        delta = base
        for _ in range(8):
            Mt = M + delta * I
            invM, info = torch.linalg.solve_ex(Mt, I)
            if (info == 0).all():
                shift = delta
                break
            delta *= 2.0
        else:
            raise RuntimeError("tuned_fp32: solve_ex failed even after jitter")

    alpha = float(b / c)
    beta = float(a - b / c)
    
    # Direct update: X_next = alpha * X + beta * (X @ invM)
    # We do this in chunks to save memory and potentially gain speed.
    X_next = torch.empty_like(X, dtype=out_dtype)
    invM_work = invM.to(dtype=X.dtype)
    for i in range(0, X.shape[0], rhs_chunk_rows):
        end = min(i + rhs_chunk_rows, X.shape[0])
        Xi = X[i:end]
        # Zi = alpha * Xi + beta * (Xi @ invM)
        Zi = torch.addmm(Xi, Xi, invM_work, beta=alpha, alpha=beta)
        X_next[i:end] = Zi.to(dtype=out_dtype)
        
    return X_next, float(shift)

@torch.no_grad()
def dwh_step_matrix_only_tuned_fp32(
    S: Tensor,
    ell: float,
    jitter_rel: float,
) -> Tuple[Tensor, float]:
    """
    Tuned DWH step for FP32, returns Q matrix for O(n^3) updates.
    """
    a, b, c = get_tuned_dwh_coeffs_fp32(ell)
    n = S.shape[0]
    device = S.device
    dtype = S.dtype
    
    I = torch.eye(n, device=device, dtype=dtype)
    M = symmetrize(I + float(c) * S)
    
    # Use solve_ex for robustness in fp32
    invM, info = torch.linalg.solve_ex(M, I)
    
    shift = 0.0
    if (info != 0).any():
        scale = float((torch.trace(M).abs() / max(n, 1)).item())
        base = max(float(jitter_rel) * max(scale, 1.0), 1e-7 * scale)
        delta = base
        for _ in range(8):
            Mt = M + delta * I
            invM, info = torch.linalg.solve_ex(Mt, I)
            if (info == 0).all():
                shift = delta
                break
            delta *= 2.0
        else:
            raise RuntimeError("tuned_fp32_matrix: solve_ex failed")

    alpha = float(b / c)
    beta = float(a - b / c)
    
    Q = alpha * I + beta * invM
    return symmetrize(Q), float(shift)
