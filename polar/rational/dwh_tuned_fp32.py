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


@torch.no_grad()
def _factor_spd_with_jitter_fp32(
    M: Tensor,
    I: Tensor,
    jitter_rel: float,
) -> Tuple[Tensor, float]:
    shift = 0.0
    L, info = torch.linalg.cholesky_ex(M)
    if int(info.item()) == 0:
        return L, shift

    scale = float((torch.trace(M).abs() / max(M.shape[0], 1)).item())
    base = max(float(jitter_rel) * max(scale, 1.0), 1e-7 * scale)
    delta = base
    for _ in range(8):
        Mt = M + delta * I
        L, info = torch.linalg.cholesky_ex(Mt)
        if int(info.item()) == 0:
            return L, float(delta)
        delta *= 2.0

    raise RuntimeError("tuned_fp32: cholesky_ex failed even after jitter")


@torch.no_grad()
def _inverse_via_lu_with_jitter_fp32(
    M: Tensor,
    I: Tensor,
    jitter_rel: float,
) -> Tuple[Tensor, float]:
    invM, info = torch.linalg.solve_ex(M, I)
    shift = 0.0
    if (info != 0).any():
        scale = float((torch.trace(M).abs() / max(M.shape[0], 1)).item())
        base = max(float(jitter_rel) * max(scale, 1.0), 1e-7 * scale)
        delta = base
        for _ in range(8):
            Mt = M + delta * I
            invM, info = torch.linalg.solve_ex(Mt, I)
            if (info == 0).all():
                return invM, float(delta)
            delta *= 2.0
        raise RuntimeError("tuned_fp32: solve_ex failed even after jitter")
    return invM, shift

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
    
    alpha = float(b / c)
    beta = float(a - b / c)

    try:
        L, shift = _factor_spd_with_jitter_fp32(M, I, jitter_rel)
        Xt = X.mT.contiguous()
        Y = torch.linalg.solve_triangular(L, Xt, upper=False)
        Z = torch.linalg.solve_triangular(L.mT, Y, upper=True)
        X_next = (alpha * X + beta * Z.mT).to(dtype=out_dtype)
        return X_next, float(shift)
    except RuntimeError:
        invM, shift = _inverse_via_lu_with_jitter_fp32(M, I, jitter_rel)
        invM_work = invM.to(dtype=X.dtype)
        X_next = torch.addmm(X, X, invM_work, beta=alpha, alpha=beta).to(dtype=out_dtype)
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
    
    alpha = float(b / c)
    beta = float(a - b / c)

    try:
        L, shift = _factor_spd_with_jitter_fp32(M, I, jitter_rel)
        invM = torch.cholesky_inverse(L)
    except RuntimeError:
        invM, shift = _inverse_via_lu_with_jitter_fp32(M, I, jitter_rel)

    Q = alpha * I + beta * invM
    return symmetrize(Q), float(shift)
