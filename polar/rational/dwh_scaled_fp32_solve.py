from __future__ import annotations

import math
from typing import Tuple

import torch

from polar.ops import symmetrize
from polar.rational.dwh import dwh_coeffs_from_ell

Tensor = torch.Tensor

@torch.no_grad()
def dwh_step_scaled_fp32_solve(
    S_fp64: Tensor,
    ell: float,
    jitter_rel: float,
) -> Tuple[Tensor, float]:
    """
    Computes DWH step Q by formulating the scaled matrix in FP64,
    casting to FP32, and solving in FP32 for massive speedup.
    """
    a, b, c = dwh_coeffs_from_ell(ell)
    n = S_fp64.shape[0]
    
    I_fp64 = torch.eye(n, device=S_fp64.device, dtype=torch.float64)
    
    # Scaling trick: M_scaled = S + (1/c) * I
    # We form it in FP64 to preserve the small shift against S's small eigenvalues
    M_scaled_fp64 = symmetrize(S_fp64 + (1.0 / c) * I_fp64)
    
    # Cast to FP32 for the extremely fast solve
    M_fp32 = M_scaled_fp64.to(torch.float32)
    I_fp32 = torch.eye(n, device=S_fp64.device, dtype=torch.float32)
    
    invM_fp32, info = torch.linalg.solve_ex(M_fp32, I_fp32)
    
    shift = 0.0
    if (info != 0).any():
        # Fallback jitter if FP32 solve fails
        scale = float((torch.trace(M_fp32).abs() / max(n, 1)).item())
        base = max(float(jitter_rel) * max(scale, 1.0), 1e-7 * scale)
        delta = base
        for _ in range(8):
            Mt = M_fp32 + delta * I_fp32
            invM_fp32, info = torch.linalg.solve_ex(Mt, I_fp32)
            if (info == 0).all():
                shift = delta
                break
            delta *= 2.0
        else:
            raise RuntimeError("scaled_fp32: solve_ex failed")

    alpha = float(b / c)
    beta = float(a - b / c)
    
    # Q = alpha * I + (beta / c) * invM_scaled
    # Reconstruct in FP32
    Q_fp32 = alpha * I_fp32 + (beta / c) * invM_fp32
    
    return Q_fp32, float(shift)
