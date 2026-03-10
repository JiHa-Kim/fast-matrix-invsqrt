from __future__ import annotations

import math
from typing import Tuple

import torch

from polar.ops import symmetrize
from polar.rational.dwh import dwh_coeffs_from_ell

Tensor = torch.Tensor

@torch.no_grad()
def dwh_step_mixed_solve(
    S_fp32: Tensor,
    ell: float,
    jitter_rel: float,
) -> Tuple[Tensor, float]:
    """
    Mixed precision DWH step using solve_ex in fp64.
    Inputs S in fp32, but performs solve in fp64 for stability.
    """
    a, b, c = dwh_coeffs_from_ell(ell)
    n = S_fp32.shape[0]
    
    # Escalate to fp64 for the critical solve
    S_fp64 = S_fp32.to(torch.float64)
    I_fp64 = torch.eye(n, device=S_fp32.device, dtype=torch.float64)
    M_fp64 = symmetrize(I_fp64 + float(c) * S_fp64)
    
    # solve_ex handles indefinite matrices more gracefully than Cholesky
    invM_fp64, info = torch.linalg.solve_ex(M_fp64, I_fp64)
    
    shift = 0.0
    if (info != 0).any():
        scale = float((torch.trace(M_fp64).abs() / max(n, 1)).item())
        base = max(float(jitter_rel) * max(scale, 1.0), 1e-30)
        delta = base
        for _ in range(8):
            Mt = M_fp64 + delta * I_fp64
            invM_fp64, info = torch.linalg.solve_ex(Mt, I_fp64)
            if (info == 0).all():
                shift = delta
                break
            delta *= 2.0
        else:
            raise RuntimeError("mixed_solve: solve_ex failed even after jitter")

    alpha = float(b / c)
    beta = float(a - b / c)
    
    Q_fp64 = alpha * I_fp64 + beta * invM_fp64
    return Q_fp64.to(torch.float32), float(shift)
