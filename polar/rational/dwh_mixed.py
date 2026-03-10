from __future__ import annotations

import math
from typing import Tuple

import torch

from polar.ops import symmetrize, chol_with_jitter_fp64
from polar.rational.dwh import dwh_coeffs_from_ell

Tensor = torch.Tensor

@torch.no_grad()
def dwh_step_mixed(
    S_fp32: Tensor,
    ell: float,
    jitter_rel: float,
) -> Tuple[Tensor, float]:
    """
    Mixed precision DWH step.
    Inputs S in fp32, but performs solve in fp64 for stability.
    """
    a, b, c = dwh_coeffs_from_ell(ell)
    n = S_fp32.shape[0]
    
    # Escalate to fp64 for the critical solve
    S_fp64 = S_fp32.to(torch.float64)
    I_fp64 = torch.eye(n, device=S_fp32.device, dtype=torch.float64)
    M_fp64 = symmetrize(I_fp64 + float(c) * S_fp64)
    
    L_fp64, shift = chol_with_jitter_fp64(M_fp64, jitter_rel=jitter_rel)
    invM_fp64 = torch.cholesky_inverse(L_fp64)
    
    alpha = float(b / c)
    beta = float(a - b / c)
    
    Q_fp64 = alpha * I_fp64 + beta * invM_fp64
    return Q_fp64.to(torch.float32), float(shift)
