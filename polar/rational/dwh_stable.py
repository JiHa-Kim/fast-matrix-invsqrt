from __future__ import annotations

import math
from typing import Tuple

import torch

from polar.ops import symmetrize

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
def dwh_step_matrix_only_stable(
    S: Tensor,
    ell: float,
    jitter_rel: float,
    force_fp32_solver: bool = False,
) -> Tuple[Tensor, float]:
    """
    DWH step using torch.linalg.solve_ex for stability.
    If force_fp32_solver is True, the solver part runs in fp32 regardless of S.dtype.
    """
    a, b, c = dwh_coeffs_from_ell(ell)
    n = S.shape[0]
    device = S.device
    orig_dtype = S.dtype
    
    solver_dtype = torch.float32 if force_fp32_solver else orig_dtype
    
    I_solver = torch.eye(n, device=device, dtype=solver_dtype)
    M_solver = symmetrize(I_solver + float(c) * S.to(solver_dtype))
    
    # Q = alpha * I + beta * M^{-1}
    # We can compute M^{-1} using solve_ex
    invM, info = torch.linalg.solve_ex(M_solver, I_solver)
    
    shift = 0.0
    if (info != 0).any():
        scale = float((torch.trace(M_solver).abs() / max(n, 1)).item())
        base = max(float(jitter_rel) * max(scale, 1.0), 1e-30)
        delta = base
        for _ in range(8):
            Mt = M_solver + delta * I_solver
            invM, info = torch.linalg.solve_ex(Mt, I_solver)
            if (info == 0).all():
                shift = delta
                break
            delta *= 2.0
        else:
            raise RuntimeError("solve_ex failed even after jitter escalation")

    alpha = float(b / c)
    beta = float(a - b / c)
    
    # We return Q in the original dtype
    I_orig = torch.eye(n, device=device, dtype=orig_dtype)
    Q = alpha * I_orig + beta * invM.to(orig_dtype)
    return Q, float(shift)


