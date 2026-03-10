from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import torch

from polar.ops import (
    cuda_time_ms,
    gram_xtx_fp64,
    symmetrize,
)
from polar.rational.dwh import (
    dwh_ell_next,
    dwh_step_chunked,
    dwh_step_matrix_only,
)
from polar.rational.dwh_scaled_fp32_solve import (
    dwh_step_scaled_fp32_solve,
)
from polar.schedules import StepSpec
from polar.rational.zolo import (
    zolo_coeffs_from_ell,
    zolo_step_matrix_only,
)
from polar.runner import RunSummary, exact_final_kappa_O, polar_express_fro_scale

Tensor = torch.Tensor

@torch.no_grad()
def run_one_case_hybrid(
    G_storage: Tensor,
    target_kappa_O: float,
    schedule: Sequence[StepSpec],
    iter_dtype: torch.dtype,
    jitter_rel: float,
    tf32: bool,
    exact_verify_device: str,
    zolo_coeff_dps: int,
) -> RunSummary:
    """
    HYBRID runner: FP64 state maintenance, but FP32 Scaled Solver.
    Combines the O(mn^2) efficiency of the baseline with the O(n^3) speed of FP32.
    """
    device = G_storage.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)

    X = G_storage.to(dtype=iter_dtype)
    ms_gram_sum = 0.0
    ms_solve_sum = 0.0
    ms_upd_sum = 0.0
    dwh_steps = 0
    zolo_steps = 0
    guards = 0
    fallbacks = 0
    last_step_kind = "none"

    # Q_acc maintains the cumulative isometry in FP64 for stability
    Q_acc = torch.eye(G_storage.shape[1], device=device, dtype=torch.float64)
    
    # Gram matrix S in FP64
    ms_gram, S = cuda_time_ms(lambda: gram_xtx_fp64(X))
    ms_gram_sum += ms_gram

    for i, step in enumerate(schedule):
        try:
            if step.kind == "DWH_SCALED_FP32_SOLVE":
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: dwh_step_scaled_fp32_solve(
                        S_fp64=S,
                        ell=step.ell_in,
                        jitter_rel=jitter_rel,
                    )
                )
                dwh_steps += 1
                last_step_kind = "DWH_SCALED_FP32_SOLVE"
            elif step.kind == "DWH":
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: dwh_step_matrix_only(S, step.ell_in, jitter_rel)
                )
                dwh_steps += 1
                last_step_kind = "DWH"
            elif step.kind == "ZOLO":
                coeffs = zolo_coeffs_from_ell(step.r, step.ell_in, dps=zolo_coeff_dps)
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: zolo_step_matrix_only(S, coeffs, jitter_rel)
                )
                zolo_steps += 1
                last_step_kind = f"ZOLO(r={step.r})"
            else:
                raise ValueError(f"Unsupported kind for hybrid: {step.kind}")
        except Exception:
            fallbacks += 1
            # Fallback to standard DWH solve in FP64
            ms_solve, (Q_step, shift) = cuda_time_ms(
                lambda: dwh_step_matrix_only(S, step.ell_in, jitter_rel)
            )
            dwh_steps += 1
            last_step_kind = "DWH(fallback)"

        # Update Q_acc and S in FP64 (O(n^3))
        Q_step = Q_step.to(torch.float64)
        Q_acc = Q_acc @ Q_step
        S = symmetrize(Q_step.T @ S @ Q_step)
        
        ms_solve_sum += ms_solve
        guards += int(shift > 0.0)

    # Finalize fusion: ONE pass over X (O(mn^2))
    # We can use TF32 here if iter_dtype is FP32
    from polar.rational.ops import apply_right_fast_full
    ms_upd, X = cuda_time_ms(
        lambda: apply_right_fast_full(X, Q_acc.to(iter_dtype), iter_dtype)
    )
    ms_upd_sum += ms_upd

    # Verification
    ms_exact_verify, final_kO_exact = cuda_time_ms(
        lambda: exact_final_kappa_O(X, exact_verify_device)
    )
    
    ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum
    return RunSummary(
        success=bool(final_kO_exact <= target_kappa_O),
        final_kO_exact=float(final_kO_exact),
        steps=len(schedule),
        dwh_steps=dwh_steps,
        zolo_steps=zolo_steps,
        guards=guards,
        fallbacks=fallbacks,
        last_step_kind=last_step_kind,
        ms_gram=ms_gram_sum,
        ms_solve=ms_solve_sum,
        ms_upd=ms_upd_sum,
        ms_total_timed=ms_total,
        ms_exact_verify=ms_exact_verify,
    )
