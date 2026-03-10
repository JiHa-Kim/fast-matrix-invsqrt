from __future__ import annotations
from typing import Sequence

import torch

from polar.ops import (
    cuda_time_ms,
    symmetrize,
)
from polar.rational.ops import (
    gram_xtx_chunked_fast,
)
from polar.rational.dwh_stable_solve import (
    dwh_step_matrix_only_stable_solve,
)
from polar.rational.dwh_tuned_fp32 import (
    dwh_step_tuned_fp32,
)
from polar.schedules import StepSpec
from polar.rational.zolo import (
    zolo_coeffs_from_ell,
)
from polar.runner import RunSummary, exact_final_kappa_O

Tensor = torch.Tensor

@torch.no_grad()
def run_one_case_ultra_fast(
    G_storage: Tensor,
    target_kappa_O: float,
    schedule: Sequence[StepSpec],
    iter_dtype: torch.dtype,
    gram_chunk_rows: int,
    rhs_chunk_rows: int,
    jitter_rel: float,
    tf32: bool,
    exact_verify_device: str,
    zolo_coeff_dps: int,
) -> RunSummary:
    """
    ULTRA-FAST pure FP32 runner.
    Avoids Q accumulation noise by updating X directly and recomputing S.
    Uses TF32 for blistering O(mn^2) speeds.
    """
    device = G_storage.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        if tf32:
            torch.set_float32_matmul_precision("high")
        else:
            torch.set_float32_matmul_precision("highest")

    X = G_storage.to(dtype=iter_dtype)
    ms_gram_sum = 0.0
    ms_solve_sum = 0.0
    ms_upd_sum = 0.0
    dwh_steps = 0
    zolo_steps = 0
    guards = 0
    fallbacks = 0
    last_step_kind = "none"

    for i, step in enumerate(schedule):
        # 1. Compute S accurately and fast using TF32
        ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fast(X, gram_chunk_rows, iter_dtype))
        ms_gram_sum += ms_gram
        
        try:
            if step.kind == "DWH_TUNED_FP32":
                # Tuned DWH is designed for this!
                ms_solve, (X, shift) = cuda_time_ms(
                    lambda: dwh_step_tuned_fp32(
                        X=X,
                        S=S,
                        ell=step.ell_in,
                        rhs_chunk_rows=rhs_chunk_rows,
                        jitter_rel=jitter_rel,
                        out_dtype=iter_dtype,
                    )
                )
                dwh_steps += 1
                last_step_kind = "DWH_TUNED_FP32"
            elif step.kind == "DWH_SCALED_FP32_SOLVE":
                from polar.rational.dwh_scaled_fp32_solve import dwh_step_scaled_fp32_solve
                # For scaled fp32 solve, we can use the pre-computed S
                # but we cast to fp64 for the stable Formulation of the scaled matrix.
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: dwh_step_scaled_fp32_solve(
                        S_fp64=S.to(torch.float64),
                        ell=step.ell_in,
                        jitter_rel=jitter_rel,
                    )
                )
                dwh_steps += 1
                last_step_kind = "DWH_SCALED_FP32_SOLVE"
                
                # Update X: O(mn^2) but fast with TF32
                def update_x_scaled():
                    return X @ Q_step.to(iter_dtype)
                ms_upd, X = cuda_time_ms(update_x_scaled)
                ms_upd_sum += ms_upd
            else:
                # For other steps, compute Q and update X
                if step.kind == "ZOLO":
                    coeffs = zolo_coeffs_from_ell(step.r, step.ell_in, dps=zolo_coeff_dps)
                    # Use a stable version of ZOLO solve
                    def zolo_solve():
                        n = S.shape[0]
                        I = torch.eye(n, device=device, dtype=iter_dtype)
                        Q = torch.eye(n, device=device, dtype=iter_dtype)
                        max_s = 0.0
                        for ce, co in zip(coeffs.c_even, coeffs.c_odd):
                            M = symmetrize(S + float(co) * I)
                            invM, info = torch.linalg.solve_ex(M, I)
                            if (info != 0).any():
                                raise RuntimeError("Zolo solve failed")
                            delta = float(ce - co)
                            Q = Q + delta * (Q @ invM)
                        Q = float(coeffs.mhat) * Q
                        return Q, max_s
                    ms_solve, (Q_step, shift) = cuda_time_ms(zolo_solve)
                    zolo_steps += 1
                    last_step_kind = f"ZOLO(r={step.r})"
                elif step.kind == "DWH_STABLE_SOLVE" or step.kind == "DWH":
                    ms_solve, (Q_step, shift) = cuda_time_ms(
                        lambda: dwh_step_matrix_only_stable_solve(S, step.ell_in, jitter_rel)
                    )
                    dwh_steps += 1
                    last_step_kind = step.kind
                else:
                    raise ValueError(f"Unsupported step kind for ultra-fast: {step.kind}")
                
                # Update X: O(mn^2) but fast with TF32
                def update_x():
                    return X @ Q_step.to(iter_dtype)
                ms_upd, X = cuda_time_ms(update_x)
                ms_upd_sum += ms_upd
                
            ms_solve_sum += ms_solve
            guards += int(shift > 0.0)
            
        except Exception:
            fallbacks += 1
            # Simple fallback: NS(1) or just continue? 
            # For now, let's just fail or do a very safe DWH step
            continue

    # Verification
    ms_exact_verify, final_kO_exact = cuda_time_ms(
        lambda: exact_final_kappa_O(X, gram_chunk_rows, exact_verify_device)
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
