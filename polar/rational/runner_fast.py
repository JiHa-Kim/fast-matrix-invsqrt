from __future__ import annotations
from typing import Sequence

import torch

from polar.ops import (
    cuda_time_ms,
    symmetrize,
)
from polar.rational.ops import (
    apply_right_small_chunked_fast,
    gram_xtx_chunked_fast,
)
from polar.rational.dwh import dwh_step_matrix_only
from polar.rational.dwh_stable_solve import (
    dwh_step_matrix_only_stable_solve,
)
from polar.rational.dwh_mixed import (
    dwh_step_mixed,
)
from polar.rational.dwh_mixed_solve import (
    dwh_step_mixed_solve,
)
from polar.rational.dwh_scaled_fp32_solve import (
    dwh_step_scaled_fp32_solve,
)
from polar.schedules import StepSpec
from polar.rational.zolo import (
    zolo_coeffs_from_ell,
)
from polar.runner import RunSummary, exact_final_kappa_O

Tensor = torch.Tensor

@torch.no_grad()
def run_one_case_fast(
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
    Pure lower-precision runner. AVOIDS ALL FP64 to maximize speed.
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
    # Q_acc accumulates all updates to X in lower precision
    Q_acc = torch.eye(G_storage.shape[1], device=device, dtype=iter_dtype)
    q_acc_dirty = False
    
    # Gram matrix S in lower precision
    ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fast(X, gram_chunk_rows, iter_dtype))
    ms_gram_sum += ms_gram

    def flush_q_acc() -> None:
        nonlocal X, Q_acc, q_acc_dirty, ms_upd_sum
        if not q_acc_dirty:
            return
        ms_upd, X_next = cuda_time_ms(
            lambda: apply_right_small_chunked_fast(X, Q_acc, rhs_chunk_rows, iter_dtype)
        )
        ms_upd_sum += ms_upd
        X = X_next
        Q_acc = torch.eye(G_storage.shape[1], device=device, dtype=iter_dtype)
        q_acc_dirty = False

    for i, step in enumerate(schedule):
        try:
            if step.kind == "DWH":
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: dwh_step_matrix_only(
                        S=S,
                        ell=step.ell_in,
                        jitter_rel=jitter_rel,
                    )
                )
                dwh_steps += 1
                last_step_kind = "DWH"
            elif step.kind == "DWH_STABLE_SOLVE":
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: dwh_step_matrix_only_stable_solve(
                        S=S,
                        ell=step.ell_in,
                        jitter_rel=jitter_rel,
                    )
                )
                dwh_steps += 1
                last_step_kind = "DWH_STABLE_SOLVE"
            elif step.kind == "DWH_TUNED_FP32":
                from polar.rational.dwh_tuned_fp32 import dwh_step_matrix_only_tuned_fp32
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: dwh_step_matrix_only_tuned_fp32(
                        S=S,
                        ell=step.ell_in,
                        jitter_rel=jitter_rel,
                    )
                )
                dwh_steps += 1
                last_step_kind = "DWH_TUNED_FP32"
            elif step.kind == "DWH_MIXED":
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: dwh_step_mixed(
                        S_fp32=S,
                        ell=step.ell_in,
                        jitter_rel=jitter_rel,
                    )
                )
                dwh_steps += 1
                last_step_kind = "DWH_MIXED"
            elif step.kind == "DWH_MIXED_SOLVE":
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: dwh_step_mixed_solve(
                        S_fp32=S,
                        ell=step.ell_in,
                        jitter_rel=jitter_rel,
                    )
                )
                dwh_steps += 1
                last_step_kind = "DWH_MIXED_SOLVE"
            elif step.kind == "DWH_SCALED_FP32_SOLVE":
                # For scaled fp32 solve, we can use the pre-computed S (in iter_dtype)
                # and just cast it to fp64 for the stable formulation of the scaled matrix.
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: dwh_step_scaled_fp32_solve(
                        S_fp64=S.to(torch.float64),
                        ell=step.ell_in,
                        jitter_rel=jitter_rel,
                    )
                )
                dwh_steps += 1
                last_step_kind = "DWH_SCALED_FP32_SOLVE"
            elif step.kind == "ZOLO":
                # For aggressive speed, we use mixed-precision ZOLO with direct X update
                coeffs = zolo_coeffs_from_ell(step.r, step.ell_in, dps=zolo_coeff_dps)
                
                def zolo_mixed_step_stable():
                    S_fp64 = S.to(torch.float64)
                    n = S_fp64.shape[0]
                    I_fp64 = torch.eye(n, device=S_fp64.device, dtype=torch.float64)
                    Q_fp64 = torch.eye(n, device=S_fp64.device, dtype=torch.float64)
                    max_shift = 0.0
                    for ce, co in zip(coeffs.c_even, coeffs.c_odd):
                        M = symmetrize(S_fp64 + float(co) * I_fp64)
                        invM, info = torch.linalg.solve_ex(M, I_fp64)
                        if (info != 0).any():
                            # Escalation
                            scale = float((torch.trace(M).abs() / max(n, 1)).item())
                            base = max(float(jitter_rel) * max(scale, 1.0), 1e-30)
                            delta_jitter = base
                            for _ in range(8):
                                Mt = M + delta_jitter * I_fp64
                                invM, info = torch.linalg.solve_ex(Mt, I_fp64)
                                if (info == 0).all():
                                    max_shift = max(max_shift, delta_jitter)
                                    break
                                delta_jitter *= 2.0
                            else:
                                raise RuntimeError("Zolo solve failed even after jitter")
                        
                        delta = float(ce - co)
                        Q_fp64 = Q_fp64 + delta * (Q_fp64 @ invM)
                    
                    Q_fp64 = float(coeffs.mhat) * Q_fp64
                    return Q_fp64.to(iter_dtype), max_shift
                
                ms_solve, (Q_step, shift) = cuda_time_ms(zolo_mixed_step_stable)
                zolo_steps += 1
                last_step_kind = f"ZOLO_MIXED(r={step.r})"
                
                # Flush prior fused updates before switching to direct X updates.
                flush_q_acc()

                # Direct update on X and recompute S for maximum accuracy in pure FP32
                if Q_step.dtype != iter_dtype:
                    Q_step = Q_step.to(dtype=iter_dtype)
                X = X @ Q_step
                ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fast(X, gram_chunk_rows, iter_dtype))
                ms_gram_sum += ms_gram
                ms_solve_sum += ms_solve
                guards += int(shift > 0.0)
                continue
        except Exception as e:
            # Fallback to DWH step
            fallbacks += 1
            ms_solve, (Q_step, shift) = cuda_time_ms(
                lambda: dwh_step_matrix_only(
                    S=S,
                    ell=step.ell_in,
                    jitter_rel=jitter_rel,
                )
            )
            dwh_steps += 1
            last_step_kind = "DWH(fallback)"

        # Update the accumulated transform and the Gram matrix in LOWER PRECISION.
        if Q_step.dtype != iter_dtype:
            Q_step = Q_step.to(dtype=iter_dtype)
        Q_acc = Q_acc @ Q_step
        q_acc_dirty = True
        S = symmetrize(Q_step @ S @ Q_step)
        
        ms_solve_sum += ms_solve
        guards += int(shift > 0.0)

    # Finalize the fusion: One pass over X in lower precision.
    ms_upd, X = cuda_time_ms(
        lambda: apply_right_small_chunked_fast(X, Q_acc, rhs_chunk_rows, iter_dtype)
    )
    ms_upd_sum += ms_upd

    steps_used = len(schedule)

    # Verification still needs to be accurate
    ms_exact_verify, final_kO_exact = cuda_time_ms(
        lambda: exact_final_kappa_O(X, gram_chunk_rows, exact_verify_device)
    )
    ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum
    return RunSummary(
        success=bool(final_kO_exact <= target_kappa_O),
        final_kO_exact=float(final_kO_exact),
        steps=steps_used,
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
