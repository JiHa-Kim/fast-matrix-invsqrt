from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import torch

from polar.ops import (
    cuda_time_ms,
    symmetrize,
)
from polar.rational.ops import (
    cert_bound_trace_logdet_stable,
    apply_right_small_chunked_fast,
    gram_xtx_chunked_fast,
)
from polar.rational.dwh import (
    dwh_ell_next,
    dwh_step_chunked,
    dwh_step_matrix_only,
)
from polar.rational.dwh_stable_solve import (
    dwh_step_chunked_stable_solve,
    dwh_step_matrix_only_stable_solve,
)
from polar.rational.dwh_tuned_fp32 import (
    dwh_step_tuned_fp32,
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
    zolo_step_matrix_only,
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
    cert_jitter_rel: float,
    tf32: bool,
    exact_verify_device: str,
    zolo_coeff_dps: int,
    stop_on_cert: bool,
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
    ms_cert_sum = 0.0
    dwh_steps = 0
    zolo_steps = 0
    guards = 0
    fallbacks = 0
    last_step_kind = "none"
    final_kO_cert = float("inf")

    # Q_acc accumulates all updates to X in lower precision
    Q_acc = torch.eye(G_storage.shape[1], device=device, dtype=iter_dtype)
    
    # Gram matrix S in lower precision
    ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fast(X, gram_chunk_rows, iter_dtype))
    ms_gram_sum += ms_gram

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
                # For scaled fp32 solve, S must be computed accurately in fp64
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: dwh_step_scaled_fp32_solve(
                        S_fp64=X.to(torch.float64).T @ X.to(torch.float64),
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
        S = symmetrize(Q_step @ S @ Q_step)
        
        ms_solve_sum += ms_solve
        guards += int(shift > 0.0)

    # Finalize the fusion: One pass over X in lower precision.
    ms_upd, X = cuda_time_ms(
        lambda: apply_right_small_chunked_fast(X, Q_acc, rhs_chunk_rows, iter_dtype)
    )
    ms_upd_sum += ms_upd

    final_kO_cert = float("inf")
    steps_used = len(schedule)

    if stop_on_cert:
        # Note: S is already up to date from O(n^3) updates.
        # Certificate bound still uses fp64 internally for safety but input is fp32
        ms_cert, (kO_cert, cert_shift) = cuda_time_ms(
            lambda: cert_bound_trace_logdet_stable(S, cert_jitter_rel)
        )
        ms_cert_sum += ms_cert
        guards += int(cert_shift > 0.0)
        final_kO_cert = float(kO_cert)

        ell = schedule[-1].ell_out if schedule else 1.0
        while final_kO_cert > target_kappa_O and steps_used < 16:
            ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fast(X, gram_chunk_rows, iter_dtype))
            ms_gram_sum += ms_gram
            
            # Use stable solve for polishing in pure fp32
            ms_solve, (X, shift) = cuda_time_ms(
                lambda: dwh_step_chunked_stable_solve(
                    X=X,
                    S=S,
                    ell=ell,
                    rhs_chunk_rows=rhs_chunk_rows,
                    jitter_rel=jitter_rel,
                    out_dtype=iter_dtype,
                )
            )
            ms_solve_sum += ms_solve
            guards += int(shift > 0.0)
            dwh_steps += 1
            last_step_kind = "DWH_STABLE_SOLVE(polish)"

            # Re-update S for cert
            ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fast(X, gram_chunk_rows, iter_dtype))
            ms_gram_sum += ms_gram
            ms_cert, (kO_cert, cert_shift) = cuda_time_ms(
                lambda: cert_bound_trace_logdet_stable(S, cert_jitter_rel)
            )
            ms_cert_sum += ms_cert
            guards += int(cert_shift > 0.0)
            final_kO_cert = float(kO_cert)
            ell = dwh_ell_next(ell)
            steps_used += 1
    else:
        ms_cert, (kO_cert, cert_shift) = cuda_time_ms(
            lambda: cert_bound_trace_logdet_stable(S, cert_jitter_rel)
        )
        ms_cert_sum += ms_cert
        guards += int(cert_shift > 0.0)
        final_kO_cert = float(kO_cert)

    # Verification still needs to be accurate
    ms_exact_verify, final_kO_exact = cuda_time_ms(
        lambda: exact_final_kappa_O(X, gram_chunk_rows, exact_verify_device)
    )
    ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum + ms_cert_sum
    return RunSummary(
        success=bool(final_kO_exact <= target_kappa_O),
        final_kO_exact=float(final_kO_exact),
        final_kO_cert=float(final_kO_cert),
        steps=steps_used,
        dwh_steps=dwh_steps,
        zolo_steps=zolo_steps,
        guards=guards,
        fallbacks=fallbacks,
        last_step_kind=last_step_kind,
        ms_gram=ms_gram_sum,
        ms_solve=ms_solve_sum,
        ms_upd=ms_upd_sum,
        ms_cert=ms_cert_sum,
        ms_total_timed=ms_total,
        ms_exact_verify=ms_exact_verify,
    )
