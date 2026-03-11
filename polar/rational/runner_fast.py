from __future__ import annotations
from typing import Sequence

import torch

from polar.ops import (
    cuda_time_ms,
    symmetrize,
)
from polar.rational.ops import (
    apply_right_fast_full,
    gram_xtx_fast,
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
from polar.runner import RunSummary, exact_final_kappa_O

Tensor = torch.Tensor

@torch.no_grad()
def run_one_case_fast(
    G_storage: Tensor,
    target_kappa_O: float,
    schedule: Sequence[StepSpec],
    iter_dtype: torch.dtype,
    jitter_rel: float,
    tf32: bool,
    exact_verify_device: str,
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
    ms_gram, S = cuda_time_ms(lambda: gram_xtx_fast(X, iter_dtype))
    ms_gram_sum += ms_gram

    def flush_q_acc() -> None:
        nonlocal X, Q_acc, q_acc_dirty, ms_upd_sum
        if not q_acc_dirty:
            return
        ms_upd, X_next = cuda_time_ms(
            lambda: apply_right_fast_full(X, Q_acc, iter_dtype)
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
            else:
                raise ValueError(f"Unsupported step kind for fast runner: {step.kind}")
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
        lambda: apply_right_fast_full(X, Q_acc, iter_dtype)
    )
    ms_upd_sum += ms_upd

    steps_used = len(schedule)

    # Verification still needs to be accurate
    ms_exact_verify, final_kO_exact = cuda_time_ms(
        lambda: exact_final_kappa_O(X, exact_verify_device)
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
