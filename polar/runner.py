from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import torch

from polar.ops import (
    apply_right,
    apply_right_typed,
    cuda_time_ms,
    exact_eigvalsh,
    gram_xtx,
    gram_xtx_fp64,
    symmetrize,
)
from polar.polynomial.express import (
    PaperPolarExpressStep,
    polar_express_deg5_step_matrix_only,
    polar_express_paper5_step_matrix_only,
    polar_express_paper_fro_scale,
)
from polar.rational.dwh import dwh_step_matrix_only
from polar.rational.dwh_stable_solve import (
    dwh_step_matrix_only_stable_solve,
)
from polar.rational.dwh_tuned_fp32 import (
    dwh_step_tuned_fp32,
)
from polar.schedules import StepSpec

Tensor = torch.Tensor


@dataclasses.dataclass
class RunSummary:
    success: bool
    final_kO_exact: float
    steps: int
    dwh_steps: int
    zolo_steps: int
    guards: int
    fallbacks: int
    last_step_kind: str
    ms_gram: float
    ms_solve: float
    ms_upd: float
    ms_total_timed: float
    ms_exact_verify: float


@torch.no_grad()
def exact_final_kappa_O(X: Tensor, eig_device: str) -> float:
    S = gram_xtx_fp64(X)
    evals = exact_eigvalsh(S, eig_device=eig_device)
    lam_min = max(float(evals[0].item()), 1e-300)
    lam_max = max(float(evals[-1].item()), lam_min)
    return float(math.sqrt(lam_max / lam_min))


@torch.no_grad()
def run_one_case(
    G_storage: Tensor,
    target_kappa_O: float,
    schedule: Sequence[StepSpec],
    iter_dtype: torch.dtype,
    jitter_rel: float,
    tf32: bool,
    exact_verify_device: str,
) -> RunSummary:
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
    # Polar Express schedules are intended to stay in the iteration dtype end to end.
    poly_schedule = any(step.kind in {"PEADD5", "PEPAPER5"} for step in schedule)
    # Q_acc accumulates all updates to X. X_final = X_init @ Q_acc.
    q_acc_dtype = iter_dtype if poly_schedule else torch.float64
    Q_acc = torch.eye(G_storage.shape[1], device=device, dtype=q_acc_dtype)
    if poly_schedule:
        ms_upd, (X, _fro_scale) = cuda_time_ms(lambda: polar_express_paper_fro_scale(X))
        ms_upd_sum += ms_upd
    # The Gram matrix S is updated in O(n^3) to avoid O(mn^2) passes.
    if poly_schedule:
        ms_gram, S = cuda_time_ms(lambda: gram_xtx(X, iter_dtype))
    else:
        ms_gram, S = cuda_time_ms(lambda: gram_xtx_fp64(X))
    ms_gram_sum += ms_gram

    for step in schedule:
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
                ms_solve, (X, shift) = cuda_time_ms(
                    lambda: dwh_step_tuned_fp32(
                        X=X,
                        S=S,
                        ell=step.ell_in,
                        jitter_rel=jitter_rel,
                        out_dtype=iter_dtype,
                    )
                )
                ms_solve_sum += ms_solve
                guards += int(shift > 0.0)
                dwh_steps += 1
                last_step_kind = "DWH_TUNED_FP32"
                # For direct updates, we must recompute S for the next step.
                ms_gram, S = cuda_time_ms(lambda: gram_xtx_fp64(X))
                ms_gram_sum += ms_gram
                continue
            elif step.kind == "PEADD5":
                a, b, c = step.coeffs
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: polar_express_deg5_step_matrix_only(
                        S=S,
                        a=a,
                        b=b,
                        c=c,
                        matmul_dtype=iter_dtype,
                    )
                )
                last_step_kind = "PEADD5"
            elif step.kind == "PEPAPER5":
                coeffs = PaperPolarExpressStep(*step.paper_coeffs)
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: polar_express_paper5_step_matrix_only(
                        S=S,
                        coeffs=coeffs,
                        matmul_dtype=iter_dtype,
                    )
                )
                last_step_kind = "PEPAPER5"
            else:
                raise ValueError(f"Unsupported step kind: {step.kind}")
        except Exception:
            # Fallback to DWH step if something goes wrong.
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

        # Update the accumulated transform and the Gram matrix.
        # S_next = Q_step^T @ S @ Q_step. In Zolo/DWH, Q is symmetric.
        if Q_step.dtype != Q_acc.dtype:
            Q_step = Q_step.to(dtype=Q_acc.dtype)
        Q_acc = Q_acc @ Q_step
        S = symmetrize(Q_step @ S @ Q_step)
        
        ms_solve_sum += ms_solve
        guards += int(shift > 0.0)

    # Finalize the fusion: One pass over X.
    if poly_schedule:
        ms_upd, X = cuda_time_ms(
            lambda: apply_right_typed(X, Q_acc, iter_dtype, iter_dtype)
        )
    else:
        ms_upd, X = cuda_time_ms(
            lambda: apply_right(X, Q_acc, iter_dtype)
        )
    ms_upd_sum += ms_upd

    steps_used = len(schedule)

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
