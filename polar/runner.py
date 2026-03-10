from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import torch

from polar.ops import (
    apply_right_small_chunked,
    apply_right_small_chunked_typed,
    cuda_time_ms,
    exact_eigvalsh,
    gram_xtx_chunked,
    gram_xtx_chunked_fp64,
    symmetrize,
)
from polar.polynomial.express import PolarExpressStep, polar_express_fro_scale, polar_express_step_matrix_only
from polar.polynomial.minimax import poly_inv_sqrt_coeffs_from_ell, poly_step_matrix_only
from polar.rational.dwh import dwh_step_matrix_only
from polar.rational.dwh_stable_solve import (
    dwh_step_matrix_only_stable_solve,
)
from polar.rational.dwh_tuned_fp32 import (
    dwh_step_tuned_fp32,
)
from polar.schedules import StepSpec
from polar.rational.zolo import zolo_coeffs_from_ell, zolo_step_matrix_only

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
def exact_final_kappa_O(X: Tensor, gram_chunk_rows: int, eig_device: str) -> float:
    S = gram_xtx_chunked_fp64(X, gram_chunk_rows)
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
    gram_chunk_rows: int,
    rhs_chunk_rows: int,
    jitter_rel: float,
    tf32: bool,
    exact_verify_device: str,
    zolo_coeff_dps: int,
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
    # Polynomial schedules are intended to stay in the iteration dtype end to end.
    poly_schedule = any(step.kind in {"POLY", "PE"} for step in schedule)

    # Q_acc accumulates all updates to X. X_final = X_init @ Q_acc.
    q_acc_dtype = iter_dtype if poly_schedule else torch.float64
    Q_acc = torch.eye(G_storage.shape[1], device=device, dtype=q_acc_dtype)
    if poly_schedule:
        ms_upd, (X, _fro_scale) = cuda_time_ms(lambda: polar_express_fro_scale(X))
        ms_upd_sum += ms_upd
    # The Gram matrix S is updated in O(n^3) to avoid O(mn^2) passes.
    if poly_schedule:
        ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked(X, gram_chunk_rows, iter_dtype))
    else:
        ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fp64(X, gram_chunk_rows))
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
                ms_solve_sum += ms_solve
                guards += int(shift > 0.0)
                dwh_steps += 1
                last_step_kind = "DWH_TUNED_FP32"
                # For direct updates, we must recompute S for the next step.
                ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fp64(X, gram_chunk_rows))
                ms_gram_sum += ms_gram
                continue
            elif step.kind == "POLY":
                coeffs = poly_inv_sqrt_coeffs_from_ell(step.degree, step.ell_in)
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: poly_step_matrix_only(
                        S=S,
                        coeffs=coeffs,
                        matmul_dtype=iter_dtype,
                    )
                )
                last_step_kind = f"POLY(d={step.degree})"
            elif step.kind == "PE":
                coeffs = PolarExpressStep(
                    sigma_lo=step.ell_in,
                    sigma_hi=step.u_in,
                    degree_q=step.pe_degree,
                    basis=step.basis,
                    anchored=step.pe_anchored,
                    interval_lo=step.pe_interval_lo,
                    interval_hi=step.pe_interval_hi,
                    coeffs=step.pe_coeffs,
                    shifted_coeffs=step.pe_shifted_coeffs,
                    shift_center=step.pe_shift_center,
                    shift_scale=step.pe_shift_scale,
                    shift_gain=step.pe_shift_gain,
                    max_step_err=float("nan"),
                    pred_sigma_min=step.ell_out,
                    pred_sigma_max=step.u_out,
                )
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: polar_express_step_matrix_only(
                        S=S,
                        coeffs=coeffs,
                        matmul_dtype=iter_dtype,
                    )
                )
                last_step_kind = f"PEq{step.pe_degree}({step.basis})"
            else:
                coeffs = zolo_coeffs_from_ell(step.r, step.ell_in, dps=zolo_coeff_dps)
                ms_solve, (Q_step, shift) = cuda_time_ms(
                    lambda: zolo_step_matrix_only(
                        S=S,
                        coeffs=coeffs,
                        jitter_rel=jitter_rel,
                    )
                )
                zolo_steps += 1
                last_step_kind = f"ZOLO(r={step.r})"
        except Exception:
            # Fallback to DWH step if something goes wrong (e.g. ZOLO instability)
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
            lambda: apply_right_small_chunked_typed(X, Q_acc, rhs_chunk_rows, iter_dtype, iter_dtype)
        )
    else:
        ms_upd, X = cuda_time_ms(
            lambda: apply_right_small_chunked(X, Q_acc, rhs_chunk_rows, iter_dtype)
        )
    ms_upd_sum += ms_upd

    steps_used = len(schedule)

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
