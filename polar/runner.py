from __future__ import annotations

import dataclasses
import math
from typing import Sequence

import torch

from polar.ops import (
    apply_right_small_chunked,
    cert_bound_trace_logdet,
    cuda_time_ms,
    exact_eigvalsh,
    gram_xtx_chunked_fp64,
    symmetrize,
)
from polar.dwh import (
    dwh_ell_next,
    dwh_step_chunked,
    dwh_step_matrix_only,
)
from polar.schedules import StepSpec
from polar.zolo import (
    zolo_coeffs_from_ell,
    zolo_step_matrix_only,
)

Tensor = torch.Tensor


@dataclasses.dataclass
class RunSummary:
    success: bool
    final_kO_exact: float
    final_kO_cert: float
    steps: int
    dwh_steps: int
    zolo_steps: int
    guards: int
    fallbacks: int
    last_step_kind: str
    ms_gram: float
    ms_solve: float
    ms_upd: float
    ms_cert: float
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
    cert_jitter_rel: float,
    tf32: bool,
    exact_verify_device: str,
    zolo_coeff_dps: int,
    stop_on_cert: bool,
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
    ms_cert_sum = 0.0
    dwh_steps = 0
    zolo_steps = 0
    guards = 0
    fallbacks = 0
    last_step_kind = "none"
    final_kO_cert = float("inf")

    # Q_acc accumulates all updates to X. X_final = X_init @ Q_acc.
    Q_acc = torch.eye(G_storage.shape[1], device=device, dtype=torch.float64)
    # The Gram matrix S is updated in O(n^3) to avoid O(mn^2) passes.
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
        Q_acc = Q_acc @ Q_step
        S = symmetrize(Q_step @ S @ Q_step)
        
        ms_solve_sum += ms_solve
        guards += int(shift > 0.0)

    # Finalize the fusion: One pass over X.
    ms_upd, X = cuda_time_ms(
        lambda: apply_right_small_chunked(X, Q_acc, rhs_chunk_rows, iter_dtype)
    )
    ms_upd_sum += ms_upd

    final_kO_cert = float("inf")
    steps_used = len(schedule)

    if stop_on_cert:
        # Note: S is already up to date from O(n^3) updates.
        ms_cert, (kO_cert, cert_shift) = cuda_time_ms(
            lambda: cert_bound_trace_logdet(S, cert_jitter_rel)
        )
        ms_cert_sum += ms_cert
        guards += int(cert_shift > 0.0)
        final_kO_cert = float(kO_cert)

        ell = schedule[-1].ell_out if schedule else 1.0
        while final_kO_cert > target_kappa_O and steps_used < 16:
            # For polish steps, we just do it iteratively on X for simplicity
            # as it should be very few steps.
            ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fp64(X, gram_chunk_rows))
            ms_gram_sum += ms_gram
            
            ms_solve, (X, shift) = cuda_time_ms(
                lambda: dwh_step_chunked(
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
            last_step_kind = "DWH(polish)"

            # Re-update S for cert
            ms_gram, S = cuda_time_ms(lambda: gram_xtx_chunked_fp64(X, gram_chunk_rows))
            ms_gram_sum += ms_gram
            ms_cert, (kO_cert, cert_shift) = cuda_time_ms(
                lambda: cert_bound_trace_logdet(S, cert_jitter_rel)
            )
            ms_cert_sum += ms_cert
            guards += int(cert_shift > 0.0)
            final_kO_cert = float(kO_cert)
            ell = dwh_ell_next(ell)
            steps_used += 1
    else:
        # S is already up to date.
        ms_cert, (kO_cert, cert_shift) = cuda_time_ms(
            lambda: cert_bound_trace_logdet(S, cert_jitter_rel)
        )
        ms_cert_sum += ms_cert
        guards += int(cert_shift > 0.0)
        final_kO_cert = float(kO_cert)

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
