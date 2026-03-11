from __future__ import annotations

from typing import Sequence

import torch

from polar.ops import cuda_time_ms
from polar.rational.dwh_stable_solve import dwh_step_matrix_only_stable_solve
from polar.rational.dwh_tuned_fp32 import dwh_step_tuned_fp32
from polar.rational.ops import gram_xtx_fast
from polar.runner import RunSummary, exact_final_kappa_O
from polar.schedules import StepSpec

Tensor = torch.Tensor


@torch.no_grad()
def run_one_case_tf32_rational(
    G_storage: Tensor,
    target_kappa_O: float,
    schedule: Sequence[StepSpec],
    iter_dtype: torch.dtype,
    jitter_rel: float,
    tf32: bool,
    exact_verify_device: str,
) -> RunSummary:
    """
    Rational-only runner that keeps the iterate in float32 and relies on TF32
    GEMMs for the large-side work. Small-side solves still use float32.
    """
    device = G_storage.device
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high" if tf32 else "highest")

    X = G_storage.to(dtype=torch.float32)
    ms_gram_sum = 0.0
    ms_solve_sum = 0.0
    ms_upd_sum = 0.0
    dwh_steps = 0
    zolo_steps = 0
    guards = 0
    fallbacks = 0
    last_step_kind = "none"

    for step in schedule:
        ms_gram, S = cuda_time_ms(lambda: gram_xtx_fast(X, torch.float32))
        ms_gram_sum += ms_gram

        try:
            if step.kind == "DWH_TUNED_FP32":
                ms_solve, (X, shift) = cuda_time_ms(
                    lambda: dwh_step_tuned_fp32(
                        X=X,
                        S=S,
                        ell=step.ell_in,
                        jitter_rel=jitter_rel,
                        out_dtype=torch.float32,
                    )
                )
                dwh_steps += 1
                last_step_kind = "DWH_TUNED_FP32"
                ms_solve_sum += ms_solve
                guards += int(shift > 0.0)
                continue

            if step.kind not in {"DWH", "DWH_STABLE_SOLVE", "DWH_SCALED_FP32_SOLVE"}:
                raise ValueError(f"Unsupported step kind for TF32 rational runner: {step.kind}")

            ms_solve, (Q_step, shift) = cuda_time_ms(
                lambda: dwh_step_matrix_only_stable_solve(
                    S=S,
                    ell=step.ell_in,
                    jitter_rel=jitter_rel,
                )
            )
            dwh_steps += 1
            last_step_kind = "DWH_STABLE_SOLVE" if step.kind != "DWH_TUNED_FP32" else step.kind

            ms_upd, X = cuda_time_ms(lambda: (X @ Q_step.to(dtype=torch.float32)).to(dtype=torch.float32))
            ms_upd_sum += ms_upd
            ms_solve_sum += ms_solve
            guards += int(shift > 0.0)
        except Exception:
            fallbacks += 1
            ms_solve, (X, shift) = cuda_time_ms(
                lambda: dwh_step_tuned_fp32(
                    X=X,
                    S=S,
                    ell=step.ell_in,
                    jitter_rel=jitter_rel,
                    out_dtype=torch.float32,
                )
            )
            ms_solve_sum += ms_solve
            dwh_steps += 1
            guards += int(shift > 0.0)
            last_step_kind = "DWH_TUNED_FP32(fallback)"

    X_out = X.to(dtype=iter_dtype)
    ms_exact_verify, final_kO_exact = cuda_time_ms(
        lambda: exact_final_kappa_O(X_out, exact_verify_device)
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
