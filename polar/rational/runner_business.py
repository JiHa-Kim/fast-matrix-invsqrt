from __future__ import annotations
from typing import Sequence

import torch

from polar.ops import cuda_time_ms
from polar.rational.ops import (
    gram_xtx_fast,
    apply_right_fast,
    exact_final_kappa_O_fast,
)
from polar.rational.dwh import dwh_ell_next
from polar.rational.dwh_stable_solve import dwh_step_matrix_only_stable_solve
from polar.polynomial.express import (
    paper_polar_express_coeff,
    PaperPolarExpressStep,
    polar_express_paper5_step_matrix_only,
    polar_express_paper_fro_scale,
)
from polar.schedules import StepSpec
from polar.runner import RunSummary

Tensor = torch.Tensor

@torch.no_grad()
def run_one_case_business(
    G_storage: Tensor,
    target_kappa_O: float,
    schedule: Sequence[StepSpec],
    iter_dtype: torch.dtype,
    jitter_rel: float,
    tf32: bool,
    exact_verify_device: str,
) -> RunSummary:
    """
    ULTIMATE WALL-CLOCK PERFORMANCE path.
    Designed for peak occupancy and minimal passes.
    """
    device = G_storage.device
    if device.type == "cuda":
        # Force TF32 for the duration of this run
        orig_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        X = G_storage.to(dtype=torch.float32)
        ms_gram_sum = 0.0
        ms_solve_sum = 0.0
        ms_upd_sum = 0.0
        dwh_steps = 0
        zolo_steps = 0

        # --- PHASE 1: FP32 DWH PRE-CONDITIONING ---
        # We target 2 steps to get sigma_min to ~0.64.
        # ell0 = 1e-6 -> ell1 = 0.025 -> ell2 = 0.64 (theory)
        ell = 1.0 / (float(G_storage.shape[1]) * 1000.0) 
        for _ in range(2):
            ms_gram, S = cuda_time_ms(lambda: gram_xtx_fast(X, torch.float32))
            ms_gram_sum += ms_gram
            
            ms_solve, (Q_step, _) = cuda_time_ms(
                lambda: dwh_step_matrix_only_stable_solve(S, ell, jitter_rel)
            )
            ms_solve_sum += ms_solve
            dwh_steps += 1
            
            ms_upd, X = cuda_time_ms(lambda: apply_right_fast(X, Q_step, torch.float32))
            ms_upd_sum += ms_upd
            
            ell = dwh_ell_next(ell)

        # --- STEP 2+: SPAM BF16 POLAR EXPRESS ---
        # Switch to BF16 for peak Tensor Core throughput.
        X_bf16 = X.to(dtype=torch.bfloat16)
        
        # Fro-scale to protect BF16 range.
        ms_upd, (X_bf16, _) = cuda_time_ms(lambda: polar_express_paper_fro_scale(X_bf16))
        ms_upd_sum += ms_upd

        for step_idx in range(5):
            ms_gram, S_bf16 = cuda_time_ms(lambda: gram_xtx_fast(X_bf16, torch.bfloat16))
            ms_gram_sum += ms_gram

            coeffs = paper_polar_express_coeff(step_idx)
            ms_solve, (Q_step, _) = cuda_time_ms(
                lambda: polar_express_paper5_step_matrix_only(
                    S_bf16,
                    PaperPolarExpressStep(coeffs.a, coeffs.b, coeffs.c),
                    torch.bfloat16,
                )
            )
            ms_solve_sum += ms_solve
            zolo_steps += 1
            ms_upd, X_bf16 = cuda_time_ms(lambda: apply_right_fast(X_bf16, Q_step, torch.bfloat16))
            ms_upd_sum += ms_upd

        X = X_bf16.to(dtype=iter_dtype)

        # Verification
        ms_exact_verify, final_kO_exact = cuda_time_ms(
            lambda: exact_final_kappa_O_fast(X)
        )
        
        ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum
        return RunSummary(
            success=bool(final_kO_exact <= target_kappa_O),
            final_kO_exact=float(final_kO_exact),
            steps=dwh_steps + zolo_steps,
            dwh_steps=dwh_steps,
            zolo_steps=zolo_steps,
            guards=0,
            fallbacks=0,
            last_step_kind="BUSINESS_HYBRID",
            ms_gram=ms_gram_sum,
            ms_solve=ms_solve_sum,
            ms_upd=ms_upd_sum,
            ms_total_timed=ms_total,
            ms_exact_verify=ms_exact_verify,
        )
    finally:
        if device.type == "cuda":
            torch.set_float32_matmul_precision(orig_precision)
