from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from fast_iroot.coeff_tuner import (
    plan_coupled_quadratic_affine_opt_schedule,
    plan_coupled_local_minimax_schedule,
    plan_coupled_quadratic_newton_schedule,
)
from fast_iroot import precond_spd
from fast_iroot.chebyshev import (
    apply_inverse_chebyshev_with_coeffs,
    select_inverse_proot_chebyshev_minimax_auto,
)

from .bench_common import median, time_ms_any, time_ms_repeat

MATRIX_SOLVE_METHODS: List[str] = [
    "PE-Quad-Inverse-Multiply",
    "PE-Quad-Coupled-Apply",
    "Chebyshev-Apply",
]


@dataclass
class SolvePreparedInput:
    A_norm: torch.Tensor
    B: torch.Tensor
    stats: object


@dataclass
class SolveBenchResult:
    ms: float
    ms_iter: float
    ms_precond: float
    rel_err: float
    mem_alloc_mb: float
    mem_reserved_mb: float
    cheb_degree_used: float
    pe_newton_steps_used: float
    pe_minimax_steps_used: float
    pe_affine_opt_steps_used: float


@torch.no_grad()
def prepare_solve_inputs(
    mats: List[torch.Tensor],
    device: torch.device,
    k: int,
    precond: str,
    precond_ruiz_iters: int,
    ridge_rel: float,
    l_target: float,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tuple[List[SolvePreparedInput], float]:
    prepared: List[SolvePreparedInput] = []
    ms_pre_list: List[float] = []

    for A in mats:
        t_pre, out = time_ms_any(
            lambda: precond_spd(
                A,
                mode=precond,
                ruiz_iters=precond_ruiz_iters,
                ridge_rel=ridge_rel,
                l_target=l_target,
            ),
            device,
        )
        A_norm, stats = out
        ms_pre_list.append(t_pre)

        n = A_norm.shape[-1]
        B = torch.randn(
            *A_norm.shape[:-2], n, k, device=device, dtype=dtype, generator=generator
        )

        prepared.append(SolvePreparedInput(A_norm=A_norm, B=B, stats=stats))

    return prepared, (median(ms_pre_list) if ms_pre_list else float("nan"))


def _build_solve_runner(
    method: str,
    pe_step_coeffs: List[Tuple[float, float, float]],
    cheb_degree: int,
    cheb_coeffs: Optional[Tuple[float, ...]],
    p_val: int,
    l_min: float,
    symmetrize_every: int,
    online_stop_tol: Optional[float],
    online_min_steps: int,
    uncoupled_fn: Callable[..., Tuple[torch.Tensor, object]],
    coupled_solve_fn: Callable[..., Tuple[torch.Tensor, object]],
    cheb_apply_fn: Callable[..., Tuple[torch.Tensor, object]],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if method == "PE-Quad-Inverse-Multiply":
        ws_unc = None

        def run(A_norm: torch.Tensor, B: torch.Tensor):
            nonlocal ws_unc
            Xn, ws_unc = uncoupled_fn(
                A_norm,
                abc_t=pe_step_coeffs,
                p_val=p_val,
                ws=ws_unc,
                symmetrize_X=True,
            )
            return Xn @ B

        return run

    if method == "PE-Quad-Coupled-Apply":
        ws_cpl = None

        def run(A_norm: torch.Tensor, B: torch.Tensor):
            nonlocal ws_cpl
            Zn, ws_cpl = coupled_solve_fn(
                A_norm,
                B,
                abc_t=pe_step_coeffs,
                p_val=p_val,
                ws=ws_cpl,
                symmetrize_Y=True,
                symmetrize_every=symmetrize_every,
                terminal_last_step=True,
                online_stop_tol=online_stop_tol,
                online_min_steps=online_min_steps,
            )
            return Zn

        return run

    if method == "Chebyshev-Apply":
        ws_cheb = None

        if cheb_coeffs is not None:

            def run(A_norm: torch.Tensor, B: torch.Tensor):
                nonlocal ws_cheb
                Zn, ws_cheb = apply_inverse_chebyshev_with_coeffs(
                    A_norm,
                    B,
                    c_list=cheb_coeffs,
                    degree=cheb_degree,
                    l_min=l_min,
                    l_max=1.0,
                    ws=ws_cheb,
                )
                return Zn

            return run

        def run(A_norm: torch.Tensor, B: torch.Tensor):
            nonlocal ws_cheb
            Zn, ws_cheb = cheb_apply_fn(
                A_norm,
                B,
                p_val=p_val,
                degree=cheb_degree,
                l_min=l_min,
                l_max=1.0,
                ws=ws_cheb,
            )
            return Zn

        return run

    raise ValueError(f"unknown method: {method}")


@torch.no_grad()
def eval_solve_method(
    prepared_inputs: List[SolvePreparedInput],
    ms_precond_median: float,
    ground_truth_Z: List[torch.Tensor],
    device: torch.device,
    method: str,
    pe_quad_coeffs: List[Tuple[float, float, float]],
    cheb_degree: int,
    cheb_mode: str,
    cheb_candidate_degrees: Tuple[int, ...],
    cheb_error_grid_n: int,
    cheb_max_relerr_mult: float,
    timing_reps: int,
    p_val: int,
    l_min: float,
    symmetrize_every: int,
    online_stop_tol: Optional[float],
    online_min_steps: int,
    online_coeff_mode: str,
    online_coeff_min_rel_improve: float,
    online_coeff_min_ns_logwidth_rel_improve: float,
    uncoupled_fn: Callable[..., Tuple[torch.Tensor, object]],
    coupled_solve_fn: Callable[..., Tuple[torch.Tensor, object]],
    cheb_apply_fn: Callable[..., Tuple[torch.Tensor, object]],
) -> SolveBenchResult:
    ms_iter_list: List[float] = []
    err_list: List[float] = []
    mem_alloc_list: List[float] = []
    mem_res_list: List[float] = []
    cheb_degree_used_list: List[float] = []
    pe_newton_steps_list: List[float] = []
    pe_minimax_steps_list: List[float] = []
    pe_affine_opt_steps_list: List[float] = []

    if len(prepared_inputs) == 0:
        return SolveBenchResult(
            ms=float("nan"),
            ms_iter=float("nan"),
            ms_precond=float("nan"),
            rel_err=float("nan"),
            mem_alloc_mb=float("nan"),
            mem_reserved_mb=float("nan"),
            cheb_degree_used=float("nan"),
            pe_newton_steps_used=float("nan"),
            pe_minimax_steps_used=float("nan"),
            pe_affine_opt_steps_used=float("nan"),
        )

    for i, prep in enumerate(prepared_inputs):
        A_norm = prep.A_norm
        B = prep.B
        Z_true = ground_truth_Z[i]

        l_min_eff = float(l_min)
        if method == "Chebyshev-Apply" and hasattr(prep.stats, "gersh_lo"):
            try:
                l_min_eff = max(l_min_eff, float(prep.stats.gersh_lo))
            except Exception:
                pass

        cheb_degree_eff = int(cheb_degree)
        cheb_coeffs_eff: Optional[Tuple[float, ...]] = None
        pe_step_coeffs_eff = list(pe_quad_coeffs)
        pe_newton_steps_eff = 0.0
        pe_minimax_steps_eff = 0.0
        pe_affine_opt_steps_eff = 0.0
        if method == "Chebyshev-Apply":
            if cheb_mode == "fixed":
                cheb_degree_used_list.append(float(cheb_degree_eff))
            elif cheb_mode == "minimax-auto":
                deg_sel, coeff_sel, _, _, mode_used = (
                    select_inverse_proot_chebyshev_minimax_auto(
                        p_val=p_val,
                        baseline_degree=int(cheb_degree),
                        l_min=l_min_eff,
                        l_max=1.0,
                        candidate_degrees=cheb_candidate_degrees,
                        error_grid_n=int(cheb_error_grid_n),
                        max_relerr_mult=float(cheb_max_relerr_mult),
                    )
                )
                cheb_degree_eff = int(deg_sel)
                cheb_degree_used_list.append(float(cheb_degree_eff))
                if mode_used == "minimax-auto":
                    cheb_coeffs_eff = coeff_sel
            else:
                raise ValueError(
                    f"Unknown cheb_mode: '{cheb_mode}'. Supported modes are 'fixed', 'minimax-auto'."
                )
        elif method == "PE-Quad-Coupled-Apply" and online_coeff_mode != "off":
            lo_hint = float(l_min)
            if hasattr(prep.stats, "gersh_lo"):
                try:
                    lo_hint = max(lo_hint, float(prep.stats.gersh_lo))
                except Exception:
                    pass
            if online_coeff_mode == "greedy-newton":
                pe_step_coeffs_eff, sched_meta = plan_coupled_quadratic_newton_schedule(
                    pe_step_coeffs_eff,
                    p_val=p_val,
                    lo_init=lo_hint,
                    hi_init=1.0,
                    min_rel_improve=float(online_coeff_min_rel_improve),
                    terminal_last_step=True,
                )
            elif online_coeff_mode == "greedy-minimax":
                pe_step_coeffs_eff, sched_meta = plan_coupled_local_minimax_schedule(
                    pe_step_coeffs_eff,
                    p_val=p_val,
                    lo_init=lo_hint,
                    hi_init=1.0,
                    min_rel_improve=float(online_coeff_min_rel_improve),
                    min_ns_logwidth_rel_improve=float(
                        online_coeff_min_ns_logwidth_rel_improve
                    ),
                    terminal_last_step=True,
                )
            elif online_coeff_mode == "greedy-affine-opt":
                pe_step_coeffs_eff, sched_meta = (
                    plan_coupled_quadratic_affine_opt_schedule(
                        pe_step_coeffs_eff,
                        p_val=p_val,
                        lo_init=lo_hint,
                        hi_init=1.0,
                        min_rel_improve=float(online_coeff_min_rel_improve),
                        terminal_last_step=True,
                    )
                )
            else:
                raise ValueError(
                    "Unknown online_coeff_mode: "
                    f"'{online_coeff_mode}'. Supported modes are "
                    "'off', 'greedy-newton', 'greedy-minimax', "
                    "'greedy-affine-opt'."
                )
            pe_newton_steps_eff = float(sched_meta.get("newton_steps", 0.0))
            pe_minimax_steps_eff = float(sched_meta.get("minimax_steps", 0.0))
            pe_affine_opt_steps_eff = float(sched_meta.get("affine_opt_steps", 0.0))

        runner = _build_solve_runner(
            method=method,
            pe_step_coeffs=pe_step_coeffs_eff,
            cheb_degree=cheb_degree_eff,
            cheb_coeffs=cheb_coeffs_eff,
            p_val=p_val,
            l_min=l_min_eff,
            symmetrize_every=symmetrize_every,
            online_stop_tol=online_stop_tol,
            online_min_steps=online_min_steps,
            uncoupled_fn=uncoupled_fn,
            coupled_solve_fn=coupled_solve_fn,
            cheb_apply_fn=cheb_apply_fn,
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
            _ = runner(A_norm, B)
            mem_alloc_list.append(
                torch.cuda.max_memory_allocated(device=device) / (1024**2)
            )
            mem_res_list.append(
                torch.cuda.max_memory_reserved(device=device) / (1024**2)
            )

        def run_once() -> torch.Tensor:
            if device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            return runner(A_norm, B)

        ms_iter, Zn = time_ms_repeat(run_once, device, reps=timing_reps)
        ms_iter_list.append(ms_iter)

        if torch.isfinite(Zn).all() and torch.isfinite(Z_true).all():
            err_list.append(
                float(
                    torch.linalg.matrix_norm(Zn - Z_true)
                    / torch.linalg.matrix_norm(Z_true)
                )
            )
        else:
            err_list.append(float("inf"))

        if method == "PE-Quad-Coupled-Apply":
            pe_newton_steps_list.append(pe_newton_steps_eff)
            pe_minimax_steps_list.append(pe_minimax_steps_eff)
            pe_affine_opt_steps_list.append(pe_affine_opt_steps_eff)

    ms_iter_med = median(ms_iter_list)
    ms_pre_med = ms_precond_median

    return SolveBenchResult(
        ms=ms_pre_med + ms_iter_med,
        ms_iter=ms_iter_med,
        ms_precond=ms_pre_med,
        rel_err=median(err_list),
        mem_alloc_mb=median(mem_alloc_list) if mem_alloc_list else float("nan"),
        mem_reserved_mb=median(mem_res_list) if mem_res_list else float("nan"),
        cheb_degree_used=(
            median(cheb_degree_used_list) if cheb_degree_used_list else float("nan")
        ),
        pe_newton_steps_used=(
            median(pe_newton_steps_list) if pe_newton_steps_list else float("nan")
        ),
        pe_minimax_steps_used=(
            median(pe_minimax_steps_list) if pe_minimax_steps_list else float("nan")
        ),
        pe_affine_opt_steps_used=(
            median(pe_affine_opt_steps_list)
            if pe_affine_opt_steps_list
            else float("nan")
        ),
    )


def compute_ground_truth(
    prepared: List[SolvePreparedInput], p_val: int
) -> List[torch.Tensor]:
    Z_true = []
    for prep in prepared:
        A = prep.A_norm.cpu().double()
        B = prep.B.cpu().double()
        L, Q = torch.linalg.eigh(A)
        L = L.clamp_min(1e-12)
        L_inv = torch.pow(L, -1.0 / p_val)
        A_inv = (Q * L_inv.unsqueeze(0)) @ Q.mT
        Z_true.append(
            (A_inv @ B).to(dtype=prep.A_norm.dtype, device=prep.A_norm.device)
        )
    return Z_true
