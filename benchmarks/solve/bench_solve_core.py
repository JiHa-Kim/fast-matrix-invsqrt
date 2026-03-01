from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from fast_iroot.coeff_tuner import (
    plan_coupled_quadratic_affine_opt_schedule,
    plan_coupled_local_minimax_schedule,
    plan_coupled_quadratic_newton_schedule,
    truncate_coupled_schedule_by_interval_error,
)
from fast_iroot.precond import precond_spd
from fast_iroot.chebyshev import (
    apply_inverse_chebyshev_with_coeffs,
    select_inverse_proot_chebyshev_minimax_auto,
)

from benchmarks.common import median, pctl, time_ms_any, time_ms_repeat

BASE_MATRIX_SOLVE_METHODS: List[str] = [
    "PE-Quad-Coupled-Apply",
    "Inverse-Newton-Coupled-Apply",
]
P_GT1_SPD_EXTRA_METHODS: List[str] = [
    "Chebyshev-Apply",
    "Torch-Linalg-Solve",
    "Torch-EVD-Solve",
]
P1_SPD_SOLVE_BASELINES: List[str] = [
    "Torch-Linalg-Solve",
    "Torch-Cholesky-Solve",
    "Torch-EVD-Solve",
]
P1_SPD_SOLVE_EXTRA_CASES: List[str] = ["Torch-Cholesky-Solve-ReuseFactor"]


def _can_use_cuda_graph_for_method(
    method: str,
    *,
    use_cuda_graph: bool,
    device: torch.device,
    online_stop_tol: Optional[float],
    cheb_cuda_graph: bool,
) -> bool:
    if device.type != "cuda":
        return False
    if method == "Chebyshev-Apply":
        return bool(cheb_cuda_graph)
    if not bool(use_cuda_graph):
        return False
    if method == "PE-Quad-Coupled-Apply":
        return online_stop_tol is None
    return False


def matrix_solve_methods(p_val: int) -> List[str]:
    methods = list(BASE_MATRIX_SOLVE_METHODS)
    if int(p_val) == 1:
        methods.extend(P1_SPD_SOLVE_BASELINES)
        methods.extend(P1_SPD_SOLVE_EXTRA_CASES)
    else:
        methods.extend(P_GT1_SPD_EXTRA_METHODS)
    # Preserve declared order but avoid duplicate benchmarks.
    return list(dict.fromkeys(methods))


def _effective_cheb_fixed_degree(
    base_degree: int,
    *,
    cheb_mode: str,
    n: int,
    k: int,
    cheb_degree_klt: int,
) -> int:
    """Return the fixed-mode Chebyshev degree after k<n cap policy."""
    degree = int(base_degree)
    mode = str(cheb_mode).strip().lower()
    cap = int(cheb_degree_klt)
    if mode == "fixed" and cap >= 0 and int(k) < int(n):
        degree = min(degree, cap)
    return degree


def _naive_newton_preprocess(
    A_norm: torch.Tensor,
    *,
    p_val: int,
) -> Tuple[torch.Tensor, float]:
    """Vanilla Newton-Schulz reference scaling.

    Scale by ||A||_F so lambda_max(A_scaled) <= 1 for SPD matrices.
    Returns (A_scaled, output_scale) where output_scale multiplies the final
    inverse-p-root apply result to map back to the original scaling.
    """
    fro = torch.linalg.matrix_norm(A_norm, ord="fro")
    if fro.ndim == 0:
        alpha = float(fro.item())
    else:
        alpha = float(torch.max(fro).item())
    if (not math.isfinite(alpha)) or alpha <= 0.0:
        alpha = 1.0
    A_scaled = A_norm / float(alpha)
    out_scale = float(alpha) ** (-1.0 / float(p_val))
    return A_scaled, out_scale


def _build_cuda_graph_replay(
    runner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    A_norm: torch.Tensor,
    B: torch.Tensor,
    *,
    warmup: int,
) -> Callable[[], torch.Tensor]:
    warmup_i = max(1, int(warmup))
    for _ in range(warmup_i):
        _ = runner(A_norm, B)
    torch.cuda.synchronize(device=A_norm.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = runner(A_norm, B)

    def replay() -> torch.Tensor:
        graph.replay()
        return out

    return replay


@dataclass
class SolvePreparedInput:
    A_norm: torch.Tensor
    B: torch.Tensor
    stats: object
    A_root_f64: Optional[torch.Tensor] = None  # Cached A^{1/p} in double
    B_f64: Optional[torch.Tensor] = None  # Cached B in double


@dataclass
class SolveBenchResult:
    ms: float
    ms_iter: float
    ms_precond: float
    rel_err: float
    rel_err_p90: float
    residual: float
    residual_p90: float
    nonfinite_rate: float
    quality_fail_rate: float
    failure_rate: float  # (nonfinite | quality_fail)
    mem_alloc_mb: float
    mem_reserved_mb: float
    cheb_degree_used: float
    pe_newton_steps_used: float
    pe_minimax_steps_used: float
    pe_affine_opt_steps_used: float
    pe_steps_used: float


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
    p_val: int,
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
                compute_rho_proxy=False,
            ),
            device,
        )
        A_norm, stats = out
        ms_pre_list.append(t_pre)

        n = A_norm.shape[-1]
        B = torch.randn(
            *A_norm.shape[:-2], n, k, device=device, dtype=dtype, generator=generator
        )

        A_f64 = A_norm.detach().cpu().double()
        B_f64 = B.detach().cpu().double()

        if p_val == 1:
            A_root_f64 = A_f64
        else:
            # Precompute A^{1/p} for faster residual checks in eval
            L, Q = torch.linalg.eigh(A_f64)
            L_root = torch.pow(L.clamp_min(1e-12), 1.0 / p_val)
            A_root_f64 = (Q * L_root.unsqueeze(0)) @ Q.mT

        prepared.append(
            SolvePreparedInput(
                A_norm=A_norm,
                B=B,
                stats=stats,
                A_root_f64=A_root_f64,
                B_f64=B_f64,
            )
        )

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
    terminal_tail_steps: int,
    online_min_steps: int,
    online_stop_metric: str,
    online_stop_check_every: int,
    post_correction_steps: int,
    post_correction_order: int,
    coupled_solve_fn: Callable[..., Tuple[torch.Tensor, object]],
    cheb_apply_fn: Callable[..., Tuple[torch.Tensor, object]],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    inv_newton_step = ((p_val + 1.0) / p_val, -1.0 / p_val, 0.0)
    inv_newton_coeffs = [inv_newton_step] * len(pe_step_coeffs)

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
                terminal_tail_steps=terminal_tail_steps,
                online_stop_tol=online_stop_tol,
                online_min_steps=online_min_steps,
                online_stop_metric=online_stop_metric,
                online_stop_check_every=online_stop_check_every,
                post_correction_steps=post_correction_steps,
                post_correction_order=post_correction_order,
            )
            return Zn

        return run

    if method == "Inverse-Newton-Coupled-Apply":
        ws_cpl = None

        def run(A_norm: torch.Tensor, B: torch.Tensor):
            nonlocal ws_cpl
            A_ref, out_scale = _naive_newton_preprocess(A_norm, p_val=p_val)
            Zn, ws_cpl = coupled_solve_fn(
                A_ref,
                B,
                abc_t=inv_newton_coeffs,
                p_val=p_val,
                ws=ws_cpl,
                symmetrize_Y=False,
                symmetrize_every=1,
                terminal_last_step=False,
                terminal_tail_steps=0,
                online_stop_tol=None,
                online_min_steps=1,
                online_stop_metric="diag",
                online_stop_check_every=1,
                post_correction_steps=0,
                post_correction_order=2,
            )
            if out_scale != 1.0:
                Zn = Zn * out_scale
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

    if method == "Torch-Linalg-Solve":

        def run(A_norm: torch.Tensor, B: torch.Tensor):
            if int(p_val) == 1:
                A_f32 = A_norm.to(torch.float32)
                B_f32 = B.to(torch.float32)
                Z, _ = torch.linalg.solve_ex(A_f32, B_f32)
                return Z  # Return in fp32 for low origin error
            else:
                # For p > 1, we must use EVD or similar to compute A^{-1/p}
                A_f32 = A_norm.to(torch.float32)
                B_f32 = B.to(torch.float32)
                L, Q = torch.linalg.eigh(A_f32)
                L_inv = torch.pow(L.clamp_min(1e-12), -1.0 / p_val)
                Z = (Q * L_inv.unsqueeze(0)) @ (Q.mT @ B_f32)
                return Z

        return run

    if method == "Torch-Cholesky-Solve":
        if int(p_val) != 1:
            raise ValueError("Torch-Cholesky-Solve baseline is only valid for p=1")

        def run(A_norm: torch.Tensor, B: torch.Tensor):
            A_f32 = A_norm.to(torch.float32)
            B_f32 = B.to(torch.float32)
            L, info = torch.linalg.cholesky_ex(A_f32)
            Z = torch.cholesky_solve(B_f32, L)
            return Z  # Return in fp32

        return run

    if method == "Torch-Cholesky-Solve-ReuseFactor":
        if int(p_val) != 1:
            raise ValueError(
                "Torch-Cholesky-Solve-ReuseFactor baseline is only valid for p=1"
            )

        A_f32_cached: Optional[torch.Tensor] = None
        L_cached: Optional[torch.Tensor] = None

        def run(A_norm: torch.Tensor, B: torch.Tensor):
            nonlocal A_f32_cached, L_cached
            if A_f32_cached is None or L_cached is None:
                A_f32_cached = A_norm.to(torch.float32)
                L_cached, info = torch.linalg.cholesky_ex(A_f32_cached)
            B_f32 = B.to(torch.float32)
            Z = torch.cholesky_solve(B_f32, L_cached)
            return Z  # Return in fp32

        return run

    if method == "Torch-EVD-Solve":

        def run(A_norm: torch.Tensor, B: torch.Tensor):
            # Compute A^{-1/p} B via EVD, casting to fp32 for CUDA compatibility
            A_f32 = A_norm.to(torch.float32)
            B_f32 = B.to(torch.float32)
            L, Q = torch.linalg.eigh(A_f32)
            L_inv = torch.pow(L.clamp_min(1e-12), -1.0 / p_val)
            Z = (Q * L_inv.unsqueeze(0)) @ (Q.mT @ B_f32)
            return Z  # Return in fp32

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
    cheb_degree_klt: int,
    cheb_mode: str,
    cheb_candidate_degrees: Tuple[int, ...],
    cheb_error_grid_n: int,
    cheb_max_relerr_mult: float,
    timing_reps: int,
    timing_warmup_reps: int,
    p_val: int,
    l_min: float,
    symmetrize_every: int,
    online_stop_tol: Optional[float],
    terminal_tail_steps: int,
    online_min_steps: int,
    online_stop_metric: str,
    online_stop_check_every: int,
    post_correction_steps: int,
    post_correction_order: int,
    online_coeff_mode: str,
    online_coeff_cost_model: str,
    online_coeff_min_rel_improve: float,
    online_coeff_min_ns_logwidth_rel_improve: float,
    online_coeff_target_interval_err: float,
    online_coeff_min_steps: int,
    use_cuda_graph: bool,
    cuda_graph_warmup: int,
    cheb_cuda_graph: bool,
    coupled_solve_fn: Callable[..., Tuple[torch.Tensor, object]],
    cheb_apply_fn: Callable[..., Tuple[torch.Tensor, object]],
) -> SolveBenchResult:
    ms_iter_list: List[float] = []
    err_list: List[float] = []
    resid_list: List[float] = []
    mem_alloc_list: List[float] = []
    mem_res_list: List[float] = []
    cheb_degree_used_list: List[float] = []
    pe_newton_steps_list: List[float] = []
    pe_minimax_steps_list: List[float] = []
    pe_affine_opt_steps_list: List[float] = []
    pe_steps_used_list: List[float] = []
    nonfinite_count = 0
    quality_fail_count = 0

    # Failure thresholds: if relerr or residual exceeds these, treat as failure
    # for accounting purposes.
    RELERR_MAX_FAIL = 1e-1
    RESID_MAX_FAIL = 1e-1

    if len(prepared_inputs) == 0:
        return SolveBenchResult(
            ms=float("nan"),
            ms_iter=float("nan"),
            ms_precond=float("nan"),
            rel_err=float("nan"),
            rel_err_p90=float("nan"),
            residual=float("nan"),
            residual_p90=float("nan"),
            nonfinite_rate=float("nan"),
            quality_fail_rate=float("nan"),
            failure_rate=float("nan"),
            quality_per_ms=float("nan"),
            mem_alloc_mb=float("nan"),
            mem_reserved_mb=float("nan"),
            cheb_degree_used=float("nan"),
            pe_newton_steps_used=float("nan"),
            pe_minimax_steps_used=float("nan"),
            pe_affine_opt_steps_used=float("nan"),
            pe_steps_used=float("nan"),
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

        cheb_degree_eff = _effective_cheb_fixed_degree(
            int(cheb_degree),
            cheb_mode=cheb_mode,
            n=int(A_norm.shape[-1]),
            k=int(B.shape[-1]),
            cheb_degree_klt=int(cheb_degree_klt),
        )
        cheb_coeffs_eff: Optional[Tuple[float, ...]] = None
        pe_step_coeffs_eff = list(pe_quad_coeffs)
        pe_newton_steps_eff = 0.0
        pe_minimax_steps_eff = 0.0
        pe_affine_opt_steps_eff = 0.0
        pe_steps_used_eff = float(len(pe_step_coeffs_eff))
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
            coeff_cost_mode = str(online_coeff_cost_model).strip().lower()
            if coeff_cost_mode == "shape-aware":
                rhs_ratio = float(B.shape[-1]) / float(A_norm.shape[-1])
                use_terminal_rhs_direct = bool(B.shape[-1] < A_norm.shape[-1])
            elif coeff_cost_mode == "gemm":
                rhs_ratio = 1.0
                use_terminal_rhs_direct = False
            else:
                raise ValueError(
                    "Unknown online_coeff_cost_model: "
                    f"'{online_coeff_cost_model}'. Supported modes are "
                    "'gemm', 'shape-aware'."
                )
            if online_coeff_mode == "greedy-newton":
                pe_step_coeffs_eff, sched_meta = plan_coupled_quadratic_newton_schedule(
                    pe_step_coeffs_eff,
                    p_val=p_val,
                    lo_init=lo_hint,
                    hi_init=1.0,
                    min_rel_improve=float(online_coeff_min_rel_improve),
                    terminal_last_step=True,
                    rhs_to_n_ratio=rhs_ratio,
                    terminal_rhs_direct=use_terminal_rhs_direct,
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
                    rhs_to_n_ratio=rhs_ratio,
                    terminal_rhs_direct=use_terminal_rhs_direct,
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
                        rhs_to_n_ratio=rhs_ratio,
                        terminal_rhs_direct=use_terminal_rhs_direct,
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

        if (
            method == "PE-Quad-Coupled-Apply"
            and float(online_coeff_target_interval_err) > 0.0
        ):
            lo_hint = float(l_min)
            if hasattr(prep.stats, "gersh_lo"):
                try:
                    lo_hint = max(lo_hint, float(prep.stats.gersh_lo))
                except Exception:
                    pass
            pe_step_coeffs_eff, trim_meta = truncate_coupled_schedule_by_interval_error(
                pe_step_coeffs_eff,
                p_val=p_val,
                lo_init=lo_hint,
                hi_init=1.0,
                target_err=float(online_coeff_target_interval_err),
                min_steps=int(online_coeff_min_steps),
            )
            pe_steps_used_eff = float(
                trim_meta.get("steps_used", len(pe_step_coeffs_eff))
            )
        elif method == "PE-Quad-Coupled-Apply":
            pe_steps_used_eff = float(len(pe_step_coeffs_eff))

        if method == "PE-Quad-Coupled-Apply":
            # Recompute per-step usage counters after optional trimming.
            ns_step = ((p_val + 1.0) / p_val, -1.0 / p_val, 0.0)

            def _same_step(
                lhs: Tuple[float, float, float],
                rhs: Tuple[float, float, float],
                *,
                tol: float = 1e-10,
            ) -> bool:
                return (
                    abs(float(lhs[0]) - float(rhs[0])) <= tol
                    and abs(float(lhs[1]) - float(rhs[1])) <= tol
                    and abs(float(lhs[2]) - float(rhs[2])) <= tol
                )

            n_newton = 0
            n_minimax = 0
            n_affine_opt = 0
            for j, step in enumerate(pe_step_coeffs_eff):
                if j < len(pe_quad_coeffs) and _same_step(step, pe_quad_coeffs[j]):
                    continue
                if _same_step(step, ns_step):
                    n_newton += 1
                elif online_coeff_mode == "greedy-minimax":
                    n_minimax += 1
                elif online_coeff_mode == "greedy-affine-opt":
                    n_affine_opt += 1

            pe_newton_steps_eff = float(n_newton)
            pe_minimax_steps_eff = float(n_minimax)
            pe_affine_opt_steps_eff = float(n_affine_opt)

        runner = _build_solve_runner(
            method=method,
            pe_step_coeffs=pe_step_coeffs_eff,
            cheb_degree=cheb_degree_eff,
            cheb_coeffs=cheb_coeffs_eff,
            p_val=p_val,
            l_min=l_min_eff,
            symmetrize_every=symmetrize_every,
            online_stop_tol=online_stop_tol,
            terminal_tail_steps=terminal_tail_steps,
            online_min_steps=online_min_steps,
            online_stop_metric=online_stop_metric,
            online_stop_check_every=online_stop_check_every,
            post_correction_steps=post_correction_steps,
            post_correction_order=post_correction_order,
            coupled_solve_fn=coupled_solve_fn,
            cheb_apply_fn=cheb_apply_fn,
        )

        if device.type == "cuda":
            # Only measure peak memory on the first trial to reduce overhead
            if i == 0:
                torch.cuda.reset_peak_memory_stats(device=device)
                _ = runner(A_norm, B)
                mem_alloc_list.append(
                    torch.cuda.max_memory_allocated(device=device) / (1024**2)
                )
                mem_res_list.append(
                    torch.cuda.max_memory_reserved(device=device) / (1024**2)
                )

        graph_active = False

        def timed_call() -> torch.Tensor:
            return runner(A_norm, B)

        if _can_use_cuda_graph_for_method(
            method,
            use_cuda_graph=bool(use_cuda_graph),
            device=device,
            online_stop_tol=online_stop_tol,
            cheb_cuda_graph=bool(cheb_cuda_graph),
        ):
            try:
                timed_call = _build_cuda_graph_replay(
                    runner, A_norm, B, warmup=int(cuda_graph_warmup)
                )
                graph_active = True
            except Exception:

                def timed_call() -> torch.Tensor:
                    return runner(A_norm, B)

        def run_once() -> torch.Tensor:
            if device.type == "cuda" and (not graph_active):
                torch.compiler.cudagraph_mark_step_begin()
            return timed_call()

        warmup_reps_i = max(0, int(timing_warmup_reps))
        for _ in range(warmup_reps_i):
            _ = run_once()
        if device.type == "cuda" and warmup_reps_i > 0:
            torch.cuda.synchronize(device=device)

        ms_iter, Zn = time_ms_repeat(run_once, device, reps=timing_reps)
        ms_iter_list.append(ms_iter)

        if torch.isfinite(Zn).all() and torch.isfinite(Z_true).all():
            # Compute relative error in double precision
            Zn_f64 = Zn.detach().cpu().double()
            Zt_f64 = Z_true.detach().cpu().double()

            norm_zt = torch.linalg.matrix_norm(Zt_f64).clamp_min(1e-12)
            rel_err = float(torch.linalg.matrix_norm(Zn_f64 - Zt_f64) / norm_zt)

            # Optimized residual calculation using cached A^{1/p} and B
            assert prep.A_root_f64 is not None
            assert prep.B_f64 is not None
            norm_b = torch.linalg.matrix_norm(prep.B_f64).clamp_min(1e-12)
            resid = float(
                torch.linalg.matrix_norm(prep.A_root_f64 @ Zn_f64 - prep.B_f64) / norm_b
            )

            err_list.append(rel_err)
            resid_list.append(resid)

            # Failure accounting: finite but garbage result
            if rel_err > RELERR_MAX_FAIL or resid > RESID_MAX_FAIL:
                quality_fail_count += 1
        else:
            err_list.append(float("inf"))
            resid_list.append(float("inf"))
            nonfinite_count += 1

        if method == "PE-Quad-Coupled-Apply":
            pe_newton_steps_list.append(pe_newton_steps_eff)
            pe_minimax_steps_list.append(pe_minimax_steps_eff)
            pe_affine_opt_steps_list.append(pe_affine_opt_steps_eff)
            pe_steps_used_list.append(pe_steps_used_eff)

    ms_iter_med = median(ms_iter_list)
    ms_pre_med = ms_precond_median
    rel_err_med = median(err_list)
    rel_err_p90 = pctl(err_list, 0.90)
    resid_med = median(resid_list)
    resid_p90 = pctl(resid_list, 0.90)
    total_count = len(prepared_inputs)
    nf_rate = (
        float(nonfinite_count) / float(total_count) if total_count > 0 else float("nan")
    )
    qf_rate = (
        float(quality_fail_count) / float(total_count)
        if total_count > 0
        else float("nan")
    )
    failure_rate = (
        float(nonfinite_count + quality_fail_count) / float(total_count)
        if total_count > 0
        else float("nan")
    )

    return SolveBenchResult(
        ms=ms_pre_med + ms_iter_med,
        ms_iter=ms_iter_med,
        ms_precond=ms_pre_med,
        rel_err=rel_err_med,
        rel_err_p90=rel_err_p90,
        residual=resid_med,
        residual_p90=resid_p90,
        nonfinite_rate=nf_rate,
        quality_fail_rate=qf_rate,
        failure_rate=failure_rate,
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
        pe_steps_used=(
            median(pe_steps_used_list) if pe_steps_used_list else float("nan")
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
        # Keep in double for error calculation
        Z_true.append((A_inv @ B).cpu())
    return Z_true
