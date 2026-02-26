"""
matrix_solve_nonspd.py

Dedicated non-SPD benchmark suite for inverse solve (p=1 only):
Z = A^{-1} B.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

try:
    from benchmarks._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from fast_iroot import (
    _quad_coeffs,
    build_pe_schedules,
    inverse_proot_pe_quadratic_uncoupled,
    inverse_solve_pe_quadratic_coupled,
    apply_inverse_root_auto,
)
from fast_iroot.nonspd import NONSPD_PRECOND_MODES, precond_nonspd
from benchmarks.common import (
    make_nonspd_cases,
    median,
    parse_shapes,
    time_ms_any,
    time_ms_repeat,
    maybe_compile,
)

METHODS: List[str] = [
    "PE-Quad-Inverse-Multiply",
    "Inverse-Newton-Inverse-Multiply",
    "PE-Quad-Coupled-Apply",
    "Inverse-Newton-Coupled-Apply",
    "PE-Quad-Coupled-Apply-Safe",
    "PE-Quad-Coupled-Apply-Adaptive",
    "Torch-Solve",
    "Auto-Switch-Production",
]


@dataclass
class NonSpdPreparedInput:
    A_norm: torch.Tensor
    B: torch.Tensor


@dataclass
class NonSpdBenchResult:
    ms: float
    ms_iter: float
    ms_precond: float
    rel_err: float
    bad: int
    mem_alloc_mb: float
    mem_reserved_mb: float


@torch.no_grad()
def prepare_nonspd_solve_inputs(
    mats: List[torch.Tensor],
    device: torch.device,
    k: int,
    dtype: torch.dtype,
    generator: torch.Generator,
    precond: str = "row-norm",
    precond_ruiz_iters: int = 2,
) -> Tuple[List[NonSpdPreparedInput], float]:
    prepared: List[NonSpdPreparedInput] = []
    ms_pre_list: List[float] = []

    for A in mats:
        t_pre, A_norm = time_ms_any(
            lambda: precond_nonspd(
                A,
                mode=precond,
                ruiz_iters=precond_ruiz_iters,
            ),
            device,
        )
        ms_pre_list.append(t_pre)

        n = A_norm.shape[-1]
        B = torch.randn(
            *A_norm.shape[:-2], n, k, device=device, dtype=dtype, generator=generator
        )
        prepared.append(NonSpdPreparedInput(A_norm=A_norm, B=B))

    return prepared, (median(ms_pre_list) if ms_pre_list else float("nan"))


@torch.no_grad()
def compute_ground_truth(prepared: List[NonSpdPreparedInput]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for prep in prepared:
        A = prep.A_norm.double()
        B = prep.B.double()
        Z = torch.linalg.solve(A, B)
        out.append(Z.to(dtype=prep.A_norm.dtype))
    return out


def _parse_case_csv(spec: str) -> List[str]:
    vals = [tok.strip() for tok in str(spec).split(",") if tok.strip()]
    if not vals:
        raise ValueError("Expected at least one case in --cases")
    return vals


def _build_runner(
    method: str,
    pe_coeffs: Sequence[Tuple[float, float, float]],
    nonspd_adaptive_resid_tol: float,
    nonspd_adaptive_growth_tol: float,
    nonspd_adaptive_check_every: int,
    nonspd_safe_fallback_tol: Optional[float],
    nonspd_safe_early_y_tol: Optional[float],
    uncoupled_fn: Callable[..., Tuple[torch.Tensor, object]],
    coupled_solve_fn: Callable[..., Tuple[torch.Tensor, object]],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    inv_newton_coeffs = [((1.0 + 1.0) / 1.0, -1.0 / 1.0, 0.0)] * len(pe_coeffs)

    if method == "PE-Quad-Inverse-Multiply":
        ws_unc: Optional[object] = None

        def run(A_norm: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            nonlocal ws_unc
            Xn, ws_unc = uncoupled_fn(
                A_norm,
                abc_t=pe_coeffs,
                p_val=1,
                ws=ws_unc,
                symmetrize_X=False,
                assume_spd=False,
            )
            return Xn @ B

        return run

    if method == "Inverse-Newton-Inverse-Multiply":
        ws_unc: Optional[object] = None

        def run(A_norm: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            nonlocal ws_unc
            Xn, ws_unc = uncoupled_fn(
                A_norm,
                abc_t=inv_newton_coeffs,
                p_val=1,
                ws=ws_unc,
                symmetrize_X=False,
                assume_spd=False,
            )
            return Xn @ B

        return run

    if method == "PE-Quad-Coupled-Apply":
        ws_cpl: Optional[object] = None

        def run(A_norm: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            nonlocal ws_cpl
            Zn, ws_cpl = coupled_solve_fn(
                A_norm,
                B,
                abc_t=pe_coeffs,
                p_val=1,
                ws=ws_cpl,
                symmetrize_Y=False,
                symmetrize_every=1,
                terminal_last_step=True,
                online_stop_tol=None,
                online_min_steps=1,
                assume_spd=False,
            )
            return Zn

        return run

    if method == "Inverse-Newton-Coupled-Apply":
        ws_cpl: Optional[object] = None

        def run(A_norm: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            nonlocal ws_cpl
            Zn, ws_cpl = coupled_solve_fn(
                A_norm,
                B,
                abc_t=inv_newton_coeffs,
                p_val=1,
                ws=ws_cpl,
                symmetrize_Y=False,
                symmetrize_every=1,
                terminal_last_step=True,
                online_stop_tol=None,
                online_min_steps=1,
                assume_spd=False,
            )
            return Zn

        return run

    if method == "PE-Quad-Coupled-Apply-Safe":
        ws_cpl_safe: Optional[object] = None

        def run(A_norm: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            nonlocal ws_cpl_safe
            Zn, ws_cpl_safe = coupled_solve_fn(
                A_norm,
                B,
                abc_t=pe_coeffs,
                p_val=1,
                ws=ws_cpl_safe,
                symmetrize_Y=False,
                symmetrize_every=1,
                terminal_last_step=True,
                online_stop_tol=None,
                online_min_steps=1,
                assume_spd=False,
                nonspd_adaptive=False,
                nonspd_safe_fallback_tol=nonspd_safe_fallback_tol,
                nonspd_safe_early_y_tol=nonspd_safe_early_y_tol,
            )
            return Zn

        return run

    if method == "PE-Quad-Coupled-Apply-Adaptive":
        ws_cpl_adapt: Optional[object] = None

        def run(A_norm: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            nonlocal ws_cpl_adapt
            Zn, ws_cpl_adapt = coupled_solve_fn(
                A_norm,
                B,
                abc_t=pe_coeffs,
                p_val=1,
                ws=ws_cpl_adapt,
                symmetrize_Y=False,
                symmetrize_every=1,
                terminal_last_step=True,
                online_stop_tol=None,
                online_min_steps=1,
                assume_spd=False,
                nonspd_adaptive=True,
                nonspd_adaptive_resid_tol=nonspd_adaptive_resid_tol,
                nonspd_adaptive_growth_tol=nonspd_adaptive_growth_tol,
                nonspd_adaptive_check_every=nonspd_adaptive_check_every,
                nonspd_safe_fallback_tol=nonspd_safe_fallback_tol,
                nonspd_safe_early_y_tol=nonspd_safe_early_y_tol,
            )
            return Zn

        return run

    if method == "Torch-Solve":

        def run(A_norm: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            A_f32 = A_norm.to(torch.float32)
            B_f32 = B.to(torch.float32)
            Z = torch.linalg.solve(A_f32, B_f32)
            return Z.to(A_norm.dtype)

        return run

    if method == "Auto-Switch-Production":

        def run(A_norm: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            # Matches our library's default auto logic
            Z, _ = apply_inverse_root_auto(
                A_norm,
                B,
                abc_t=pe_coeffs,
                p_val=1,
                k_threshold=0.1,
                nonspd_safe_fallback_tol=0.01,
                nonspd_safe_early_y_tol=0.8,
            )
            return Z

        return run

    raise ValueError(f"unknown method: {method}")


@torch.no_grad()
def eval_method(
    prepared_inputs: List[NonSpdPreparedInput],
    ground_truth_Z: List[torch.Tensor],
    device: torch.device,
    method: str,
    pe_coeffs: Sequence[Tuple[float, float, float]],
    timing_reps: int,
    timing_warmup_reps: int,
    ms_precond_median: float,
    nonspd_adaptive_resid_tol: float,
    nonspd_adaptive_growth_tol: float,
    nonspd_adaptive_check_every: int,
    nonspd_safe_fallback_tol: Optional[float],
    nonspd_safe_early_y_tol: Optional[float],
    uncoupled_fn: Callable[..., Tuple[torch.Tensor, object]],
    coupled_solve_fn: Callable[..., Tuple[torch.Tensor, object]],
) -> NonSpdBenchResult:
    ms_iter_list: List[float] = []
    relerr_list: List[float] = []
    mem_alloc_list: List[float] = []
    mem_res_list: List[float] = []
    bad = 0

    for i, prep in enumerate(prepared_inputs):
        A_norm = prep.A_norm
        B = prep.B
        Z_true = ground_truth_Z[i]
        runner = _build_runner(
            method,
            pe_coeffs,
            nonspd_adaptive_resid_tol,
            nonspd_adaptive_growth_tol,
            nonspd_adaptive_check_every,
            nonspd_safe_fallback_tol,
            nonspd_safe_early_y_tol,
            uncoupled_fn,
            coupled_solve_fn,
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

        warm = max(0, int(timing_warmup_reps))
        for _ in range(warm):
            _ = runner(A_norm, B)
        if device.type == "cuda" and warm > 0:
            torch.cuda.synchronize(device=device)

        def run_once() -> torch.Tensor:
            if device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            return runner(A_norm, B)

        ms_iter, Z_hat = time_ms_repeat(run_once, device, reps=timing_reps)
        ms_iter_list.append(ms_iter)

        if not torch.isfinite(Z_hat).all():
            bad += 1
            relerr_list.append(float("inf"))
            continue

        rel = torch.linalg.matrix_norm(Z_hat - Z_true) / torch.linalg.matrix_norm(
            Z_true
        )
        relerr_list.append(float(rel))

    return NonSpdBenchResult(
        ms=(ms_precond_median + median(ms_iter_list)),
        ms_iter=median(ms_iter_list),
        ms_precond=ms_precond_median,
        rel_err=median(relerr_list),
        bad=bad,
        mem_alloc_mb=median(mem_alloc_list) if mem_alloc_list else float("nan"),
        mem_reserved_mb=median(mem_res_list) if mem_res_list else float("nan"),
    )


def main():
    p = argparse.ArgumentParser(
        description="Dedicated non-SPD benchmark suite for solve p=1 only"
    )
    p.add_argument("--p", type=int, default=1, help="Must be 1 for this suite")
    p.add_argument("--sizes", type=str, default="1024")
    p.add_argument(
        "--k", type=str, default="1,16,64,1024", help="RHS column counts (CSV)"
    )
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--cases",
        type=str,
        default="gaussian_shifted,nonnormal_upper,similarity_posspec,similarity_posspec_hard",
        help="Comma-separated non-SPD cases",
    )
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    p.add_argument(
        "--precond",
        type=str,
        default="row-norm",
        choices=list(NONSPD_PRECOND_MODES),
        help="Non-SPD preconditioning/scaling mode",
    )
    p.add_argument(
        "--precond-ruiz-iters",
        type=int,
        default=2,
        help="Iteration count for --precond=ruiz",
    )
    p.add_argument("--timing-reps", type=int, default=5)
    p.add_argument("--timing-warmup-reps", type=int, default=2)
    p.add_argument(
        "--nonspd-adaptive-resid-tol",
        type=float,
        default=1.0,
        help="Adaptive fallback trigger on ||Y-I||_F/sqrt(n) for non-SPD p=1 coupled apply",
    )
    p.add_argument(
        "--nonspd-adaptive-growth-tol",
        type=float,
        default=1.02,
        help="Adaptive fallback trigger when residual grows by this factor between checks",
    )
    p.add_argument(
        "--nonspd-adaptive-check-every",
        type=int,
        default=1,
        help="Check interval (steps) for non-SPD adaptive fallback",
    )
    p.add_argument(
        "--nonspd-safe-fallback-tol",
        type=float,
        default=0.01,
        help=(
            "If > 0, adaptive method falls back to exact solve when final "
            "relative residual ||A Z - B||/||B|| exceeds this tolerance"
        ),
    )
    p.add_argument(
        "--nonspd-safe-early-y-tol",
        type=float,
        default=0.8,
        help=(
            "If > 0, trigger early exact-solve fallback after step 1 when "
            "max|diag(Y)-1| exceeds this threshold"
        ),
    )
    p.add_argument(
        "--coeff-mode",
        type=str,
        default="auto",
        choices=["auto", "precomputed", "tuned"],
    )
    p.add_argument("--coeff-seed", type=int, default=0)
    p.add_argument("--coeff-safety", type=float, default=1.0)
    p.add_argument("--coeff-no-final-safety", action="store_true")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()

    if int(args.p) != 1:
        raise ValueError(
            f"matrix_solve_nonspd.py is a p=1-only suite, got --p={args.p}"
        )
    ks = parse_shapes(args.k)
    if int(args.trials) < 1:
        raise ValueError(f"--trials must be >= 1, got {args.trials}")
    if int(args.timing_reps) < 1:
        raise ValueError(f"--timing-reps must be >= 1, got {args.timing_reps}")
    if int(args.timing_warmup_reps) < 0:
        raise ValueError(
            f"--timing-warmup-reps must be >= 0, got {args.timing_warmup_reps}"
        )
    if int(args.precond_ruiz_iters) < 1:
        raise ValueError(
            f"--precond-ruiz-iters must be >= 1, got {args.precond_ruiz_iters}"
        )
    if float(args.nonspd_adaptive_resid_tol) <= 0.0:
        raise ValueError(
            "--nonspd-adaptive-resid-tol must be > 0, "
            f"got {args.nonspd_adaptive_resid_tol}"
        )
    if float(args.nonspd_adaptive_growth_tol) < 1.0:
        raise ValueError(
            "--nonspd-adaptive-growth-tol must be >= 1.0, "
            f"got {args.nonspd_adaptive_growth_tol}"
        )
    if int(args.nonspd_adaptive_check_every) < 1:
        raise ValueError(
            "--nonspd-adaptive-check-every must be >= 1, "
            f"got {args.nonspd_adaptive_check_every}"
        )
    if float(args.nonspd_safe_fallback_tol) < 0.0:
        raise ValueError(
            "--nonspd-safe-fallback-tol must be >= 0, "
            f"got {args.nonspd_safe_fallback_tol}"
        )
    if float(args.nonspd_safe_early_y_tol) < 0.0:
        raise ValueError(
            "--nonspd-safe-early-y-tol must be >= 0, "
            f"got {args.nonspd_safe_early_y_tol}"
        )
    nonspd_safe_fallback_tol = (
        float(args.nonspd_safe_fallback_tol)
        if float(args.nonspd_safe_fallback_tol) > 0.0
        else None
    )
    nonspd_safe_early_y_tol = (
        float(args.nonspd_safe_early_y_tol)
        if float(args.nonspd_safe_early_y_tol) > 0.0
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dtype_compute = (
        torch.float32
        if (args.dtype == "fp32" or device.type != "cuda")
        else torch.bfloat16
    )

    sizes = parse_shapes(args.sizes)
    cases = _parse_case_csv(args.cases)
    pe_quad_t, coeff_desc = build_pe_schedules(
        l_target=0.05,
        device=device,
        coeff_mode=args.coeff_mode,
        coeff_seed=args.coeff_seed,
        coeff_safety=args.coeff_safety,
        coeff_no_final_safety=args.coeff_no_final_safety,
        p_val=1,
    )
    pe_quad_coeffs = _quad_coeffs(pe_quad_t)
    print(f"[coeff] using {coeff_desc} (PE steps = {len(pe_quad_t)})")

    uncoupled_fn = maybe_compile(inverse_proot_pe_quadratic_uncoupled, args.compile)
    coupled_solve_fn = maybe_compile(inverse_solve_pe_quadratic_coupled, args.compile)

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    with torch.inference_mode():
        for n in sizes:
            for k in ks:
                print(
                    f"\n== Non-SPD Size {n}x{n} | RHS {n}x{k} | dtype={dtype_compute} =="
                )
                print(
                    f"p=1 | precond={args.precond} | precond_ruiz_iters={args.precond_ruiz_iters} | "
                    f"compile={args.compile} | timing_reps={args.timing_reps} | "
                    f"timing_warmup_reps={args.timing_warmup_reps} | "
                    f"adapt(resid_tol={args.nonspd_adaptive_resid_tol}, "
                    f"growth_tol={args.nonspd_adaptive_growth_tol}, "
                    f"check_every={args.nonspd_adaptive_check_every}, "
                    f"safe_fallback_tol={args.nonspd_safe_fallback_tol}, "
                    f"safe_early_y_tol={args.nonspd_safe_early_y_tol})"
                )

                for case in cases:
                    mats = make_nonspd_cases(
                        case, n, args.trials, device, torch.float32, g
                    )
                    mats = [m.to(dtype_compute) for m in mats]
                    prepared_inputs, ms_precond_med = prepare_nonspd_solve_inputs(
                        mats=mats,
                        device=device,
                        k=k,
                        dtype=dtype_compute,
                        generator=g,
                        precond=str(args.precond),
                        precond_ruiz_iters=int(args.precond_ruiz_iters),
                    )
                    Z_true = compute_ground_truth(prepared_inputs)

                    rows: List[Tuple[str, NonSpdBenchResult]] = []
                    for method in METHODS:
                        try:
                            rr = eval_method(
                                prepared_inputs=prepared_inputs,
                                ground_truth_Z=Z_true,
                                device=device,
                                method=method,
                                pe_coeffs=pe_quad_coeffs,
                                timing_reps=args.timing_reps,
                                timing_warmup_reps=args.timing_warmup_reps,
                                ms_precond_median=ms_precond_med,
                                nonspd_adaptive_resid_tol=float(
                                    args.nonspd_adaptive_resid_tol
                                ),
                                nonspd_adaptive_growth_tol=float(
                                    args.nonspd_adaptive_growth_tol
                                ),
                                nonspd_adaptive_check_every=int(
                                    args.nonspd_adaptive_check_every
                                ),
                                nonspd_safe_fallback_tol=nonspd_safe_fallback_tol,
                                nonspd_safe_early_y_tol=nonspd_safe_early_y_tol,
                                uncoupled_fn=uncoupled_fn,
                                coupled_solve_fn=coupled_solve_fn,
                            )
                        except (NotImplementedError, RuntimeError) as e:
                            print(f"  SKIP {method}: {e}")
                            continue
                        rows.append((method, rr))

                    print(f"\n-- case {case} --")
                    for name, rr in rows:
                        mem_str = (
                            f" | mem {rr.mem_alloc_mb:4.0f}MB"
                            if not math.isnan(rr.mem_alloc_mb)
                            else ""
                        )
                        print(
                            f"{name:<28s} {rr.ms:8.3f} ms "
                            f"(pre {rr.ms_precond:.3f} + iter {rr.ms_iter:.3f}){mem_str} | "
                            f"relerr vs solve: {rr.rel_err:.3e} | bad {rr.bad}"
                        )

                    finite = [
                        (nm, rr)
                        for nm, rr in rows
                        if rr.bad == 0 and math.isfinite(rr.rel_err)
                    ]
                    if finite:
                        best_name, best_rr = min(finite, key=lambda t: t[1].ms)
                        print(
                            f"BEST finite: {best_name} @ {best_rr.ms:.3f} ms, relerr={best_rr.rel_err:.3e}"
                        )
                    else:
                        print("BEST finite: none")


if __name__ == "__main__":
    main()

