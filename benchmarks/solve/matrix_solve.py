"""
matrix_solve.py

Benchmark harness for inverse-p-th-root applied directly to a block of vectors B:
Z = A^{-1/p} B.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import math

import torch

try:
    from benchmarks._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from fast_iroot import (
    SPD_PRECOND_MODES,
    _quad_coeffs,
    apply_inverse_proot_chebyshev,
    build_pe_schedules,
    inverse_proot_pe_quadratic_uncoupled,
    inverse_solve_pe_quadratic_coupled,
)
from benchmarks.common import parse_shapes, make_spd_cases, maybe_compile
from benchmarks.solve.bench_solve_core import (
    matrix_solve_methods,
    prepare_solve_inputs,
    eval_solve_method,
    compute_ground_truth,
)


def _parse_int_csv(csv: str) -> tuple[int, ...]:
    vals = [int(tok.strip()) for tok in csv.split(",") if tok.strip()]
    if len(vals) == 0:
        raise ValueError("Expected at least one integer in comma-separated list")
    if any(v < 0 for v in vals):
        raise ValueError(f"All values must be >= 0, got {vals}")
    return tuple(sorted(set(vals)))


def _parse_case_csv(csv: str) -> list[str]:
    vals = [tok.strip() for tok in str(csv).split(",") if tok.strip()]
    if len(vals) == 0:
        raise ValueError("Expected at least one case in --cases")
    return vals


def _parse_methods_csv(spec: str, available: list[str]) -> list[str]:
    toks = [tok.strip() for tok in str(spec).split(",") if tok.strip()]
    if not toks:
        return list(available)
    unknown = [m for m in toks if m not in available]
    if unknown:
        raise ValueError(
            "Unknown method(s) in --methods: "
            f"{unknown}. Available: {available}"
        )
    out: list[str] = []
    seen: set[str] = set()
    for m in toks:
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


def main():
    p = argparse.ArgumentParser(
        description="Benchmark inverse-p-th-root Solve (Z = A^{-1/p} B)"
    )
    p.add_argument("--p", type=int, default=2, help="Root exponent")
    p.add_argument("--sizes", type=str, default="1024")
    p.add_argument(
        "--k", type=str, default="1,16,64,1024", help="RHS column counts (CSV)"
    )
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--cases",
        type=str,
        default="gaussian_spd,illcond_1e6",
        help="Comma-separated SPD case names",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="PE-Quad-Coupled-Apply",
        help=(
            "Optional comma-separated method subset. Defaults to best target method "
            "only (`PE-Quad-Coupled-Apply`). "
            "Example: 'PE-Quad-Coupled-Apply,Inverse-Newton-Coupled-Apply'"
        ),
    )
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    p.add_argument(
        "--precond", type=str, default="jacobi", choices=list(SPD_PRECOND_MODES)
    )
    p.add_argument("--precond-ruiz-iters", type=int, default=2)
    p.add_argument("--l-target", type=float, default=0.05)
    p.add_argument("--timing-reps", type=int, default=5)
    p.add_argument(
        "--timing-warmup-reps",
        type=int,
        default=2,
        help=(
            "Extra untimed warmup calls before timing each method/input cell "
            "(helps reduce first-run order bias)."
        ),
    )
    p.add_argument(
        "--symmetrize-every",
        type=int,
        default=1,
        help="Apply Y symmetrization every k non-terminal steps in coupled apply (k>=1).",
    )
    p.add_argument(
        "--cheb-degree", type=int, default=32, help="Degree for Chebyshev polynomial"
    )
    p.add_argument(
        "--cheb-degree-klt",
        type=int,
        default=24,
        help=(
            "When --cheb-mode=fixed and k<n, cap effective Chebyshev degree to this "
            "value; set -1 to disable."
        ),
    )
    p.add_argument(
        "--cheb-mode",
        type=str,
        default="fixed",
        choices=["fixed", "minimax-auto"],
        help=(
            "Chebyshev mode: fixed uses --cheb-degree; minimax-auto chooses the "
            "smallest minimax-approx degree (from --cheb-candidate-degrees) whose "
            "grid error bound is no worse than fixed baseline."
        ),
    )
    p.add_argument(
        "--cheb-candidate-degrees",
        type=str,
        default="8,12,16,24,32",
        help="Comma-separated degree candidates for --cheb-mode=minimax-auto.",
    )
    p.add_argument(
        "--cheb-error-grid",
        type=int,
        default=4097,
        help="Grid size used for chebyshev minimax-auto bound checks (>=257).",
    )
    p.add_argument(
        "--cheb-max-relerr-mult",
        type=float,
        default=1.0,
        help=(
            "Accept minimax-auto candidate only if its grid max-relative-error <= "
            "baseline_error * this multiplier (>=1.0)."
        ),
    )
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help=(
            "Enable CUDA graph replay for fixed-shape PE-Quad-Coupled-Apply timing "
            "path (CUDA only, falls back to eager on capture failure)."
        ),
    )
    p.add_argument(
        "--cuda-graph-warmup",
        type=int,
        default=3,
        help="Warmup calls before CUDA graph capture when --cuda-graph is enabled.",
    )
    p.add_argument(
        "--cheb-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow CUDA graph replay for Chebyshev-Apply (CUDA only), independent "
            "of the global --cuda-graph toggle used by other methods."
        ),
    )
    p.add_argument(
        "--online-coeff-mode",
        type=str,
        default="greedy-affine-opt",
        choices=["off", "greedy-newton", "greedy-minimax", "greedy-affine-opt"],
        help=(
            "Optional coupled PE coefficient schedule adaptation. "
            "'greedy-newton' picks per-step between base quadratic and inverse-Newton "
            "affine coefficients using a scalar interval cost model; "
            "'greedy-affine-opt' picks per-step among base quadratic, inverse-Newton, "
            "and interval-optimal affine coefficients; "
            "'greedy-minimax' also includes a local-basis minimax-alpha candidate "
            "with NS dominance gating."
        ),
    )
    p.add_argument(
        "--online-coeff-cost-model",
        type=str,
        default="shape-aware",
        choices=["gemm", "shape-aware"],
        help=(
            "Cost model used by coupled PE online coefficient planning: "
            "'gemm' matches legacy equal-GEMM weighting, "
            "'shape-aware' uses k/n-weighted rhs costs and terminal rhs-direct modeling."
        ),
    )
    p.add_argument(
        "--online-coeff-min-rel-improve",
        type=float,
        default=0.0,
        help=(
            "Minimum relative score improvement required to switch a step from base "
            "quadratic to another candidate in online coefficient modes."
        ),
    )
    p.add_argument(
        "--online-coeff-min-ns-logwidth-rel-improve",
        type=float,
        default=0.0,
        help=(
            "For --online-coeff-mode=greedy-minimax: minimum relative mapped "
            "log-width improvement vs inverse-Newton required to accept minimax."
        ),
    )
    p.add_argument(
        "--online-coeff-target-interval-err",
        type=float,
        default=0.01,
        help=(
            "If > 0, trim the coupled PE schedule to the shortest prefix whose "
            "predicted scalar interval error is <= this target."
        ),
    )
    p.add_argument(
        "--online-coeff-min-steps",
        type=int,
        default=1,
        help=(
            "Minimum number of PE coupled steps to keep when interval-target trimming is enabled."
        ),
    )
    p.add_argument(
        "--online-stop-tol",
        type=float,
        default=0.0,
        help=(
            "If > 0, enable low-overhead online PE early-stop for coupled apply based on "
            "max|diag(Y)-1| <= tol."
        ),
    )
    p.add_argument(
        "--online-stop-metric",
        type=str,
        default="diag",
        choices=["diag", "fro"],
        help=(
            "Metric used for coupled online early-stop when --online-stop-tol > 0: "
            "'diag' uses max|diag(Y)-1|, 'fro' uses ||Y-I||_F/sqrt(n)."
        ),
    )
    p.add_argument(
        "--online-stop-check-every",
        type=int,
        default=1,
        help=(
            "Evaluate the online early-stop metric every k non-terminal PE steps (k>=1)."
        ),
    )
    p.add_argument(
        "--terminal-tail-steps",
        type=int,
        default=1,
        help=(
            "Number of final PE steps to run in terminal RHS-only mode "
            "(skip Y update). Default 1 preserves existing behavior."
        ),
    )
    p.add_argument(
        "--online-min-steps",
        type=int,
        default=2,
        help="Minimum number of coupled PE steps before online early-stop is allowed.",
    )
    p.add_argument(
        "--post-correction-steps",
        type=int,
        default=0,
        help=(
            "Optional number of RHS-only residual-binomial post-correction passes for "
            "coupled apply (currently supported for SPD p=2,4)."
        ),
    )
    p.add_argument(
        "--post-correction-order",
        type=int,
        default=2,
        choices=[1, 2],
        help=(
            "Residual-binomial post-correction order: 1 (affine) or 2 (quadratic)."
        ),
    )
    args = p.parse_args()
    if int(args.symmetrize_every) < 1:
        raise ValueError(
            f"--symmetrize-every must be >= 1, got {args.symmetrize_every}"
        )
    if int(args.precond_ruiz_iters) < 1:
        raise ValueError(
            f"--precond-ruiz-iters must be >= 1, got {args.precond_ruiz_iters}"
        )
    if int(args.cheb_degree) < 0:
        raise ValueError(f"--cheb-degree must be >= 0, got {args.cheb_degree}")
    if int(args.cheb_degree_klt) < -1:
        raise ValueError(
            f"--cheb-degree-klt must be >= -1, got {args.cheb_degree_klt}"
        )
    if int(args.cheb_error_grid) < 257:
        raise ValueError(
            f"--cheb-error-grid must be >= 257, got {args.cheb_error_grid}"
        )
    if float(args.cheb_max_relerr_mult) < 1.0:
        raise ValueError(
            f"--cheb-max-relerr-mult must be >= 1.0, got {args.cheb_max_relerr_mult}"
        )
    if float(args.online_stop_tol) < 0.0:
        raise ValueError(f"--online-stop-tol must be >= 0, got {args.online_stop_tol}")
    if int(args.terminal_tail_steps) < 0:
        raise ValueError(
            f"--terminal-tail-steps must be >= 0, got {args.terminal_tail_steps}"
        )
    if int(args.online_min_steps) < 1:
        raise ValueError(
            f"--online-min-steps must be >= 1, got {args.online_min_steps}"
        )
    if int(args.online_stop_check_every) < 1:
        raise ValueError(
            "--online-stop-check-every must be >= 1, "
            f"got {args.online_stop_check_every}"
        )
    if int(args.post_correction_steps) < 0:
        raise ValueError(
            "--post-correction-steps must be >= 0, "
            f"got {args.post_correction_steps}"
        )
    if int(args.timing_warmup_reps) < 0:
        raise ValueError(
            f"--timing-warmup-reps must be >= 0, got {args.timing_warmup_reps}"
        )
    if int(args.cuda_graph_warmup) < 1:
        raise ValueError(
            f"--cuda-graph-warmup must be >= 1, got {args.cuda_graph_warmup}"
        )
    if float(args.online_coeff_min_rel_improve) < 0.0:
        raise ValueError(
            "--online-coeff-min-rel-improve must be >= 0, "
            f"got {args.online_coeff_min_rel_improve}"
        )
    if float(args.online_coeff_min_ns_logwidth_rel_improve) < 0.0:
        raise ValueError(
            "--online-coeff-min-ns-logwidth-rel-improve must be >= 0, "
            f"got {args.online_coeff_min_ns_logwidth_rel_improve}"
        )
    if float(args.online_coeff_target_interval_err) < 0.0:
        raise ValueError(
            "--online-coeff-target-interval-err must be >= 0, "
            f"got {args.online_coeff_target_interval_err}"
        )
    if int(args.online_coeff_min_steps) < 1:
        raise ValueError(
            f"--online-coeff-min-steps must be >= 1, got {args.online_coeff_min_steps}"
        )
    cheb_candidate_degrees = _parse_int_csv(args.cheb_candidate_degrees)
    ks = parse_shapes(args.k)
    online_stop_tol = (
        float(args.online_stop_tol) if float(args.online_stop_tol) > 0.0 else None
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
    p_val = args.p
    online_coeff_mode = str(args.online_coeff_mode)
    online_coeff_cost_model = str(args.online_coeff_cost_model)
    cases = _parse_case_csv(args.cases)
    methods = _parse_methods_csv(str(args.methods), matrix_solve_methods(p_val))

    pe_quad_t, coeff_desc = build_pe_schedules(
        l_target=args.l_target,
        device=device,
        coeff_mode="auto",
        coeff_seed=0,
        coeff_safety=1.0,
        coeff_no_final_safety=False,
        p_val=p_val,
    )
    print(f"[coeff] using {coeff_desc} (PE steps = {len(pe_quad_t)})")
    pe_quad_coeffs = _quad_coeffs(pe_quad_t)

    uncoupled_fn = maybe_compile(inverse_proot_pe_quadratic_uncoupled, args.compile)
    coupled_solve_fn = maybe_compile(inverse_solve_pe_quadratic_coupled, args.compile)
    cheb_apply_fn = maybe_compile(apply_inverse_proot_chebyshev, args.compile)

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    with torch.inference_mode():
        for n in sizes:
            for k in ks:
                print(f"\n== SPD Size {n}x{n} | RHS {n}x{k} | dtype={dtype_compute} ==")
                print(
                    f"precond={args.precond} | l_target={args.l_target} | p={p_val} | "
                    f"ruiz_iters={args.precond_ruiz_iters} | "
                    f"cheb_deg={args.cheb_degree} | cheb_deg_klt={args.cheb_degree_klt} | "
                    f"cheb_mode={args.cheb_mode} | "
                    f"symEvery={args.symmetrize_every} | "
                    f"online_coeff_mode={online_coeff_mode} | "
                    f"online_coeff_cost_model={online_coeff_cost_model} | "
                    f"online_coeff_target_err={args.online_coeff_target_interval_err} | "
                    f"online_stop_tol={args.online_stop_tol} | "
                    f"terminal_tail_steps={args.terminal_tail_steps} | "
                    f"online_stop_metric={args.online_stop_metric} | "
                    f"post_correction_steps={args.post_correction_steps} | "
                    f"cuda_graph={bool(args.cuda_graph)} | "
                    f"methods={','.join(methods)}"
                )

                for case in cases:
                    mats = make_spd_cases(
                        case, n, args.trials, device, torch.float32, g
                    )
                    mats = [m.to(dtype_compute) for m in mats]

                    prepared_inputs, ms_precond_med = prepare_solve_inputs(
                        mats=mats,
                        device=device,
                        k=k,
                        precond=args.precond,
                        precond_ruiz_iters=args.precond_ruiz_iters,
                        ridge_rel=1e-4,
                        l_target=args.l_target,
                        dtype=dtype_compute,
                        generator=g,
                    )

                    Z_true = compute_ground_truth(prepared_inputs, p_val)
                    rows = []

                    for name in methods:
                        try:
                            rr = eval_solve_method(
                                prepared_inputs=prepared_inputs,
                                ms_precond_median=ms_precond_med,
                                ground_truth_Z=Z_true,
                                device=device,
                                method=name,
                                pe_quad_coeffs=pe_quad_coeffs,
                                cheb_degree=args.cheb_degree,
                                cheb_degree_klt=args.cheb_degree_klt,
                                cheb_mode=args.cheb_mode,
                                cheb_candidate_degrees=cheb_candidate_degrees,
                                cheb_error_grid_n=args.cheb_error_grid,
                                cheb_max_relerr_mult=args.cheb_max_relerr_mult,
                                timing_reps=args.timing_reps,
                                timing_warmup_reps=args.timing_warmup_reps,
                                p_val=p_val,
                                l_min=args.l_target,
                                symmetrize_every=args.symmetrize_every,
                                online_stop_tol=online_stop_tol,
                                terminal_tail_steps=args.terminal_tail_steps,
                                online_min_steps=args.online_min_steps,
                                online_stop_metric=args.online_stop_metric,
                                online_stop_check_every=args.online_stop_check_every,
                                post_correction_steps=args.post_correction_steps,
                                post_correction_order=args.post_correction_order,
                                online_coeff_mode=online_coeff_mode,
                                online_coeff_cost_model=online_coeff_cost_model,
                                online_coeff_min_rel_improve=args.online_coeff_min_rel_improve,
                                online_coeff_min_ns_logwidth_rel_improve=(
                                    args.online_coeff_min_ns_logwidth_rel_improve
                                ),
                                online_coeff_target_interval_err=(
                                    args.online_coeff_target_interval_err
                                ),
                                online_coeff_min_steps=args.online_coeff_min_steps,
                                use_cuda_graph=bool(args.cuda_graph),
                                cuda_graph_warmup=int(args.cuda_graph_warmup),
                                cheb_cuda_graph=bool(args.cheb_cuda_graph),
                                uncoupled_fn=uncoupled_fn,
                                coupled_solve_fn=coupled_solve_fn,
                                cheb_apply_fn=cheb_apply_fn,
                            )
                        except (NotImplementedError, RuntimeError) as e:
                            print(f"  SKIP {name}: {e}")
                            continue
                        rows.append((name, rr))

                    print(f"\n-- case {case} --")
                    for name, rr in rows:
                        mem_str = (
                            f" | mem {rr.mem_alloc_mb:4.0f}MB"
                            if not math.isnan(rr.mem_alloc_mb)
                            else ""
                        )
                        cheb_deg_str = (
                            f" | cheb_deg {rr.cheb_degree_used:.0f}"
                            if not math.isnan(rr.cheb_degree_used)
                            else ""
                        )
                        pe_newton_str = (
                            f" | newton_steps {rr.pe_newton_steps_used:.0f}"
                            if not math.isnan(rr.pe_newton_steps_used)
                            else ""
                        )
                        pe_minimax_str = (
                            f" | minimax_steps {rr.pe_minimax_steps_used:.0f}"
                            if not math.isnan(rr.pe_minimax_steps_used)
                            else ""
                        )
                        pe_affine_str = (
                            f" | affineopt_steps {rr.pe_affine_opt_steps_used:.0f}"
                            if not math.isnan(rr.pe_affine_opt_steps_used)
                            else ""
                        )
                        pe_steps_str = (
                            f" | steps {rr.pe_steps_used:.0f}"
                            if not math.isnan(rr.pe_steps_used)
                            else ""
                        )
                        print(
                            f"{name:<28s} {rr.ms:8.3f} ms (pre {rr.ms_precond:.3f} + iter {rr.ms_iter:.3f}){mem_str} | "
                            f"relerr vs true: {rr.rel_err:.3e}"
                            f"{cheb_deg_str}{pe_newton_str}{pe_minimax_str}{pe_affine_str}{pe_steps_str}"
                        )


if __name__ == "__main__":
    main()

