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
    from scripts._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from fast_iroot import (
    _quad_coeffs,
    apply_inverse_proot_chebyshev,
    build_pe_schedules,
    inverse_proot_pe_quadratic_uncoupled,
    inverse_solve_pe_quadratic_coupled,
)
from scripts.bench_common import parse_shapes, make_spd_cases, maybe_compile
from scripts.bench_solve_core import (
    MATRIX_SOLVE_METHODS,
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


def main():
    p = argparse.ArgumentParser(
        description="Benchmark inverse-p-th-root Solve (Z = A^{-1/p} B)"
    )
    p.add_argument("--p", type=int, default=2, help="Root exponent")
    p.add_argument("--sizes", type=str, default="1024")
    p.add_argument("--k", type=int, default=16, help="Number of RHS columns (K << N)")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    p.add_argument(
        "--precond", type=str, default="frob", choices=["none", "frob", "aol"]
    )
    p.add_argument("--l-target", type=float, default=0.05)
    p.add_argument("--timing-reps", type=int, default=5)
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
        "--online-stop-tol",
        type=float,
        default=0.0,
        help=(
            "If > 0, enable low-overhead online PE early-stop for coupled apply based on "
            "max|diag(Y)-1| <= tol."
        ),
    )
    p.add_argument(
        "--online-min-steps",
        type=int,
        default=2,
        help="Minimum number of coupled PE steps before online early-stop is allowed.",
    )

    args = p.parse_args()
    if int(args.symmetrize_every) < 1:
        raise ValueError(f"--symmetrize-every must be >= 1, got {args.symmetrize_every}")
    if int(args.cheb_degree) < 0:
        raise ValueError(f"--cheb-degree must be >= 0, got {args.cheb_degree}")
    if int(args.cheb_error_grid) < 257:
        raise ValueError(f"--cheb-error-grid must be >= 257, got {args.cheb_error_grid}")
    if float(args.cheb_max_relerr_mult) < 1.0:
        raise ValueError(
            f"--cheb-max-relerr-mult must be >= 1.0, got {args.cheb_max_relerr_mult}"
        )
    if float(args.online_stop_tol) < 0.0:
        raise ValueError(f"--online-stop-tol must be >= 0, got {args.online_stop_tol}")
    if int(args.online_min_steps) < 1:
        raise ValueError(f"--online-min-steps must be >= 1, got {args.online_min_steps}")
    cheb_candidate_degrees = _parse_int_csv(args.cheb_candidate_degrees)
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

    cases = ["gaussian_spd", "illcond_1e6"]

    with torch.inference_mode():
        for n in sizes:
            k = args.k
            print(f"\n== SPD Size {n}x{n} | RHS {n}x{k} | dtype={dtype_compute} ==")
            print(
                f"precond={args.precond} | l_target={args.l_target} | p={p_val} | "
                f"cheb_deg={args.cheb_degree} | cheb_mode={args.cheb_mode} | "
                f"symEvery={args.symmetrize_every} | online_stop_tol={args.online_stop_tol}"
            )

            for case in cases:
                mats = make_spd_cases(case, n, args.trials, device, torch.float32, g)
                mats = [m.to(dtype_compute) for m in mats]

                prepared_inputs, ms_precond_med = prepare_solve_inputs(
                    mats=mats,
                    device=device,
                    k=k,
                    precond=args.precond,
                    ridge_rel=1e-4,
                    l_target=args.l_target,
                    dtype=dtype_compute,
                    generator=g,
                )

                Z_true = compute_ground_truth(prepared_inputs, p_val)
                rows = []

                for name in MATRIX_SOLVE_METHODS:
                    rr = eval_solve_method(
                        prepared_inputs=prepared_inputs,
                        ms_precond_median=ms_precond_med,
                        ground_truth_Z=Z_true,
                        device=device,
                        method=name,
                        pe_quad_coeffs=pe_quad_coeffs,
                        cheb_degree=args.cheb_degree,
                        cheb_mode=args.cheb_mode,
                        cheb_candidate_degrees=cheb_candidate_degrees,
                        cheb_error_grid_n=args.cheb_error_grid,
                        cheb_max_relerr_mult=args.cheb_max_relerr_mult,
                        timing_reps=args.timing_reps,
                        p_val=p_val,
                        l_min=args.l_target,
                        symmetrize_every=args.symmetrize_every,
                        online_stop_tol=online_stop_tol,
                        online_min_steps=args.online_min_steps,
                        uncoupled_fn=uncoupled_fn,
                        coupled_solve_fn=coupled_solve_fn,
                        cheb_apply_fn=cheb_apply_fn,
                    )
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
                    print(
                        f"{name:<28s} {rr.ms:8.3f} ms (pre {rr.ms_precond:.3f} + iter {rr.ms_iter:.3f}){mem_str} | "
                        f"relerr vs true: {rr.rel_err:.3e}{cheb_deg_str}"
                    )


if __name__ == "__main__":
    main()
