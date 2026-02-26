"""
matrix_solve.py

Benchmark harness for inverse-p-th-root applied directly to a block of vectors B:
Z = A^{-1/p} B.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

# Allow running as `python scripts/matrix_solve.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fast_iroot import (
    _quad_coeffs,
    build_pe_schedules,
    inverse_proot_pe_quadratic_uncoupled,
    inverse_solve_pe_quadratic_coupled,
    apply_inverse_proot_chebyshev,
)

try:
    from .bench_common import parse_shapes, make_spd_cases, maybe_compile
    from .bench_solve_core import (
        MATRIX_SOLVE_METHODS,
        prepare_solve_inputs,
        eval_solve_method,
        compute_ground_truth,
    )
except ImportError:
    from scripts.bench_common import parse_shapes, make_spd_cases, maybe_compile
    from scripts.bench_solve_core import (
        MATRIX_SOLVE_METHODS,
        prepare_solve_inputs,
        eval_solve_method,
        compute_ground_truth,
    )


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
    p.add_argument("--compile", action="store_true")

    args = p.parse_args()
    if int(args.symmetrize_every) < 1:
        raise ValueError(f"--symmetrize-every must be >= 1, got {args.symmetrize_every}")

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
                f"cheb_deg={args.cheb_degree} | symEvery={args.symmetrize_every}"
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
                        timing_reps=args.timing_reps,
                        p_val=p_val,
                        l_min=args.l_target,
                        symmetrize_every=args.symmetrize_every,
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
                    print(
                        f"{name:<28s} {rr.ms:8.3f} ms (pre {rr.ms_precond:.3f} + iter {rr.ms_iter:.3f}){mem_str} | "
                        f"relerr vs true: {rr.rel_err:.3e}"
                    )


if __name__ == "__main__":
    main()
