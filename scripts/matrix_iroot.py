"""
matrix_iroot.py

Matmul-only inverse-p-th-root benchmark harness.
Core iteration/preconditioning code lives in fast_iroot/.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# Allow running as `python scripts/matrix_iroot.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fast_iroot import (
    _quad_coeffs,
    build_pe_schedules,
    inverse_proot_pe_quadratic_uncoupled,
    inverse_proot_pe_quadratic_coupled,
    precond_spd,
)

try:
    from .bench_common import parse_shapes, make_spd_cases, maybe_compile, _spd_from_eigs
    from .bench_iroot_core import (
        BenchResult,
        MATRIX_IROOT_METHODS,
        prepare_preconditioned_inputs,
        eval_method,
    )
except ImportError:
    from scripts.bench_common import (
        parse_shapes,
        make_spd_cases,
        maybe_compile,
        _spd_from_eigs,
    )
    from scripts.bench_iroot_core import (
        BenchResult,
        MATRIX_IROOT_METHODS,
        prepare_preconditioned_inputs,
        eval_method,
    )


def main():
    p = argparse.ArgumentParser(
        description="Benchmark inverse-p-th-root iterations (PE-Quad schedules)"
    )
    p.add_argument(
        "--p", type=int, default=4, help="Root exponent (e.g. 4 for inverse 4th root)"
    )
    p.add_argument("--sizes", type=str, default="256,512,1024")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    p.add_argument(
        "--compile",
        action="store_true",
        help="Compile for maximum performance",
    )
    p.add_argument(
        "--precond", type=str, default="aol", choices=["none", "frob", "aol"]
    )
    p.add_argument("--ridge-rel", type=float, default=1e-4)
    p.add_argument("--l-target", type=float, default=0.05)
    p.add_argument("--target-resid", type=float, default=0.01)
    p.add_argument(
        "--target-metric",
        type=str,
        default="residual",
        choices=["residual", "hard_dir"],
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
    p.add_argument("--timing-reps", type=int, default=1)
    p.add_argument("--no-symmetrize-y", action="store_true")
    p.add_argument(
        "--symmetrize-every",
        type=int,
        default=1,
        help="Apply Y symmetrization every k non-terminal steps (k>=1).",
    )
    p.add_argument(
        "--metrics-mode", type=str, default="full", choices=["full", "coupled"]
    )
    p.add_argument("--power-iters", type=int, default=0)
    p.add_argument("--mv-samples", type=int, default=0)
    p.add_argument("--hard-probe-iters", type=int, default=0)
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
    cases = ["gaussian_spd", "illcond_1e6", "illcond_1e12", "near_rank_def", "spike"]
    p_val = args.p

    pe_quad_t, coeff_desc = build_pe_schedules(
        l_target=args.l_target,
        device=device,
        coeff_mode=args.coeff_mode,
        coeff_seed=args.coeff_seed,
        coeff_safety=args.coeff_safety,
        coeff_no_final_safety=args.coeff_no_final_safety,
        p_val=p_val,
    )
    print(f"[coeff] using {coeff_desc}")
    pe_quad_coeffs = _quad_coeffs(pe_quad_t)

    uncoupled_fn = maybe_compile(inverse_proot_pe_quadratic_uncoupled, args.compile)
    coupled_fn = maybe_compile(inverse_proot_pe_quadratic_coupled, args.compile)

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    with torch.inference_mode():
        for n in sizes:
            print(
                f"== SPD size {n}x{n} | dtype={dtype_compute} | compile={args.compile} | "
                f"precond={args.precond} | l_target={args.l_target} | lmax=row_sum | terminal=True | "
                f"timing_reps={max(1, args.timing_reps)} | symY={not args.no_symmetrize_y} | "
                f"symEvery={args.symmetrize_every} | "
                f"metrics={args.metrics_mode} | power_it={args.power_iters} | "
                f"mv_k={args.mv_samples} | hard_it={args.hard_probe_iters} =="
            )

            warm = make_spd_cases(
                "gaussian_spd", n, max(1, args.warmup), device, torch.float32, g
            )
            ws: Optional[object] = None
            for A in warm:
                A = A.to(dtype_compute)
                A_norm, _ = precond_spd(
                    A,
                    mode=args.precond,
                    ridge_rel=args.ridge_rel,
                    l_target=args.l_target,
                )
                _, ws = uncoupled_fn(
                    A_norm,
                    abc_t=pe_quad_coeffs,
                    p_val=p_val,
                    ws=ws,
                    symmetrize_X=not args.no_symmetrize_y,
                )

            for case in cases:
                mats = make_spd_cases(case, n, args.trials, device, torch.float32, g)
                mats = [m.to(dtype_compute) for m in mats]

                prepared_inputs, ms_precond_med = prepare_preconditioned_inputs(
                    mats=mats,
                    device=device,
                    precond=args.precond,
                    ridge_rel=args.ridge_rel,
                    l_target=args.l_target,
                )

                rows: List[Tuple[str, BenchResult]] = []
                for name in MATRIX_IROOT_METHODS:
                    rr = eval_method(
                        prepared_inputs=prepared_inputs,
                        ms_precond_median=ms_precond_med,
                        device=device,
                        method=name,
                        pe_quad_coeffs=pe_quad_coeffs,
                        timing_reps=args.timing_reps,
                        symmetrize_Y=not args.no_symmetrize_y,
                        symmetrize_every=args.symmetrize_every,
                        compute_relerr=(args.metrics_mode == "full"),
                        power_iters=args.power_iters,
                        mv_samples=args.mv_samples,
                        hard_probe_iters=args.hard_probe_iters,
                        p_val=p_val,
                        uncoupled_fn=uncoupled_fn,
                        coupled_fn=coupled_fn,
                    )
                    rows.append((name, rr))

                print(f"-- case {case} --")
                for name, rr in rows:
                    if not math.isnan(rr.mem_alloc_mb):
                        mem_str = f" | mem {rr.mem_alloc_mb:4.0f}MB"
                    else:
                        mem_str = ""
                    y_str = (
                        f" | Y_res {rr.coupled_y_resid:.3e}"
                        if not math.isnan(rr.coupled_y_resid)
                        else ""
                    )
                    print(
                        f"{name:<22s} {rr.ms:8.3f} ms (pre {rr.ms_precond:.3f} + iter {rr.ms_iter:.3f}){mem_str} | "
                        f"resid {rr.residual:.3e} p95 {rr.residual_p95:.3e} max {rr.residual_max:.3e}{y_str} | "
                        f"relerr {rr.relerr:.3e} | r2 {rr.residual_spec:.3e} | hard {rr.hard_dir:.3e} | "
                        f"symX {rr.sym_x:.2e} symW {rr.sym_w:.2e} | mv {rr.mv_err:.3e} | bad {rr.bad}"
                    )

                if args.target_metric == "hard_dir":
                    feasible = [
                        (nm, rr)
                        for nm, rr in rows
                        if rr.bad == 0 and rr.hard_dir <= args.target_resid
                    ]
                else:
                    feasible = [
                        (nm, rr)
                        for nm, rr in rows
                        if rr.bad == 0 and rr.residual <= args.target_resid
                    ]

                if feasible:
                    best_name, best_rr = min(feasible, key=lambda t: t[1].ms)
                    print(
                        f"BEST<={args.target_metric}={args.target_resid:.3g}: "
                        f"{best_name} @ {best_rr.ms:.3f} ms, resid={best_rr.residual:.3e}, hard={best_rr.hard_dir:.3e}"
                    )
                else:
                    ok = [(nm, rr) for nm, rr in rows if rr.bad == 0]
                    if ok:
                        best_name, best_rr = min(ok, key=lambda t: t[1].ms)
                        print(
                            f"BEST overall: {best_name} @ {best_rr.ms:.3f} ms, resid={best_rr.residual:.3e}, hard={best_rr.hard_dir:.3e}"
                        )
                    else:
                        print("BEST overall: none (all runs produced non-finite output)")
                print()


if __name__ == "__main__":
    main()
