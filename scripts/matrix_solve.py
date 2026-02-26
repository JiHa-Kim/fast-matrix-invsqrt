"""
matrix_solve.py

Benchmark harness for inverse-p-th-root applied directly to a block of vectors B:
Z = A^{-1/p} B.

Compares:
1) PE-Quad uncoupled inverse (forms X then X @ B)
2) PE-Quad coupled apply (streaming Z_t = B_t Z_{t-1})
3) Chebyshev apply (Clenshaw recurrence, O(N^2 K) memory/compute)
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import torch

# Allow running as `python scripts/matrix_solve.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fast_iroot import (
    precond_spd,
    _quad_coeffs,
    build_pe_schedules,
    inverse_proot_pe_quadratic_uncoupled,
    inverse_solve_pe_quadratic_coupled,
    apply_inverse_proot_chebyshev,
)

try:
    from .matrix_iroot import (
        time_ms_any,
        time_ms_repeat,
        parse_shapes,
        make_spd_cases,
        median,
    )
except ImportError:
    from matrix_iroot import (
        time_ms_any,
        time_ms_repeat,
        parse_shapes,
        make_spd_cases,
        median,
    )


@dataclass
class SolvePreparedInput:
    A_norm: torch.Tensor
    B: torch.Tensor
    stats: object


@torch.no_grad()
def prepare_solve_inputs(
    mats: List[torch.Tensor],
    device: torch.device,
    k: int,
    precond: str,
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
                ridge_rel=ridge_rel,
                l_target=l_target,
            ),
            device,
        )
        A_norm, stats = out
        ms_pre_list.append(t_pre)

        # Generate random RHS block B of shape (..., N, K)
        n = A_norm.shape[-1]
        # B is typically Gaussian iid for preconditioning/whitening tests
        B = torch.randn(
            *A_norm.shape[:-2], n, k, device=device, dtype=dtype, generator=generator
        )

        prepared.append(SolvePreparedInput(A_norm=A_norm, B=B, stats=stats))

    return prepared, (median(ms_pre_list) if ms_pre_list else float("nan"))


@dataclass
class SolveBenchResult:
    ms: float
    ms_iter: float
    ms_precond: float
    rel_err: float
    mem_alloc_mb: float
    mem_reserved_mb: float


def _build_solve_runner(
    method: str,
    pe_quad_coeffs: List[Tuple[float, float, float]],
    cheb_degree: int,
    p_val: int = 2,
    l_min: float = 0.05,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if method == "PE-Quad-Inverse-Multiply":
        ws_unc = None

        def run(A_norm: torch.Tensor, B: torch.Tensor):
            nonlocal ws_unc
            Xn, ws_unc = inverse_proot_pe_quadratic_uncoupled(
                A_norm,
                abc_t=pe_quad_coeffs,
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
            Zn, ws_cpl = inverse_solve_pe_quadratic_coupled(
                A_norm,
                B,
                abc_t=pe_quad_coeffs,
                p_val=p_val,
                ws=ws_cpl,
                symmetrize_Y=True,
                terminal_last_step=True,
            )
            return Zn

        return run

    if method == "Chebyshev-Apply":
        ws_cheb = None

        def run(A_norm: torch.Tensor, B: torch.Tensor):
            nonlocal ws_cheb
            Zn, ws_cheb = apply_inverse_proot_chebyshev(
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
    timing_reps: int,
    p_val: int = 2,
    l_min: float = 0.05,
) -> SolveBenchResult:
    ms_iter_list: List[float] = []
    err_list: List[float] = []
    mem_alloc_list: List[float] = []
    mem_res_list: List[float] = []

    if len(prepared_inputs) == 0:
        return SolveBenchResult(
            ms=float("nan"),
            ms_iter=float("nan"),
            ms_precond=float("nan"),
            rel_err=float("nan"),
            mem_alloc_mb=float("nan"),
            mem_reserved_mb=float("nan"),
        )

    for i, prep in enumerate(prepared_inputs):
        A_norm = prep.A_norm
        B = prep.B
        Z_true = ground_truth_Z[i]

        # If available, tighten Chebyshev's lower interval bound using preconditioning stats.
        # Larger l_min (while still <= lambda_min) reduces the approximation interval and typically
        # improves accuracy at fixed degree.
        l_min_eff = float(l_min)
        if method == "Chebyshev-Apply" and hasattr(prep.stats, "gersh_lo"):
            try:
                l_min_eff = max(l_min_eff, float(prep.stats.gersh_lo))
            except Exception:
                pass

        runner = _build_solve_runner(
            method=method,
            pe_quad_coeffs=pe_quad_coeffs,
            cheb_degree=cheb_degree,
            p_val=p_val,
            l_min=l_min_eff,
        )

        # measure memory
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)
            # Warm up and grab memory
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

        # compute rel err against "dense X @ B" ground truth (which itself is tested in other harnesses)
        if torch.isfinite(Zn).all() and torch.isfinite(Z_true).all():
            err_list.append(
                float(
                    torch.linalg.matrix_norm(Zn - Z_true)
                    / torch.linalg.matrix_norm(Z_true)
                )
            )
        else:
            err_list.append(float("inf"))

    ms_iter_med = median(ms_iter_list)
    ms_pre_med = ms_precond_median

    return SolveBenchResult(
        ms=ms_pre_med + ms_iter_med,
        ms_iter=ms_iter_med,
        ms_precond=ms_pre_med,
        rel_err=median(err_list),
        mem_alloc_mb=median(mem_alloc_list) if mem_alloc_list else float("nan"),
        mem_reserved_mb=median(mem_res_list) if mem_res_list else float("nan"),
    )


def compute_ground_truth(
    prepared: List[SolvePreparedInput], p_val: int
) -> List[torch.Tensor]:
    # Compute A_norm^{-1/p} explicitly and multiply
    # We use CPU / float64 for ground truth
    Z_true = []
    for prep in prepared:
        A = prep.A_norm.cpu().double()
        B = prep.B.cpu().double()
        L, Q = torch.linalg.eigh(A)
        # Eigenvalues might be slightly negative due to numerics, clamp
        L = L.clamp_min(1e-12)
        L_inv = torch.pow(L, -1.0 / p_val)
        A_inv = (Q * L_inv.unsqueeze(0)) @ Q.mT
        Z_true.append(
            (A_inv @ B).to(dtype=prep.A_norm.dtype, device=prep.A_norm.device)
        )
    return Z_true


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
        "--cheb-degree", type=int, default=32, help="Degree for Chebyshev polynomial"
    )
    p.add_argument("--compile", action="store_true")

    args = p.parse_args()

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

    # Get PE quad schedule
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

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    cases = ["gaussian_spd", "illcond_1e6"]

    with torch.inference_mode():
        for n in sizes:
            k = args.k
            print(f"\n== SPD Size {n}x{n} | RHS {n}x{k} | dtype={dtype_compute} ==")
            print(
                f"precond={args.precond} | l_target={args.l_target} | p={p_val} | cheb_deg={args.cheb_degree}"
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

                methods = [
                    "PE-Quad-Inverse-Multiply",
                    "PE-Quad-Coupled-Apply",
                    "Chebyshev-Apply",
                ]
                rows = []

                for name in methods:
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
                        l_min=args.l_target,  # Assumed via preconditioning
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
