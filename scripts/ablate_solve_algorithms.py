"""
ablate_solve_algorithms.py

Rigorous side-by-side ablation of new GEMM-heavy solve algorithms
(NSRC, Block CG, Chebyshev iterative, Hybrid PE+NSRC) against the
existing PE baseline and torch.linalg.solve.

Each algorithm is a separate cell benchmarked on identical inputs.
Outputs a markdown report with speed, accuracy, and delta-vs-baseline.
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:
    from scripts._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from fast_iroot import (
    _quad_coeffs,
    build_pe_schedules,
    inverse_proot_pe_quadratic_uncoupled,
    inverse_solve_pe_quadratic_coupled,
)
from fast_iroot.nsrc import nsrc_solve, nsrc_solve_preconditioned, hybrid_pe_nsrc_solve
from fast_iroot.block_cg import block_cg_solve, block_cg_solve_with_precond
from fast_iroot.chebyshev_iterative import chebyshev_iterative_solve
from fast_iroot.coupled import inverse_proot_pe_quadratic_coupled
from fast_iroot.lu_ir import lu_ir_solve, lu_solve_direct

from scripts.bench_common import (
    make_nonspd_cases,
    parse_shapes,
    maybe_compile,
    median,
    time_ms_repeat,
    time_ms_any,
)
from scripts.matrix_solve_nonspd import (
    NonSpdBenchResult,
    NonSpdPreparedInput,
    compute_ground_truth,
    prepare_nonspd_solve_inputs,
    precond_nonspd,
)


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlgoCell:
    tag: str
    desc: str
    algo: str  # algorithm key


def _default_cells() -> List[AlgoCell]:
    return [
        # ---- Baselines ----
        AlgoCell("B0", "baseline: torch.linalg.solve (exact)", "torch-solve"),
        AlgoCell(
            "B1", "baseline: PE-Quad-Coupled-Apply (current, full steps)", "pe-coupled"
        ),
        # ---- NSRC ----
        AlgoCell("N0", "NSRC scalar alpha, T=5", "nsrc-scalar-5"),
        AlgoCell("N1", "NSRC scalar alpha, T=10", "nsrc-scalar-10"),
        AlgoCell("N2", "NSRC scalar alpha, T=20", "nsrc-scalar-20"),
        # ---- Hybrid PE+NSRC ----
        AlgoCell("H0", "Hybrid PE(1)+NSRC(3)", "hybrid-pe1-nsrc3"),
        AlgoCell("H1", "Hybrid PE(2)+NSRC(3)", "hybrid-pe2-nsrc3"),
        AlgoCell("H2", "Hybrid PE(2)+NSRC(5)", "hybrid-pe2-nsrc5"),
        # ---- Block CG ----
        AlgoCell("C0", "Block CG, max 5 iters, tol=1e-3", "cg-5"),
        AlgoCell("C1", "Block CG, max 10 iters, tol=1e-3", "cg-10"),
        # ---- Chebyshev semi-iterative ----
        AlgoCell("CH0", "Chebyshev iterative, T=10", "cheb-iter-10"),
        AlgoCell("CH1", "Chebyshev iterative, T=20", "cheb-iter-20"),
        # ---- LU + Iterative Refinement ----
        AlgoCell("L0", "LU direct (lu_factor+lu_solve)", "lu-direct"),
        AlgoCell("L1", "LU + IR (1 refine step)", "lu-ir-1"),
        AlgoCell("L2", "LU + IR (3 refine steps)", "lu-ir-3"),
    ]


# ---------------------------------------------------------------------------
# Runner factory
# ---------------------------------------------------------------------------


def _build_runner(
    algo: str,
    pe_coeffs: Sequence[Tuple[float, float, float]],
    coupled_solve_fn,
    l_min_est: float,
    pe_quad_t,
):
    """Build a callable: (A_norm, B) -> Z for each algorithm."""

    if algo == "torch-solve":

        def run(A, B):
            return torch.linalg.solve(A, B)

        return run

    if algo == "pe-coupled":

        def run(A, B):
            Z, _ = coupled_solve_fn(
                A,
                B,
                abc_t=pe_coeffs,
                p_val=1,
                symmetrize_Y=False,
                terminal_last_step=True,
                assume_spd=False,
            )
            return Z

        return run

    if algo.startswith("nsrc-scalar-"):
        T = int(algo.split("-")[-1])
        alpha = 2.0 / (1.0 + l_min_est) if l_min_est > 0 else 1.0

        def run(A, B):
            Z, _ = nsrc_solve(A, B, alpha=alpha, max_iter=T)
            return Z

        return run

    if algo.startswith("hybrid-pe"):
        # Parse hybrid-peX-nsrcY
        parts = algo.replace("hybrid-pe", "").split("-nsrc")
        pe_steps = int(parts[0])
        ref_steps = int(parts[1])

        def run(A, B):
            Z, _ = hybrid_pe_nsrc_solve(
                A,
                B,
                abc_t=pe_quad_t,
                pe_steps=pe_steps,
                ref_steps=ref_steps,
            )
            return Z

        return run

    if algo.startswith("cg-"):
        max_iter = int(algo.split("-")[-1])

        def run(A, B):
            # Use diagonal preconditioning
            diag_inv = 1.0 / A.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
            Z, _, _ = block_cg_solve(
                A,
                B,
                max_iter=max_iter,
                tol=1e-3,
                diag_precond=diag_inv,
            )
            return Z

        return run

    if algo.startswith("cheb-iter-"):
        T = int(algo.split("-")[-1])
        l_min = max(l_min_est, 1e-6)

        def run(A, B):
            Z, _, _ = chebyshev_iterative_solve(
                A,
                B,
                l_min=l_min,
                l_max=1.0,
                max_iter=T,
            )
            return Z

        return run

    if algo == "lu-direct":

        def run(A, B):
            return lu_solve_direct(A, B)

        return run

    if algo.startswith("lu-ir-"):
        n_refine = int(algo.split("-")[-1])

        def run(A, B):
            Z, _ = lu_ir_solve(A, B, max_refine=n_refine)
            return Z

        return run

    raise ValueError(f"Unknown algo: {algo}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_cell(
    prepared: List[NonSpdPreparedInput],
    ground_truth: List[torch.Tensor],
    device: torch.device,
    cell: AlgoCell,
    pe_coeffs: Sequence[Tuple[float, float, float]],
    coupled_solve_fn,
    l_min_est: float,
    pe_quad_t,
    timing_reps: int,
    timing_warmup: int,
    ms_precond: float,
) -> NonSpdBenchResult:
    runner = _build_runner(
        cell.algo,
        pe_coeffs,
        coupled_solve_fn,
        l_min_est,
        pe_quad_t,
    )

    ms_list: List[float] = []
    err_list: List[float] = []
    bad = 0

    for i, prep in enumerate(prepared):
        A, B = prep.A_norm, prep.B
        Z_true = ground_truth[i]

        # Warmup
        for _ in range(timing_warmup):
            _ = runner(A, B)
        if device.type == "cuda" and timing_warmup > 0:
            torch.cuda.synchronize(device=device)

        ms, Z_hat = time_ms_repeat(lambda: runner(A, B), device, reps=timing_reps)
        ms_list.append(ms)

        if not torch.isfinite(Z_hat).all():
            bad += 1
            err_list.append(float("inf"))
            continue

        rel = float(
            torch.linalg.matrix_norm(Z_hat - Z_true)
            / torch.linalg.matrix_norm(Z_true).clamp_min(1e-30)
        )
        err_list.append(rel)

    return NonSpdBenchResult(
        ms=ms_precond + median(ms_list),
        ms_iter=median(ms_list),
        ms_precond=ms_precond,
        rel_err=median(err_list),
        bad=bad,
        mem_alloc_mb=float("nan"),
        mem_reserved_mb=float("nan"),
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _format_report(
    rows: List[Tuple[int, int, str, AlgoCell, NonSpdBenchResult]],
    cases: List[str],
) -> str:
    out: List[str] = []
    out.append("# GEMM-Heavy Solve Algorithm Ablation")
    out.append("")

    # Collect baselines for delta computation
    baselines: Dict[Tuple[int, int, str], NonSpdBenchResult] = {}
    for n, k, case, cell, rr in rows:
        if cell.tag == "B0":
            baselines[(n, k, case)] = rr

    out.append("## Per-Cell Results")
    out.append("")
    out.append(
        "| n | k | case | tag | algorithm | ms_iter | rel_err | delta ms vs torch.solve | bad |"
    )
    out.append("|---:|---:|---|---|---|---:|---:|---:|---:|")
    for n, k, case, cell, rr in rows:
        base = baselines.get((n, k, case))
        delta = rr.ms_iter - base.ms_iter if base else float("nan")
        out.append(
            f"| {n} | {k} | {case} | {cell.tag} | {cell.desc} "
            f"| {rr.ms_iter:.3f} | {rr.rel_err:.3e} | {delta:+.3f} | {rr.bad} |"
        )

    # Aggregate by tag
    out.append("")
    out.append("## Aggregated by Algorithm")
    out.append("")
    out.append(
        "| tag | algorithm | mean ms_iter | mean rel_err | max rel_err | total bad |"
    )
    out.append("|---|---|---:|---:|---:|---:|")
    tags_seen = []
    tag_desc = {}
    for _, _, _, cell, _ in rows:
        if cell.tag not in tags_seen:
            tags_seen.append(cell.tag)
            tag_desc[cell.tag] = cell.desc

    for tag in tags_seen:
        tag_rows = [rr for _, _, _, c, rr in rows if c.tag == tag]
        mean_ms = sum(r.ms_iter for r in tag_rows) / len(tag_rows)
        mean_err = sum(r.rel_err for r in tag_rows) / len(tag_rows)
        max_err = max(r.rel_err for r in tag_rows)
        total_bad = sum(r.bad for r in tag_rows)
        out.append(
            f"| {tag} | {tag_desc[tag]} "
            f"| {mean_ms:.3f} | {mean_err:.3e} | {max_err:.3e} | {total_bad} |"
        )

    out.append("")
    out.append("## Cell Legend")
    out.append("")
    for tag in tags_seen:
        out.append(f"- `{tag}`: {tag_desc[tag]}")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Ablation of GEMM-heavy solve algorithms vs baselines",
    )
    p.add_argument("--sizes", type=str, default="256,512,1024")
    p.add_argument("--k", type=str, default="16")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--cases",
        type=str,
        default="gaussian_shifted,similarity_posspec,similarity_posspec_hard",
    )
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--timing-reps", type=int, default=5)
    p.add_argument("--timing-warmup", type=int, default=2)
    p.add_argument("--precond", type=str, default="row-norm")
    p.add_argument("--precond-ruiz-iters", type=int, default=2)
    p.add_argument("--l-target", type=float, default=0.05)
    p.add_argument("--coeff-safety", type=float, default=1.0)
    p.add_argument("--out-md", type=str, default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dtype_compute = (
        torch.float32
        if args.dtype == "fp32" or device.type != "cuda"
        else torch.bfloat16
    )

    sizes = parse_shapes(args.sizes)
    ks = parse_shapes(args.k)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    cells = _default_cells()

    # Build PE coefficients
    pe_quad_t, coeff_desc = build_pe_schedules(
        l_target=args.l_target,
        device=device,
        coeff_mode="auto",
        coeff_seed=0,
        coeff_safety=args.coeff_safety,
        coeff_no_final_safety=False,
        p_val=1,
    )
    pe_coeffs = _quad_coeffs(pe_quad_t)
    print(f"[PE coeffs] {coeff_desc} (steps={len(pe_quad_t)})")

    coupled_solve_fn = inverse_solve_pe_quadratic_coupled
    all_rows: List[Tuple[int, int, str, AlgoCell, NonSpdBenchResult]] = []

    with torch.inference_mode():
        for n in sizes:
            for k in ks:
                print(f"\n== n={n}, k={k}, dtype={dtype_compute} ==")
                for case_idx, case in enumerate(cases):
                    g = torch.Generator(device=device)
                    g.manual_seed(args.seed + 100000 * n + 1000 * k + case_idx)
                    mats = make_nonspd_cases(
                        case, n, args.trials, device, torch.float32, g
                    )
                    mats = [m.to(dtype_compute) for m in mats]

                    # Prepare + precondition
                    g_pre = torch.Generator(device=device)
                    g_pre.manual_seed(args.seed + 7919 * n + 131 * case_idx + k)
                    prepared, ms_pre = prepare_nonspd_solve_inputs(
                        mats=mats,
                        device=device,
                        k=k,
                        dtype=dtype_compute,
                        generator=g_pre,
                        precond=args.precond,
                        precond_ruiz_iters=args.precond_ruiz_iters,
                    )
                    gt = compute_ground_truth(prepared)

                    # Estimate l_min: use l_target as conservative lower bound.
                    # Gershgorin can give 0 or negative for non-SPD matrices,
                    # so we fall back to the preconditioning target.
                    l_min_est = max(float(args.l_target), 0.01)

                    print(f"  -- case={case}, l_min_est={l_min_est:.4f} --")
                    for cell in cells:
                        try:
                            rr = eval_cell(
                                prepared=prepared,
                                ground_truth=gt,
                                device=device,
                                cell=cell,
                                pe_coeffs=pe_coeffs,
                                coupled_solve_fn=coupled_solve_fn,
                                l_min_est=l_min_est,
                                pe_quad_t=pe_quad_t,
                                timing_reps=args.timing_reps,
                                timing_warmup=args.timing_warmup,
                                ms_precond=ms_pre,
                            )
                        except Exception as e:
                            print(f"    {cell.tag} {cell.algo}: ERROR - {e}")
                            rr = NonSpdBenchResult(
                                ms=float("nan"),
                                ms_iter=float("nan"),
                                ms_precond=ms_pre,
                                rel_err=float("nan"),
                                bad=len(prepared),
                                mem_alloc_mb=float("nan"),
                                mem_reserved_mb=float("nan"),
                            )
                        all_rows.append((n, k, case, cell, rr))
                        print(
                            f"    {cell.tag:<4s} {cell.algo:<22s} "
                            f"{rr.ms_iter:8.3f} ms | relerr {rr.rel_err:.3e} | bad={rr.bad}"
                        )

    report = _format_report(all_rows, cases)
    print("\n" + report)

    if args.out_md:
        out_path = Path(args.out_md)
    else:
        today = dt.date.today().strftime("%Y_%m_%d")
        out_path = Path("benchmark_results") / today / "algo_ablation" / "summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n", encoding="utf-8")
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
