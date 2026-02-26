"""
ablate_solve_inverse_ideas.py

Careful one-factor-at-a-time ablations for ideas in ideas/solve_inverse.md.
Focus: non-SPD p=1 solve path (Z = A^{-1} B) with fixed PE budget.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
)
from benchmarks.common import make_nonspd_cases, parse_shapes, maybe_compile
from benchmarks.solve.matrix_solve_nonspd import (
    NonSpdBenchResult,
    compute_ground_truth,
    eval_method,
    prepare_nonspd_solve_inputs,
)


@dataclass(frozen=True)
class AblationCell:
    tag: str
    desc: str
    method: str
    precond: str
    coeff_mode: str
    coeff_safety_mult: float
    safe_fallback_tol: Optional[float]
    safe_early_y_tol: Optional[float]


def _parse_case_csv(spec: str) -> List[str]:
    vals = [tok.strip() for tok in str(spec).split(",") if tok.strip()]
    if not vals:
        raise ValueError("Expected at least one case in --cases")
    return vals


def _default_cells(
    safe_fallback_tol: float,
    safe_early_y_tol: float,
) -> List[AblationCell]:
    return [
        AblationCell(
            tag="A0",
            desc="baseline: row-norm + tuned coeff + coupled apply",
            method="PE-Quad-Coupled-Apply",
            precond="row-norm",
            coeff_mode="tuned",
            coeff_safety_mult=1.0,
            safe_fallback_tol=None,
            safe_early_y_tol=None,
        ),
        AblationCell(
            tag="A1",
            desc="precond only: frob scaling",
            method="PE-Quad-Coupled-Apply",
            precond="frob",
            coeff_mode="tuned",
            coeff_safety_mult=1.0,
            safe_fallback_tol=None,
            safe_early_y_tol=None,
        ),
        AblationCell(
            tag="A2",
            desc="precond only: ruiz equilibration",
            method="PE-Quad-Coupled-Apply",
            precond="ruiz",
            coeff_mode="tuned",
            coeff_safety_mult=1.0,
            safe_fallback_tol=None,
            safe_early_y_tol=None,
        ),
        AblationCell(
            tag="A3",
            desc="coeff only: tuned safety x1.2 (more conservative)",
            method="PE-Quad-Coupled-Apply",
            precond="row-norm",
            coeff_mode="tuned",
            coeff_safety_mult=1.2,
            safe_fallback_tol=None,
            safe_early_y_tol=None,
        ),
        AblationCell(
            tag="A4",
            desc="safety only: final residual fallback",
            method="PE-Quad-Coupled-Apply-Safe",
            precond="row-norm",
            coeff_mode="tuned",
            coeff_safety_mult=1.0,
            safe_fallback_tol=float(safe_fallback_tol),
            safe_early_y_tol=None,
        ),
        AblationCell(
            tag="A5",
            desc="safety only: final + early fallback",
            method="PE-Quad-Coupled-Apply-Safe",
            precond="row-norm",
            coeff_mode="tuned",
            coeff_safety_mult=1.0,
            safe_fallback_tol=float(safe_fallback_tol),
            safe_early_y_tol=float(safe_early_y_tol),
        ),
        AblationCell(
            tag="A6",
            desc="adaptive runtime switch + safety",
            method="PE-Quad-Coupled-Apply-Adaptive",
            precond="row-norm",
            coeff_mode="tuned",
            coeff_safety_mult=1.0,
            safe_fallback_tol=float(safe_fallback_tol),
            safe_early_y_tol=float(safe_early_y_tol),
        ),
        AblationCell(
            tag="A7",
            desc="exact reference: torch solve",
            method="Torch-Solve",
            precond="row-norm",
            coeff_mode="tuned",
            coeff_safety_mult=1.0,
            safe_fallback_tol=None,
            safe_early_y_tol=None,
        ),
    ]


def _format_md(
    rows: List[Tuple[int, str, AblationCell, NonSpdBenchResult]],
    *,
    cases: Sequence[str],
) -> str:
    baseline: Dict[Tuple[int, str], NonSpdBenchResult] = {}
    for n, case, cell, rr in rows:
        if cell.tag == "A0":
            baseline[(n, case)] = rr

    out: List[str] = []
    out.append("# Solve-Inverse Ideas Ablation")
    out.append("")
    out.append("## Per-Case Results")
    out.append("")
    out.append(
        "| size | case | tag | method | precond | coeff | ms | relerr vs solve | delta ms vs A0 |"
    )
    out.append("|---:|---|---|---|---|---|---:|---:|---:|")
    for n, case, cell, rr in rows:
        base = baseline.get((n, case))
        delta = rr.ms - base.ms if base is not None else float("nan")
        out.append(
            f"| {n} | {case} | {cell.tag} | {cell.method} | {cell.precond} | "
            f"{cell.coeff_mode}@x{cell.coeff_safety_mult:.2f} | {rr.ms:.3f} | {rr.rel_err:.3e} | {delta:+.3f} |"
        )

    out.append("")
    out.append("## Aggregated Effects")
    out.append("")
    hard_cases = [c for c in cases if c.endswith("hard")]
    moderate_cases = [c for c in cases if c not in hard_cases]
    out.append(
        "| size | tag | moderate mean ms | moderate mean relerr | hard mean ms | hard mean relerr |"
    )
    out.append("|---:|---|---:|---:|---:|---:|")
    for n in sorted({r[0] for r in rows}):
        tags = sorted({r[2].tag for r in rows if r[0] == n})
        for tag in tags:
            rs_mod = [
                rr
                for n0, case, cell, rr in rows
                if n0 == n and cell.tag == tag and case in moderate_cases
            ]
            rs_hard = [
                rr
                for n0, case, cell, rr in rows
                if n0 == n and cell.tag == tag and case in hard_cases
            ]
            mm = (
                sum(r.ms for r in rs_mod) / len(rs_mod)
                if rs_mod
                else float("nan")
            )
            em = (
                sum(r.rel_err for r in rs_mod) / len(rs_mod)
                if rs_mod
                else float("nan")
            )
            hm = (
                sum(r.ms for r in rs_hard) / len(rs_hard)
                if rs_hard
                else float("nan")
            )
            eh = (
                sum(r.rel_err for r in rs_hard) / len(rs_hard)
                if rs_hard
                else float("nan")
            )
            out.append(
                f"| {n} | {tag} | {mm:.3f} | {em:.3e} | {hm:.3f} | {eh:.3e} |"
            )
    out.append("")
    out.append("## Cell Definitions")
    out.append("")
    for _, _, cell, _ in rows:
        if any(line.startswith(f"- `{cell.tag}`") for line in out):
            continue
        out.append(f"- `{cell.tag}`: {cell.desc}")
    out.append("")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Ablations for ideas/solve_inverse.md on non-SPD p=1 solve"
    )
    p.add_argument("--sizes", type=str, default="1024")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--cases",
        type=str,
        default="gaussian_shifted,nonnormal_upper,similarity_posspec,similarity_posspec_hard",
    )
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--timing-reps", type=int, default=5)
    p.add_argument("--timing-warmup-reps", type=int, default=2)
    p.add_argument("--precond-ruiz-iters", type=int, default=2)
    p.add_argument("--safe-fallback-tol", type=float, default=0.01)
    p.add_argument("--safe-early-y-tol", type=float, default=0.8)
    p.add_argument("--adaptive-resid-tol", type=float, default=1.0)
    p.add_argument("--adaptive-growth-tol", type=float, default=1.02)
    p.add_argument("--adaptive-check-every", type=int, default=1)
    p.add_argument("--coeff-seed", type=int, default=0)
    p.add_argument("--coeff-safety", type=float, default=1.0)
    p.add_argument("--coeff-no-final-safety", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--out-md",
        type=str,
        default="",
        help="Optional markdown summary path",
    )
    args = p.parse_args()

    if int(args.k) < 1:
        raise ValueError(f"--k must be >= 1, got {args.k}")
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
    if float(args.safe_fallback_tol) <= 0.0:
        raise ValueError(
            f"--safe-fallback-tol must be > 0, got {args.safe_fallback_tol}"
        )
    if float(args.safe_early_y_tol) <= 0.0:
        raise ValueError(
            f"--safe-early-y-tol must be > 0, got {args.safe_early_y_tol}"
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
    cells = _default_cells(
        safe_fallback_tol=float(args.safe_fallback_tol),
        safe_early_y_tol=float(args.safe_early_y_tol),
    )

    coeff_cache: Dict[Tuple[str, float], Sequence[Tuple[float, float, float]]] = {}
    for mode, safety_mult in sorted({(c.coeff_mode, c.coeff_safety_mult) for c in cells}):
        pe_quad_t, coeff_desc = build_pe_schedules(
            l_target=0.05,
            device=device,
            coeff_mode=mode,
            coeff_seed=args.coeff_seed,
            coeff_safety=float(args.coeff_safety) * float(safety_mult),
            coeff_no_final_safety=args.coeff_no_final_safety,
            p_val=1,
        )
        coeff_cache[(mode, safety_mult)] = _quad_coeffs(pe_quad_t)
        print(
            f"[coeff/{mode}@x{safety_mult:.2f}] {coeff_desc} "
            f"(steps={len(pe_quad_t)})"
        )

    uncoupled_fn = maybe_compile(inverse_proot_pe_quadratic_uncoupled, args.compile)
    coupled_solve_fn = maybe_compile(inverse_solve_pe_quadratic_coupled, args.compile)
    rows: List[Tuple[int, str, AblationCell, NonSpdBenchResult]] = []

    with torch.inference_mode():
        for n in sizes:
            print(
                f"\n== ideas-ablation non-SPD size {n}x{n} | rhs {n}x{args.k} | dtype={dtype_compute} =="
            )
            for case_idx, case in enumerate(cases):
                g_mats = torch.Generator(device=device)
                g_mats.manual_seed(int(args.seed) + 100000 * n + case_idx)
                mats = make_nonspd_cases(case, n, args.trials, device, torch.float32, g_mats)
                mats = [m.to(dtype_compute) for m in mats]

                prepared_cache = {}
                gt_cache = {}
                for precond_mode in sorted({c.precond for c in cells}):
                    g_pre = torch.Generator(device=device)
                    g_pre.manual_seed(int(args.seed) + 7919 * n + 131 * case_idx)
                    prepared_inputs, ms_precond_med = prepare_nonspd_solve_inputs(
                        mats=mats,
                        device=device,
                        k=args.k,
                        dtype=dtype_compute,
                        generator=g_pre,
                        precond=precond_mode,
                        precond_ruiz_iters=int(args.precond_ruiz_iters),
                    )
                    prepared_cache[precond_mode] = (prepared_inputs, ms_precond_med)
                    gt_cache[precond_mode] = compute_ground_truth(prepared_inputs)

                print(f"\n-- case {case} --")
                for cell in cells:
                    prepared_inputs, ms_pre = prepared_cache[cell.precond]
                    gt = gt_cache[cell.precond]
                    rr = eval_method(
                        prepared_inputs=prepared_inputs,
                        ground_truth_Z=gt,
                        device=device,
                        method=cell.method,
                        pe_coeffs=coeff_cache[(cell.coeff_mode, cell.coeff_safety_mult)],
                        timing_reps=int(args.timing_reps),
                        timing_warmup_reps=int(args.timing_warmup_reps),
                        ms_precond_median=ms_pre,
                        nonspd_adaptive_resid_tol=float(args.adaptive_resid_tol),
                        nonspd_adaptive_growth_tol=float(args.adaptive_growth_tol),
                        nonspd_adaptive_check_every=int(args.adaptive_check_every),
                        nonspd_safe_fallback_tol=cell.safe_fallback_tol,
                        nonspd_safe_early_y_tol=cell.safe_early_y_tol,
                        uncoupled_fn=uncoupled_fn,
                        coupled_solve_fn=coupled_solve_fn,
                    )
                    rows.append((n, case, cell, rr))
                    print(
                        f"{cell.tag:>2s} {cell.method:<28s} "
                        f"precond={cell.precond:<8s} "
                        f"coeff={cell.coeff_mode}@x{cell.coeff_safety_mult:.2f} "
                        f"{rr.ms:7.3f} ms | relerr {rr.rel_err:.3e}"
                    )

    md = _format_md(rows, cases=cases)
    print("\n" + md)

    if args.out_md:
        out_path = Path(args.out_md)
    else:
        today = dt.date.today().strftime("%Y_%m_%d")
        out_path = (
            Path("benchmark_results")
            / today
            / "idea_solve_inverse_ablation_t10"
            / "summary.md"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md + "\n", encoding="utf-8")
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()

