"""
matrix_solve_gram_rhs.py

Benchmark harness for Gram-RHS apply:
Z = (G^T G)^(-1/p) G^T B

Compares:
  - PE-Quad-Coupled-Apply-Primal-Gram: build M = G^T B, then primal Gram apply.
  - PE-Quad-Coupled-Apply-Dual-Gram-RHS: dual Gram RHS apply directly from B.
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

from benchmarks.common import parse_shapes, time_ms_repeat
from fast_iroot import (
    DUAL_GRAM_PRECOND_MODES,
    GRAM_PRECOND_MODES,
    SPD_PRECOND_MODES,
    DualGramInverseApplyWorkspace,
    GramInverseApplyWorkspace,
    _quad_coeffs,
    apply_inverse_root_gram_rhs_spd,
    apply_inverse_root_gram_spd,
    build_pe_schedules,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Benchmark Gram RHS apply: Z=(G^T G)^(-1/p)G^T B "
            "via primal Gram vs dual Gram-RHS paths."
        )
    )
    p.add_argument("--p", type=int, default=2, help="Root exponent p")
    p.add_argument("--m", type=int, default=256, help="Rows of G (batch/sample dim)")
    p.add_argument("--n", type=int, default=1024, help="Cols of G (feature dim)")
    p.add_argument("--k", type=str, default="1,16,64", help="RHS columns (CSV)")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--timing-reps", type=int, default=5)
    p.add_argument("--timing-warmup-reps", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    p.add_argument(
        "--primal-gram-mode",
        type=str,
        default="col-norm",
        choices=list(GRAM_PRECOND_MODES),
    )
    p.add_argument(
        "--dual-gram-mode",
        type=str,
        default="row-norm",
        choices=list(DUAL_GRAM_PRECOND_MODES),
    )
    p.add_argument(
        "--precond", type=str, default="jacobi", choices=list(SPD_PRECOND_MODES)
    )
    p.add_argument("--precond-ruiz-iters", type=int, default=2)
    p.add_argument("--l-target", type=float, default=0.05)
    p.add_argument("--ridge-rel", type=float, default=1e-4)
    args = p.parse_args()

    if int(args.p) < 1:
        raise ValueError(f"--p must be >= 1, got {args.p}")
    if int(args.m) < 2 or int(args.n) < 2:
        raise ValueError(f"--m and --n must be >= 2, got m={args.m}, n={args.n}")
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

    ks = parse_shapes(args.k)
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

    pe_quad_t, coeff_desc = build_pe_schedules(
        l_target=float(args.l_target),
        device=device,
        coeff_mode="auto",
        coeff_seed=0,
        coeff_safety=1.0,
        coeff_no_final_safety=False,
        p_val=int(args.p),
    )
    abc_t = _quad_coeffs(pe_quad_t)

    g = torch.Generator(device=device)
    g.manual_seed(int(args.seed))
    G = torch.randn(
        int(args.m), int(args.n), device=device, dtype=torch.float32, generator=g
    ).to(dtype_compute)

    print(f"[coeff] using {coeff_desc} (PE steps = {len(abc_t)})")

    with torch.inference_mode():
        for k in ks:
            print(
                f"\n== SPD Size {int(args.n)}x{int(args.n)} | RHS {int(args.n)}x{k} | "
                f"dtype={dtype_compute} =="
            )
            print(
                f"precond={args.precond} | l_target={args.l_target} | p={int(args.p)} | "
                f"ruiz_iters={int(args.precond_ruiz_iters)} | m={int(args.m)} | "
                f"primal_gram_mode={args.primal_gram_mode} | "
                f"dual_gram_mode={args.dual_gram_mode}"
            )
            print("\n-- case gram_rhs_gtb --")

            ws_primal: GramInverseApplyWorkspace | None = GramInverseApplyWorkspace()
            ws_dual: DualGramInverseApplyWorkspace | None = (
                DualGramInverseApplyWorkspace()
            )
            primal_ms: list[float] = []
            dual_ms: list[float] = []
            rel_diffs: list[float] = []

            for _ in range(int(args.trials)):
                B = torch.randn(
                    int(args.m), int(k), device=device, dtype=dtype_compute, generator=g
                )

                def _run_primal() -> torch.Tensor:
                    nonlocal ws_primal
                    M = G.mT @ B
                    z, ws_primal, _ = apply_inverse_root_gram_spd(
                        G,
                        M,
                        abc_t=abc_t,
                        p_val=int(args.p),
                        ws=ws_primal,
                        strategy="direct-solve",
                        expected_reuse=1,
                        reuse_precond=True,
                        gram_mode=str(args.primal_gram_mode),
                        precond_mode=str(args.precond),
                        precond_ruiz_iters=int(args.precond_ruiz_iters),
                        ridge_rel=float(args.ridge_rel),
                        l_target=float(args.l_target),
                    )
                    return z

                def _run_dual() -> torch.Tensor:
                    nonlocal ws_dual
                    z, ws_dual, _ = apply_inverse_root_gram_rhs_spd(
                        G,
                        B,
                        abc_t=abc_t,
                        p_val=int(args.p),
                        ws=ws_dual,
                        strategy="direct-solve",
                        expected_reuse=1,
                        reuse_precond=True,
                        gram_mode=str(args.dual_gram_mode),
                        precond_mode=str(args.precond),
                        precond_ruiz_iters=int(args.precond_ruiz_iters),
                        ridge_rel=float(args.ridge_rel),
                        l_target=float(args.l_target),
                    )
                    return z

                for _ in range(int(args.timing_warmup_reps)):
                    _ = _run_primal()
                    _ = _run_dual()

                ms_p, z_p = time_ms_repeat(_run_primal, device, reps=int(args.timing_reps))
                ms_d, z_d = time_ms_repeat(_run_dual, device, reps=int(args.timing_reps))
                primal_ms.append(float(ms_p))
                dual_ms.append(float(ms_d))
                rel = float(
                    (
                        torch.linalg.matrix_norm(z_p - z_d)
                        / torch.linalg.matrix_norm(z_d).clamp_min(1e-12)
                    ).item()
                )
                rel_diffs.append(rel)

            primal_ms_med = float(torch.tensor(primal_ms).median().item())
            dual_ms_med = float(torch.tensor(dual_ms).median().item())
            rel_med = float(torch.tensor(rel_diffs).median().item())

            for method_name, total_ms in (
                ("PE-Quad-Coupled-Apply-Primal-Gram", primal_ms_med),
                ("PE-Quad-Coupled-Apply-Dual-Gram-RHS", dual_ms_med),
            ):
                print(
                    f"{method_name:<36s} {total_ms:8.3f} ms "
                    f"(pre {0.000:.3f} + iter {total_ms:.3f}) | "
                    f"relerr vs solve: {rel_med:.3e}"
                )

            speedup = primal_ms_med / max(dual_ms_med, 1e-12)
            print(f"speedup dual_vs_primal: {speedup:.3f}x")
            if any(not math.isfinite(x) for x in (primal_ms_med, dual_ms_med, rel_med)):
                print("WARNING: non-finite benchmark statistic detected")


if __name__ == "__main__":
    main()

