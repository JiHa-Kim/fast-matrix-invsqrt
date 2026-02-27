"""
matrix_solve_gram.py

Focused benchmark for repeated Gram-matrix inverse-root apply:
Z = (G^T G)^(-1/p) M.

Compares:
  A) reuse_precond=False  (rebuild preconditioned Gram each call)
  B) reuse_precond=True   (cache/reuse preconditioned Gram in workspace)
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from benchmarks._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from benchmarks.common import time_ms_repeat
from fast_iroot.apply import GramInverseApplyWorkspace, apply_inverse_root_gram_spd
from fast_iroot.coeffs import _quad_coeffs, build_pe_schedules
from fast_iroot.precond import GRAM_PRECOND_MODES, SPD_PRECOND_MODES


def _dtype_from_flag(dtype_flag: str, device: torch.device) -> torch.dtype:
    if str(dtype_flag) == "fp32":
        return torch.float32
    if str(dtype_flag) == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    raise ValueError(f"Unknown dtype flag: {dtype_flag}")


def _render_report(
    *,
    device: torch.device,
    dtype_compute: torch.dtype,
    p_val: int,
    m: int,
    n: int,
    k: int,
    trials: int,
    timing_reps: int,
    warmup_reps: int,
    gram_mode: str,
    precond_mode: str,
    l_target: float,
    ms_off: float,
    ms_on: float,
    rel_diff: float,
) -> str:
    speedup = ms_off / max(ms_on, 1e-12)
    lines = [
        "# Gram Reuse-Precond Benchmark",
        "",
        "## Config",
        f"- device: `{device}`",
        f"- dtype: `{dtype_compute}`",
        f"- p: `{p_val}`",
        f"- G shape: `{m}x{n}`",
        f"- M shape: `{n}x{k}`",
        f"- trials: `{trials}`",
        f"- timing_reps: `{timing_reps}`",
        f"- warmup_reps: `{warmup_reps}`",
        f"- gram_mode: `{gram_mode}`",
        f"- precond_mode: `{precond_mode}`",
        f"- l_target: `{l_target}`",
        "",
        "## Results",
        f"- A (`reuse_precond=False`): `{ms_off:.3f} ms`",
        f"- B (`reuse_precond=True`): `{ms_on:.3f} ms`",
        f"- speedup (A/B): `{speedup:.3f}x`",
        f"- output rel diff (A vs B): `{rel_diff:.3e}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark repeated Gram apply with and without precondition cache"
    )
    p.add_argument("--p", type=int, default=2, help="Root exponent p")
    p.add_argument("--m", type=int, default=4096, help="Gram source rows (samples)")
    p.add_argument("--n", type=int, default=1024, help="Gram source cols (features)")
    p.add_argument("--k", type=int, default=64, help="RHS columns")
    p.add_argument("--trials", type=int, default=20, help="Independent M draws")
    p.add_argument("--timing-reps", type=int, default=5, help="Timed reps per trial")
    p.add_argument(
        "--warmup-reps", type=int, default=2, help="Untimed warmups before timed reps"
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    p.add_argument(
        "--gram-mode", type=str, default="col-norm", choices=list(GRAM_PRECOND_MODES)
    )
    p.add_argument(
        "--precond-mode", type=str, default="jacobi", choices=list(SPD_PRECOND_MODES)
    )
    p.add_argument("--precond-ruiz-iters", type=int, default=2)
    p.add_argument("--l-target", type=float, default=0.05)
    p.add_argument("--ridge-rel", type=float, default=1e-4)
    p.add_argument("--markdown", action="store_true", help="Render markdown report")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output report path (printed to stdout when omitted)",
    )
    args = p.parse_args()

    if int(args.p) < 1:
        raise ValueError(f"--p must be >= 1, got {args.p}")
    if int(args.m) < 2 or int(args.n) < 2:
        raise ValueError(f"--m and --n must be >= 2, got m={args.m}, n={args.n}")
    if int(args.k) < 1:
        raise ValueError(f"--k must be >= 1, got {args.k}")
    if int(args.trials) < 1:
        raise ValueError(f"--trials must be >= 1, got {args.trials}")
    if int(args.timing_reps) < 1:
        raise ValueError(f"--timing-reps must be >= 1, got {args.timing_reps}")
    if int(args.warmup_reps) < 0:
        raise ValueError(f"--warmup-reps must be >= 0, got {args.warmup_reps}")
    if int(args.precond_ruiz_iters) < 1:
        raise ValueError(
            f"--precond-ruiz-iters must be >= 1, got {args.precond_ruiz_iters}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    dtype_compute = _dtype_from_flag(args.dtype, device)

    g = torch.Generator(device=device)
    g.manual_seed(int(args.seed))

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

    G = torch.randn(
        int(args.m), int(args.n), device=device, dtype=torch.float32, generator=g
    ).to(dtype_compute)

    ws_off: GramInverseApplyWorkspace | None = None
    ws_on: GramInverseApplyWorkspace | None = GramInverseApplyWorkspace()
    ms_off_list: list[float] = []
    ms_on_list: list[float] = []
    rel_diffs: list[float] = []

    with torch.inference_mode():
        for _ in range(int(args.trials)):
            M = torch.randn(
                int(args.n),
                int(args.k),
                device=device,
                dtype=dtype_compute,
                generator=g,
            )

            def _run_off() -> torch.Tensor:
                nonlocal ws_off
                z, ws_off, _ = apply_inverse_root_gram_spd(
                    G,
                    M,
                    abc_t=abc_t,
                    p_val=int(args.p),
                    ws=ws_off,
                    strategy="direct-solve",
                    expected_reuse=1,
                    reuse_precond=False,
                    gram_mode=str(args.gram_mode),
                    precond_mode=str(args.precond_mode),
                    precond_ruiz_iters=int(args.precond_ruiz_iters),
                    ridge_rel=float(args.ridge_rel),
                    l_target=float(args.l_target),
                )
                return z

            def _run_on() -> torch.Tensor:
                nonlocal ws_on
                z, ws_on, _ = apply_inverse_root_gram_spd(
                    G,
                    M,
                    abc_t=abc_t,
                    p_val=int(args.p),
                    ws=ws_on,
                    strategy="direct-solve",
                    expected_reuse=1,
                    reuse_precond=True,
                    gram_mode=str(args.gram_mode),
                    precond_mode=str(args.precond_mode),
                    precond_ruiz_iters=int(args.precond_ruiz_iters),
                    ridge_rel=float(args.ridge_rel),
                    l_target=float(args.l_target),
                )
                return z

            for _ in range(int(args.warmup_reps)):
                _ = _run_off()
                _ = _run_on()

            ms_off, z_off = time_ms_repeat(_run_off, device, reps=int(args.timing_reps))
            ms_on, z_on = time_ms_repeat(_run_on, device, reps=int(args.timing_reps))
            ms_off_list.append(float(ms_off))
            ms_on_list.append(float(ms_on))

            rel = float(
                (
                    torch.linalg.matrix_norm(z_off - z_on)
                    / torch.linalg.matrix_norm(z_off).clamp_min(1e-12)
                ).item()
            )
            rel_diffs.append(rel)

    ms_off_med = float(torch.tensor(ms_off_list).median().item())
    ms_on_med = float(torch.tensor(ms_on_list).median().item())
    rel_med = float(torch.tensor(rel_diffs).median().item())

    if args.markdown:
        report = _render_report(
            device=device,
            dtype_compute=dtype_compute,
            p_val=int(args.p),
            m=int(args.m),
            n=int(args.n),
            k=int(args.k),
            trials=int(args.trials),
            timing_reps=int(args.timing_reps),
            warmup_reps=int(args.warmup_reps),
            gram_mode=str(args.gram_mode),
            precond_mode=str(args.precond_mode),
            l_target=float(args.l_target),
            ms_off=ms_off_med,
            ms_on=ms_on_med,
            rel_diff=rel_med,
        )
    else:
        speedup = ms_off_med / max(ms_on_med, 1e-12)
        report = (
            f"[coeff] using {coeff_desc} (PE steps={len(abc_t)})\n"
            f"device={device} dtype={dtype_compute} p={int(args.p)}\n"
            f"G={int(args.m)}x{int(args.n)} M={int(args.n)}x{int(args.k)}\n"
            f"reuse_precond=False: {ms_off_med:.3f} ms\n"
            f"reuse_precond=True : {ms_on_med:.3f} ms\n"
            f"speedup (off/on): {speedup:.3f}x\n"
            f"rel diff (off vs on): {rel_med:.3e}\n"
        )

    out = str(args.out).strip()
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Wrote report to {out_path}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
