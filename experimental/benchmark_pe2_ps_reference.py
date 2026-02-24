from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from isqrt_core import (
    IsqrtWorkspace,
    _affine_coeffs,
    _alloc_ws,
    _matmul_into,
    _quad_coeffs,
    _symmetrize_inplace,
    _ws_ok,
    build_pe_schedules,
    inverse_sqrt_pe_quadratic,
    precond_spd,
)
from isqrt_metrics import compute_quality_stats
from matrix_isqrt import make_spd_cases


def median(xs: Sequence[float]) -> float:
    ys = sorted(float(x) for x in xs)
    if not ys:
        return float("nan")
    return ys[len(ys) // 2]


@torch.no_grad()
def inverse_sqrt_pe_quadratic_ps_ref(
    A_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[IsqrtWorkspace] = None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspace]:
    if not _ws_ok(ws, A_norm):
        ws = _alloc_ws(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)
    coeffs = _quad_coeffs(abc_t)

    T = len(coeffs)
    for t, (a, b, c) in enumerate(coeffs):
        _matmul_into(ws.Y, ws.Y, ws.Y2)
        ws.B.copy_(ws.Y2).mul_(c)
        ws.B.add_(ws.Y, alpha=b)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(a)

        _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        if terminal_last_step and (t == T - 1):
            break

        a2 = a * a
        b2 = b * b
        c2 = c * c
        ab2 = 2.0 * a * b
        bc2 = 2.0 * b * c
        b2_2ac = b2 + 2.0 * a * c

        ws.Ybuf.copy_(ws.Y).mul_(a2)
        ws.Ybuf.add_(ws.Y2, alpha=ab2)

        ws.B2.copy_(ws.Y2).mul_(c2)
        ws.B2.add_(ws.Y, alpha=bc2)
        ws.B2.diagonal(dim1=-2, dim2=-1).add_(b2_2ac)

        _matmul_into(ws.Y2, ws.Y, ws.B)
        _matmul_into(ws.B, ws.B2, ws.Y)
        ws.Y.add_(ws.Ybuf)

        if symmetrize_Y:
            _symmetrize_inplace(ws.Y, ws.B)

    return ws.X, ws


def time_ms_repeat(
    fn: Callable[[], torch.Tensor], device: torch.device, reps: int
) -> Tuple[float, torch.Tensor]:
    reps_i = max(1, int(reps))
    if device.type == "cuda":
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out: Optional[torch.Tensor] = None
        for _ in range(reps_i):
            out = fn()
        e.record()
        torch.cuda.synchronize()
        assert out is not None
        return float(s.elapsed_time(e)) / float(reps_i), out

    t0 = time.perf_counter()
    out = None
    for _ in range(reps_i):
        out = fn()
    assert out is not None
    return 1000.0 * (time.perf_counter() - t0) / float(reps_i), out


@dataclass
class Row:
    ms: float
    residual: float
    hard: float
    sym_x: float
    sym_w: float
    bad: int


@torch.no_grad()
def eval_runner(
    mats: List[torch.Tensor],
    runner: Callable[[torch.Tensor, Optional[IsqrtWorkspace]], Tuple[torch.Tensor, IsqrtWorkspace]],
    device: torch.device,
    precond: str,
    ridge_rel: float,
    l_target: float,
    timing_reps: int,
    power_iters: int,
    mv_samples: int,
    hard_probe_iters: int,
) -> Row:
    ms_list: List[float] = []
    resid_list: List[float] = []
    hard_list: List[float] = []
    symx_list: List[float] = []
    symw_list: List[float] = []
    bad = 0
    ws: Optional[IsqrtWorkspace] = None
    eye: Optional[torch.Tensor] = None

    for A in mats:
        A_norm, _ = precond_spd(A, mode=precond, ridge_rel=ridge_rel, l_target=l_target)
        n = A_norm.shape[-1]
        if eye is None or eye.shape != (n, n) or eye.device != A_norm.device:
            eye = torch.eye(n, device=A_norm.device, dtype=torch.float32)

        def run_once() -> torch.Tensor:
            nonlocal ws
            Xn, ws2 = runner(A_norm, ws)
            ws = ws2
            return Xn

        ms, Xn = time_ms_repeat(run_once, device, timing_reps)
        ms_list.append(ms)

        if not torch.isfinite(Xn).all():
            bad += 1
            resid_list.append(float("inf"))
            hard_list.append(float("inf"))
            symx_list.append(float("inf"))
            symw_list.append(float("inf"))
            continue

        q = compute_quality_stats(
            Xn,
            A_norm,
            power_iters=power_iters,
            mv_samples=mv_samples,
            hard_probe_iters=hard_probe_iters,
            eye_mat=eye,
        )
        resid_list.append(q.residual_fro)
        hard_list.append(q.hard_dir_err)
        symx_list.append(q.sym_x)
        symw_list.append(q.sym_w)

    return Row(
        ms=median(ms_list),
        residual=median(resid_list),
        hard=median(hard_list),
        sym_x=median(symx_list),
        sym_w=median(symw_list),
        bad=bad,
    )


def parse_sizes(s: str) -> List[int]:
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("empty sizes")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Rigorous PE2 vs PE2-PS reference benchmark")
    p.add_argument("--sizes", type=str, default="256,512,1024")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--timing-reps", type=int, default=40)
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--precond", type=str, default="aol", choices=["none", "frob", "aol"])
    p.add_argument("--ridge-rel", type=float, default=1e-4)
    p.add_argument("--l-target", type=float, default=0.05)
    p.add_argument("--coeff-mode", type=str, default="auto", choices=["auto", "precomputed", "tuned"])
    p.add_argument("--coeff-seed", type=int, default=0)
    p.add_argument("--coeff-safety", type=float, default=1.0)
    p.add_argument("--coeff-no-final-safety", action="store_true")
    p.add_argument("--power-iters", type=int, default=8)
    p.add_argument("--mv-samples", type=int, default=8)
    p.add_argument("--hard-probe-iters", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dtype_compute = (
        torch.float32 if (args.dtype == "fp32" or device.type != "cuda") else torch.bfloat16
    )
    sizes = parse_sizes(args.sizes)
    cases = ["gaussian_spd", "illcond_1e6", "illcond_1e12", "near_rank_def", "spike"]

    pe_aff, pe2, coeff_desc = build_pe_schedules(
        l_target=args.l_target,
        device=device,
        coeff_mode=args.coeff_mode,
        coeff_seed=args.coeff_seed,
        coeff_safety=args.coeff_safety,
        coeff_no_final_safety=args.coeff_no_final_safety,
    )
    _ = _affine_coeffs(pe_aff)  # keep schedule path aligned with harness
    pe2_coeffs = _quad_coeffs(pe2)
    print(f"[coeff] using {coeff_desc}")
    print(
        f"[setup] device={device.type} dtype={dtype_compute} trials={args.trials} warmup={args.warmup} "
        f"timing_reps={args.timing_reps} power_it={args.power_iters} mv_k={args.mv_samples} hard_it={args.hard_probe_iters}"
    )

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    with torch.inference_mode():
        for n in sizes:
            print(f"== size {n}x{n} ==")

            warm = make_spd_cases("gaussian_spd", n, max(1, args.warmup), device, torch.float32, g)
            for A in warm:
                A = A.to(dtype_compute)
                A_norm, _ = precond_spd(A, mode=args.precond, ridge_rel=args.ridge_rel, l_target=args.l_target)
                inverse_sqrt_pe_quadratic(A_norm, abc_t=pe2_coeffs, ws=None, symmetrize_Y=True, terminal_last_step=True)
                inverse_sqrt_pe_quadratic_ps_ref(A_norm, abc_t=pe2_coeffs, ws=None, symmetrize_Y=True, terminal_last_step=True)

            for case in cases:
                mats = make_spd_cases(case, n, args.trials, device, torch.float32, g)
                mats = [m.to(dtype_compute) for m in mats]

                r_pe2 = eval_runner(
                    mats,
                    lambda A, ws: inverse_sqrt_pe_quadratic(
                        A, abc_t=pe2_coeffs, ws=ws, symmetrize_Y=True, terminal_last_step=True
                    ),
                    device=device,
                    precond=args.precond,
                    ridge_rel=args.ridge_rel,
                    l_target=args.l_target,
                    timing_reps=args.timing_reps,
                    power_iters=args.power_iters,
                    mv_samples=args.mv_samples,
                    hard_probe_iters=args.hard_probe_iters,
                )
                r_ps = eval_runner(
                    mats,
                    lambda A, ws: inverse_sqrt_pe_quadratic_ps_ref(
                        A, abc_t=pe2_coeffs, ws=ws, symmetrize_Y=True, terminal_last_step=True
                    ),
                    device=device,
                    precond=args.precond,
                    ridge_rel=args.ridge_rel,
                    l_target=args.l_target,
                    timing_reps=args.timing_reps,
                    power_iters=args.power_iters,
                    mv_samples=args.mv_samples,
                    hard_probe_iters=args.hard_probe_iters,
                )
                print(f"-- case {case} --")
                print(
                    f"PE2     {r_pe2.ms:8.3f} ms | resid {r_pe2.residual:.3e} | hard {r_pe2.hard:.3e} | "
                    f"symX {r_pe2.sym_x:.2e} symW {r_pe2.sym_w:.2e} | bad {r_pe2.bad}"
                )
                print(
                    f"PE2-PS  {r_ps.ms:8.3f} ms | resid {r_ps.residual:.3e} | hard {r_ps.hard:.3e} | "
                    f"symX {r_ps.sym_x:.2e} symW {r_ps.sym_w:.2e} | bad {r_ps.bad}"
                )
                print(
                    f"delta   {(r_ps.ms - r_pe2.ms):+8.3f} ms | resid x{(r_ps.residual / r_pe2.residual):.3f} | "
                    f"hard x{(r_ps.hard / r_pe2.hard):.3f}"
                )
                print()


if __name__ == "__main__":
    main()
