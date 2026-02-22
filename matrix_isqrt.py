"""
matrix_isqrt.py

Matmul-only inverse-square-root benchmark harness.
Core iteration/preconditioning code lives in isqrt_core.py.
Metrics code lives in isqrt_metrics.py.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from isqrt_core import (
    AutoPolicyConfig,
    IsqrtWorkspace,
    PrecondStats,
    _affine_coeffs,
    _quad_coeffs,
    build_pe_schedules,
    choose_auto_method,
    inverse_sqrt_ns,
    inverse_sqrt_pe_affine,
    inverse_sqrt_pe_quadratic,
    precond_spd,
)
from isqrt_metrics import compute_quality_stats, isqrt_relative_error


def median(xs: Sequence[float]) -> float:
    ys = sorted(float(x) for x in xs)
    if not ys:
        return float("nan")
    return ys[len(ys) // 2]


def pctl(xs: Sequence[float], q: float) -> float:
    ys = sorted(float(x) for x in xs)
    if not ys:
        return float("nan")
    qf = max(0.0, min(1.0, float(q)))
    idx = int(round(qf * (len(ys) - 1)))
    return float(ys[idx])


def time_ms(
    fn: Callable[[], torch.Tensor], device: torch.device
) -> Tuple[float, torch.Tensor]:
    if device.type == "cuda":
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        return float(s.elapsed_time(e)), out
    t0 = time.perf_counter()
    out = fn()
    return 1000.0 * (time.perf_counter() - t0), out


def time_ms_any(fn: Callable[[], object], device: torch.device) -> Tuple[float, object]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0), out


def time_ms_repeat(
    fn: Callable[[], torch.Tensor], device: torch.device, reps: int = 1
) -> Tuple[float, torch.Tensor]:
    reps_i = max(1, int(reps))
    if reps_i == 1:
        return time_ms(fn, device)

    def run_many() -> torch.Tensor:
        out: Optional[torch.Tensor] = None
        for _ in range(reps_i):
            out = fn()
        assert out is not None
        return out

    ms_total, out = time_ms(run_many, device)
    return ms_total / float(reps_i), out


def _qr_orthonormal(
    n: int, device: torch.device, dtype: torch.dtype, g: torch.Generator
) -> torch.Tensor:
    Q, _ = torch.linalg.qr(
        torch.randn(n, n, device=device, dtype=dtype, generator=g), mode="reduced"
    )
    return Q


def _spd_from_eigs(
    eigs: torch.Tensor, device: torch.device, dtype: torch.dtype, g: torch.Generator
) -> torch.Tensor:
    n = int(eigs.numel())
    Q = _qr_orthonormal(n, device, dtype, g)
    return (Q * eigs.unsqueeze(0)) @ Q.mT


def make_spd_cases(
    case: str,
    n: int,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    g: torch.Generator,
) -> List[torch.Tensor]:
    mats: List[torch.Tensor] = []
    for _ in range(int(count)):
        if case == "gaussian_spd":
            X = torch.randn(n, n, device=device, dtype=dtype, generator=g)
            A = (X @ X.mT) / float(n)
            A.diagonal().add_(1e-3)
        elif case == "illcond_1e6":
            e = torch.logspace(0.0, -6.0, steps=n, device=device, dtype=dtype)
            A = _spd_from_eigs(e, device, dtype, g)
        elif case == "illcond_1e12":
            e = torch.logspace(0.0, -12.0, steps=n, device=device, dtype=dtype)
            A = _spd_from_eigs(e, device, dtype, g)
        elif case == "near_rank_def":
            e = torch.logspace(
                0.0, -16.0, steps=n, device=device, dtype=dtype
            ).clamp_min(1e-12)
            A = _spd_from_eigs(e, device, dtype, g)
        elif case == "spike":
            e = torch.ones(n, device=device, dtype=dtype)
            e[0] = 1e3
            A = _spd_from_eigs(e, device, dtype, g)
        else:
            raise ValueError(f"unknown case: {case}")
        mats.append(A)
    return mats


@dataclass
class BenchResult:
    ms: float
    ms_iter: float
    ms_precond: float
    residual: float
    residual_p95: float
    residual_max: float
    residual_spec: float
    sym_x: float
    sym_w: float
    mv_err: float
    hard_dir: float
    relerr: float
    bad: int


@dataclass
class PreparedInput:
    A_norm: torch.Tensor
    stats: PrecondStats


@torch.no_grad()
def prepare_preconditioned_inputs(
    mats: List[torch.Tensor],
    device: torch.device,
    precond: str,
    ridge_rel: float,
    l_target: float,
) -> Tuple[List[PreparedInput], float]:
    prepared: List[PreparedInput] = []
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
        prepared.append(PreparedInput(A_norm=A_norm, stats=stats))
    return prepared, (median(ms_pre_list) if ms_pre_list else float("nan"))


def parse_shapes(spec: str) -> List[int]:
    vals: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    if not vals:
        raise ValueError("empty shape list")
    return vals


def maybe_compile(fn, enabled: bool):
    if not enabled:
        return fn
    try:
        return torch.compile(fn, mode="max-autotune", fullgraph=False)
    except Exception:
        return fn


def _build_runner(
    method: str,
    n: int,
    stats: PrecondStats,
    auto_cfg: AutoPolicyConfig,
    pe_affine_coeffs: Sequence[Tuple[float, float]],
    pe_quad_coeffs: Sequence[Tuple[float, float, float]],
    symmetrize_Y: bool,
) -> Callable[
    [torch.Tensor, Optional[IsqrtWorkspace]], Tuple[torch.Tensor, IsqrtWorkspace]
]:
    """
    Returns a function runner(A_norm, ws) -> (Xn, ws2) with method choice fixed
    (AUTO is resolved using n/stats/auto_cfg).
    """
    chosen = method
    if method == "AUTO":
        chosen = choose_auto_method(n, stats, auto_cfg)

    if chosen == "NS3":

        def run(A_norm: torch.Tensor, ws: Optional[IsqrtWorkspace]):
            return inverse_sqrt_ns(
                A_norm,
                iters=3,
                ws=ws,
                symmetrize_Y=symmetrize_Y,
                terminal_last_step=True,
            )

        return run

    if chosen == "NS4":

        def run(A_norm: torch.Tensor, ws: Optional[IsqrtWorkspace]):
            return inverse_sqrt_ns(
                A_norm,
                iters=4,
                ws=ws,
                symmetrize_Y=symmetrize_Y,
                terminal_last_step=True,
            )

        return run

    if chosen == "PE-NS3":

        def run(A_norm: torch.Tensor, ws: Optional[IsqrtWorkspace]):
            return inverse_sqrt_pe_affine(
                A_norm,
                ab_t=pe_affine_coeffs,
                ws=ws,
                symmetrize_Y=symmetrize_Y,
                terminal_last_step=True,
            )

        return run

    if chosen == "PE2":

        def run(A_norm: torch.Tensor, ws: Optional[IsqrtWorkspace]):
            return inverse_sqrt_pe_quadratic(
                A_norm,
                abc_t=pe_quad_coeffs,
                ws=ws,
                symmetrize_Y=symmetrize_Y,
                terminal_last_step=True,
            )

        return run

    raise ValueError(f"unknown method: {method} (resolved to {chosen})")


@torch.no_grad()
def eval_method(
    prepared_inputs: List[PreparedInput],
    ms_precond_median: float,
    device: torch.device,
    method: str,
    pe_affine_coeffs: Sequence[Tuple[float, float]],
    pe_quad_coeffs: Sequence[Tuple[float, float, float]],
    auto_cfg: AutoPolicyConfig,
    timing_reps: int,
    symmetrize_Y: bool,
    compute_relerr: bool,
    power_iters: int,
    mv_samples: int,
    hard_probe_iters: int,
) -> BenchResult:
    ms_iter_list: List[float] = []
    res_list: List[float] = []
    res2_list: List[float] = []
    symx_list: List[float] = []
    symw_list: List[float] = []
    mv_list: List[float] = []
    hard_list: List[float] = []
    err_list: List[float] = []
    bad = 0

    if len(prepared_inputs) == 0:
        return BenchResult(
            ms=float("nan"),
            ms_iter=float("nan"),
            ms_precond=float("nan"),
            residual=float("nan"),
            residual_p95=float("nan"),
            residual_max=float("nan"),
            residual_spec=float("nan"),
            sym_x=float("nan"),
            sym_w=float("nan"),
            mv_err=float("nan"),
            hard_dir=float("nan"),
            relerr=float("nan"),
            bad=0,
        )

    ws: Optional[IsqrtWorkspace] = None
    eye_mat: Optional[torch.Tensor] = None

    for prep in prepared_inputs:
        A_norm = prep.A_norm
        stats = prep.stats
        n = A_norm.shape[-1]

        # FIX: robust, avoids comparing against A_norm.shape and avoids rebuilding per trial.
        if (
            eye_mat is None
            or eye_mat.shape != (n, n)
            or eye_mat.device != A_norm.device
        ):
            eye_mat = torch.eye(n, device=A_norm.device, dtype=torch.float32)

        runner = _build_runner(
            method=method,
            n=n,
            stats=stats,
            auto_cfg=auto_cfg,
            pe_affine_coeffs=pe_affine_coeffs,
            pe_quad_coeffs=pe_quad_coeffs,
            symmetrize_Y=symmetrize_Y,
        )

        def run_once() -> torch.Tensor:
            nonlocal ws
            Xn, ws2 = runner(A_norm, ws)
            ws = ws2
            return Xn

        ms_iter, Xn = time_ms_repeat(run_once, device, reps=timing_reps)
        ms_iter_list.append(ms_iter)

        if not torch.isfinite(Xn).all():
            bad += 1
            res_list.append(float("inf"))
            res2_list.append(float("inf"))
            symx_list.append(float("inf"))
            symw_list.append(float("inf"))
            mv_list.append(float("inf"))
            hard_list.append(float("inf"))
            err_list.append(float("inf"))
            continue

        q = compute_quality_stats(
            Xn,
            A_norm,
            power_iters=power_iters,
            mv_samples=mv_samples,
            hard_probe_iters=hard_probe_iters,
            eye_mat=eye_mat,
        )
        res_list.append(q.residual_fro)
        res2_list.append(q.residual_spec)
        symx_list.append(q.sym_x)
        symw_list.append(q.sym_w)
        mv_list.append(q.mv_err)
        hard_list.append(q.hard_dir_err)

        if compute_relerr:
            # isqrt_relative_error returns per-batch (here scalar) tensor; take item safely.
            err_list.append(
                float(isqrt_relative_error(Xn.float(), A_norm.float()).mean().item())
            )
        else:
            err_list.append(float("nan"))

    ms_iter_med = median(ms_iter_list)
    ms_pre_med = ms_precond_median

    return BenchResult(
        ms=ms_pre_med + ms_iter_med,
        ms_iter=ms_iter_med,
        ms_precond=ms_pre_med,
        residual=median(res_list),
        residual_p95=pctl(res_list, 0.95),
        residual_max=max(float(x) for x in res_list) if res_list else float("nan"),
        residual_spec=median(res2_list),
        sym_x=median(symx_list),
        sym_w=median(symw_list),
        mv_err=median(mv_list),
        hard_dir=median(hard_list),
        relerr=median(err_list),
        bad=bad,
    )


def main():
    p = argparse.ArgumentParser(
        description="Benchmark inverse-square-root iterations (NS vs PE schedules)"
    )
    p.add_argument("--sizes", type=str, default="256,512,1024")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--compile", action="store_true")
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
    p.add_argument("--n-switch", type=int, default=768)
    p.add_argument("--rho-switch", type=float, default=4.0)
    p.add_argument(
        "--auto-policy",
        type=str,
        default="hybrid",
        choices=["size_rho", "interval", "hybrid"],
    )
    p.add_argument("--kappa-ns3-max", type=float, default=32.0)
    p.add_argument("--kappa-pe2-min", type=float, default=96.0)
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
    p.add_argument("--metrics-mode", type=str, default="full", choices=["full", "fast"])
    p.add_argument("--power-iters", type=int, default=0)
    p.add_argument("--mv-samples", type=int, default=0)
    p.add_argument("--hard-probe-iters", type=int, default=0)
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
    cases = ["gaussian_spd", "illcond_1e6", "illcond_1e12", "near_rank_def", "spike"]

    pe_ns3_005, pe2_005, coeff_desc = build_pe_schedules(
        l_target=args.l_target,
        device=device,
        coeff_mode=args.coeff_mode,
        coeff_seed=args.coeff_seed,
        coeff_safety=args.coeff_safety,
        coeff_no_final_safety=args.coeff_no_final_safety,
    )
    print(f"[coeff] using {coeff_desc}")
    pe_affine_coeffs = _affine_coeffs(pe_ns3_005)
    pe_quad_coeffs = _quad_coeffs(pe2_005)

    auto_cfg = AutoPolicyConfig(
        policy=args.auto_policy,
        n_switch=args.n_switch,
        rho_switch=args.rho_switch,
        kappa_ns3_max=args.kappa_ns3_max,
        kappa_pe2_min=args.kappa_pe2_min,
    )
    print(
        f"[auto] policy={auto_cfg.policy} n_switch={auto_cfg.n_switch} "
        f"rho_switch={auto_cfg.rho_switch} kappa_ns3_max={auto_cfg.kappa_ns3_max} "
        f"kappa_pe2_min={auto_cfg.kappa_pe2_min}"
    )

    # Compile (store compiled fns back into module globals so _build_runner uses them).
    if args.compile:
        globals()["inverse_sqrt_ns"] = maybe_compile(inverse_sqrt_ns, True)
        globals()["inverse_sqrt_pe_affine"] = maybe_compile(
            inverse_sqrt_pe_affine, True
        )
        globals()["inverse_sqrt_pe_quadratic"] = maybe_compile(
            inverse_sqrt_pe_quadratic, True
        )

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)

    with torch.inference_mode():
        for n in sizes:
            print(
                f"== SPD size {n}x{n} | dtype={dtype_compute} | compile={args.compile} | "
                f"precond={args.precond} | l_target={args.l_target} | lmax=row_sum | terminal=True | "
                f"timing_reps={max(1, args.timing_reps)} | symY={not args.no_symmetrize_y} | "
                f"auto={args.auto_policy} | metrics={args.metrics_mode} | power_it={args.power_iters} | "
                f"mv_k={args.mv_samples} | hard_it={args.hard_probe_iters} =="
            )

            # Warmup
            warm = make_spd_cases(
                "gaussian_spd", n, max(1, args.warmup), device, torch.float32, g
            )
            ws: Optional[IsqrtWorkspace] = None
            for A in warm:
                A = A.to(dtype_compute)
                A_norm, _ = precond_spd(
                    A,
                    mode=args.precond,
                    ridge_rel=args.ridge_rel,
                    l_target=args.l_target,
                )
                _, ws = inverse_sqrt_ns(
                    A_norm,
                    iters=2,
                    ws=ws,
                    symmetrize_Y=not args.no_symmetrize_y,
                    terminal_last_step=True,
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
                for name in ["NS3", "NS4", "PE-NS3", "PE2", "AUTO"]:
                    rr = eval_method(
                        prepared_inputs=prepared_inputs,
                        ms_precond_median=ms_precond_med,
                        device=device,
                        method=name,
                        pe_affine_coeffs=pe_affine_coeffs,
                        pe_quad_coeffs=pe_quad_coeffs,
                        auto_cfg=auto_cfg,
                        timing_reps=args.timing_reps,
                        symmetrize_Y=not args.no_symmetrize_y,
                        compute_relerr=(args.metrics_mode == "full"),
                        power_iters=args.power_iters,
                        mv_samples=args.mv_samples,
                        hard_probe_iters=args.hard_probe_iters,
                    )
                    rows.append((name, rr))

                print(f"-- case {case} --")
                for name, rr in rows:
                    print(
                        f"{name:<10s} {rr.ms:8.3f} ms (pre {rr.ms_precond:.3f} + iter {rr.ms_iter:.3f}) | "
                        f"resid {rr.residual:.3e} p95 {rr.residual_p95:.3e} max {rr.residual_max:.3e} | "
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
                        f"BEST<=target({args.target_metric}={args.target_resid:.3g}): "
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
                        print(
                            "BEST overall: none (all runs produced non-finite output)"
                        )
                print()


if __name__ == "__main__":
    main()
