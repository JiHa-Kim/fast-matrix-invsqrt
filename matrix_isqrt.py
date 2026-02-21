"""
isqrt_bench_full.py

Matmul-only inverse-square-root iterations for SPD matrices:
- NS (coupled Newton-Schulz): B = 1.5 I - 0.5 Y
- PE-quadratic schedule (your "quintic-in-X" variant): B = a I + b Y + c Y^2
- PE-affine schedule ("PE-NS"): B = a I + b Y   (same GEMM count as NS)
- AUTO policy: chooses among NS3, PE-NS3, PE2 based on size and a cheap proxy.

Critical preconditioning:
- AOL diagonal similarity scaling (optional)
- ridge
- normalize lambda_max <= 1 via max row sum bound
- enforce floor lambda_min >= l_target using Gershgorin lower bound + diagonal shift, then renormalize

This means you are effectively computing inverse sqrt of a damped matrix, which is what you want for ML preconditioning.

Example:
  python isqrt_bench_full.py --sizes 256,512,1024 --dtype bf16 --trials 8 --ridge-rel 1e-4 --l-target 0.05 --target-resid 0.01
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch


# ---------------------------
# Helpers
# ---------------------------


def median(xs: Sequence[float]) -> float:
    ys = sorted(float(x) for x in xs)
    return ys[len(ys) // 2]


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


# ---------------------------
# Workspace
# ---------------------------


@dataclass
class IsqrtWorkspace:
    X: torch.Tensor
    Xbuf: torch.Tensor
    Y: torch.Tensor
    Ybuf: torch.Tensor
    Y2: torch.Tensor
    B: torch.Tensor
    B2: torch.Tensor
    eye_mat: torch.Tensor


def _alloc_ws(A: torch.Tensor) -> IsqrtWorkspace:
    shape = A.shape
    n = shape[-1]
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A).contiguous()
    return IsqrtWorkspace(
        X=A.new_empty(shape),
        Xbuf=A.new_empty(shape),
        Y=A.new_empty(shape),
        Ybuf=A.new_empty(shape),
        Y2=A.new_empty(shape),
        B=A.new_empty(shape),
        B2=A.new_empty(shape),
        eye_mat=eye,
    )


def _ws_ok(ws: Optional[IsqrtWorkspace], A: torch.Tensor) -> bool:
    if ws is None:
        return False
    return ws.X.device == A.device and ws.X.dtype == A.dtype and ws.X.shape == A.shape


@torch.no_grad()
def _symmetrize_inplace(M: torch.Tensor) -> None:
    M.copy_(0.5 * (M + M.mT))


# ---------------------------
# Preconditioning
# ---------------------------


@torch.no_grad()
def precond_spd(
    A: torch.Tensor,
    mode: str,
    eps: float = 1e-12,
    ridge_rel: float = 0.0,
    l_target: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (A_norm, rho_proxy).

    A_norm: normalized SPD-ish matrix with spectrum roughly in [l_target, 1].
    rho_proxy: cheap proxy for spikiness: max_row_sum(|A_norm|) / mean_diag(A_norm).
               After normalization, max_row_sum is ~1, so rho ~ 1/mean_diag.

    Note: enforcing a floor implies a diagonal shift; this is intentional damping.
    """
    n = A.shape[-1]

    if mode == "none":
        A_pre = A
    elif mode == "frob":
        s = torch.linalg.matrix_norm(A, ord="fro")
        s = torch.clamp(s / math.sqrt(n), min=eps)
        A_pre = A / s.unsqueeze(-1).unsqueeze(-1)
    elif mode == "aol":
        d = torch.rsqrt(A.abs().sum(dim=-1).clamp_min(eps))
        A_pre = (d.unsqueeze(-1) * A) * d.unsqueeze(-2)
    else:
        raise ValueError(f"unknown preconditioner: {mode}")

    if ridge_rel > 0.0:
        diag_mean = A_pre.diagonal(dim1=-2, dim2=-1).mean(dim=-1).abs()
        lam = torch.clamp(ridge_rel * diag_mean, min=eps)
        A_pre = A_pre.clone()
        A_pre.diagonal(dim1=-2, dim2=-1).add_(lam.unsqueeze(-1))

    # Upper-bound normalize via max row sum (Gershgorin-ish): lambda_max <= 1
    abs_row_sum = A_pre.abs().sum(dim=-1)
    u = abs_row_sum.max(dim=-1)[0].clamp_min(eps)
    A_norm = A_pre / u.unsqueeze(-1).unsqueeze(-1)

    # Enforce a lower spectral floor via Gershgorin lower bound + diagonal shift, then renormalize top.
    if l_target > 0.0:
        abs_row_sum2 = A_norm.abs().sum(dim=-1)
        diag = A_norm.diagonal(dim1=-2, dim2=-1)
        off = abs_row_sum2 - diag.abs()
        g_lo = (diag - off).min(dim=-1)[0]
        shift = (float(l_target) - g_lo).clamp_min(0.0)

        if torch.any(shift > 0):
            A_norm = A_norm.clone()
            A_norm.diagonal(dim1=-2, dim2=-1).add_(shift.unsqueeze(-1))
            abs_row_sum3 = A_norm.abs().sum(dim=-1)
            u2 = abs_row_sum3.max(dim=-1)[0].clamp_min(eps)
            A_norm = A_norm / u2.unsqueeze(-1).unsqueeze(-1)

    # rho proxy in fp32
    diag_mean = A_norm.float().diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
    max_row = A_norm.float().abs().sum(dim=-1).max(dim=-1)[0].clamp_min(1e-12)
    rho = max_row / diag_mean

    return A_norm, rho


# ---------------------------
# Iteration kernels
# ---------------------------


@torch.no_grad()
def inverse_sqrt_ns(
    A_norm: torch.Tensor,
    iters: int,
    ws: Optional[IsqrtWorkspace] = None,
    symmetrize_Y: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspace]:
    """
    NS: B = 1.5 I - 0.5 Y
        X <- X B
        Y <- Y B^2
    """
    if not _ws_ok(ws, A_norm):
        ws = _alloc_ws(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)

    for _ in range(int(iters)):
        ws.B.copy_(ws.Y).mul_(-0.5)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(1.5)

        ws.Xbuf = ws.X @ ws.B
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        ws.B2 = ws.B @ ws.B
        ws.Ybuf = ws.Y @ ws.B2
        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws


@torch.no_grad()
def inverse_sqrt_pe_affine(
    A_norm: torch.Tensor,
    ab_t: torch.Tensor,  # [T,2], B = a I + b Y
    ws: Optional[IsqrtWorkspace] = None,
    symmetrize_Y: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspace]:
    """
    PE-NS: affine schedule, NS GEMM count:
      B = a I + b Y
      X <- X B
      Y <- Y B^2
    """
    if not _ws_ok(ws, A_norm):
        ws = _alloc_ws(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)

    T = int(ab_t.shape[0])
    for t in range(T):
        a = float(ab_t[t, 0])
        b = float(ab_t[t, 1])

        ws.B.copy_(ws.Y).mul_(b)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(a)

        ws.Xbuf = ws.X @ ws.B
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        ws.B2 = ws.B @ ws.B
        ws.Ybuf = ws.Y @ ws.B2
        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws


@torch.no_grad()
def inverse_sqrt_pe_quadratic(
    A_norm: torch.Tensor,
    abc_t: torch.Tensor,  # [T,3], B = a I + b Y + c Y^2
    ws: Optional[IsqrtWorkspace] = None,
    symmetrize_Y: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspace]:
    """
    PE-quadratic:
      B = a I + b Y + c Y^2
      X <- X B
      Y <- Y B^2
    Extra GEMM vs NS due to Y^2.
    """
    if not _ws_ok(ws, A_norm):
        ws = _alloc_ws(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)

    T = int(abc_t.shape[0])
    for t in range(T):
        a = float(abc_t[t, 0])
        b = float(abc_t[t, 1])
        c = float(abc_t[t, 2])

        ws.Y2 = ws.Y @ ws.Y
        ws.B.copy_(ws.Y2).mul_(c)
        ws.B.add_(ws.Y, alpha=b)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(a)

        ws.Xbuf = ws.X @ ws.B
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        ws.B2 = ws.B @ ws.B
        ws.Ybuf = ws.Y @ ws.B2
        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws


# ---------------------------
# Metrics
# ---------------------------


@torch.no_grad()
def isqrt_residual(X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    n = A.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A).contiguous()
    R = eye - (X @ A @ X)
    return torch.linalg.matrix_norm(R, ord="fro") / math.sqrt(n)


@torch.no_grad()
def exact_inverse_sqrt(A: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    eigvals, V = torch.linalg.eigh(A.double())
    eigvals = eigvals.clamp_min(eps)
    D = torch.diag_embed(torch.rsqrt(eigvals))
    X = V @ D @ V.mT
    return X.to(dtype=A.dtype)


@torch.no_grad()
def isqrt_relative_error(Xhat: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    Xref = exact_inverse_sqrt(A)
    denom = torch.linalg.matrix_norm(Xref, ord="fro").clamp_min(1e-12)
    num = torch.linalg.matrix_norm(Xhat - Xref, ord="fro")
    return num / denom


# ---------------------------
# SPD cases
# ---------------------------


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


# ---------------------------
# Benchmarking
# ---------------------------


@dataclass
class BenchResult:
    ms: float
    residual: float
    relerr: float
    bad: int


@torch.no_grad()
def eval_method(
    mats: List[torch.Tensor],
    device: torch.device,
    precond: str,
    ridge_rel: float,
    l_target: float,
    method: str,
    pe_affine: torch.Tensor,
    pe_quad: torch.Tensor,
    n_switch: int,
    rho_switch: float,
) -> BenchResult:
    ms_list: List[float] = []
    res_list: List[float] = []
    err_list: List[float] = []
    bad = 0
    ws: Optional[IsqrtWorkspace] = None

    for A in mats:
        A_norm, rho = precond_spd(
            A, mode=precond, ridge_rel=ridge_rel, l_target=l_target
        )
        n = A_norm.shape[-1]
        rho_val = float(rho.item())

        def run():
            nonlocal ws
            if method == "NS3":
                Xn, ws2 = inverse_sqrt_ns(A_norm, iters=3, ws=ws)
            elif method == "NS4":
                Xn, ws2 = inverse_sqrt_ns(A_norm, iters=4, ws=ws)
            elif method == "PE-NS3":
                Xn, ws2 = inverse_sqrt_pe_affine(A_norm, ab_t=pe_affine, ws=ws)
            elif method == "PE2":
                Xn, ws2 = inverse_sqrt_pe_quadratic(A_norm, abc_t=pe_quad, ws=ws)
            elif method == "AUTO":
                # Simple, stable policy based on your observed scaling:
                # - large n: PE2 wins strongly in bf16
                # - small/med: prefer NS3 unless spiky proxy suggests PE helps
                if (n >= int(n_switch)) or (rho_val >= float(rho_switch)):
                    Xn, ws2 = inverse_sqrt_pe_quadratic(A_norm, abc_t=pe_quad, ws=ws)
                else:
                    # PE-NS3 is the "NS-cost" improvement; if it ever regresses, swap to NS3 here.
                    Xn, ws2 = inverse_sqrt_pe_affine(A_norm, ab_t=pe_affine, ws=ws)
            else:
                raise ValueError(f"unknown method: {method}")
            ws = ws2
            return Xn

        ms, Xn = time_ms(run, device)
        ms_list.append(ms)

        if not torch.isfinite(Xn).all():
            bad += 1
            res_list.append(float("inf"))
            err_list.append(float("inf"))
            continue

        res_list.append(float(isqrt_residual(Xn.float(), A_norm.float()).mean().item()))
        err_list.append(
            float(isqrt_relative_error(Xn.float(), A_norm.float()).mean().item())
        )

    return BenchResult(
        ms=median(ms_list), residual=median(res_list), relerr=median(err_list), bad=bad
    )


def parse_shapes(spec: str) -> List[int]:
    vals = []
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
    p.add_argument("--n-switch", type=int, default=768)
    p.add_argument("--rho-switch", type=float, default=4.0)
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

    # PE-quadratic schedule for [0.05, 1.0] (B = a I + b Y + c Y^2)
    pe4_005 = torch.tensor(
        [
            [3.9167303160, -7.5013552885, 4.6874100508],
            [1.9487493185, -1.3745051244, 0.4234345641],
            [1.8750975412, -1.2501586662, 0.3750611211],
            [1.8749537616, -1.2499075251, 0.3749537634],
        ],
        device=device,
        dtype=torch.float32,
    )
    pe2_005 = pe4_005[:2].contiguous()

    # PE-affine (PE-NS) schedule for [0.05, 1.0] (B = a I + b Y), 3 steps.
    # These were computed with a minimax-ish smooth-max objective + positivity constraint + interval propagation.
    pe_ns3_005 = torch.tensor(
        [
            [2.8783235621, -2.2575982465],
            [1.6910184521, -0.6213401081],
            [1.5241523252, -0.5209329213],
        ],
        device=device,
        dtype=torch.float32,
    )

    # Optional compile
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
                f"== SPD size {n}x{n} | dtype={dtype_compute} | compile={args.compile} | precond={args.precond} | l_target={args.l_target} =="
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
                _, ws = inverse_sqrt_ns(A_norm, iters=2, ws=ws)

            for case in cases:
                mats = make_spd_cases(case, n, args.trials, device, torch.float32, g)
                mats = [m.to(dtype_compute) for m in mats]

                rows = []
                for name in ["NS3", "NS4", "PE-NS3", "PE2", "AUTO"]:
                    rr = eval_method(
                        mats=mats,
                        device=device,
                        precond=args.precond,
                        ridge_rel=args.ridge_rel,
                        l_target=args.l_target,
                        method=name,
                        pe_affine=pe_ns3_005,
                        pe_quad=pe2_005,
                        n_switch=args.n_switch,
                        rho_switch=args.rho_switch,
                    )
                    rows.append((name, rr))

                print(f"-- case {case} --")
                for name, rr in rows:
                    print(
                        f"{name:<10s} {rr.ms:8.3f} ms | resid {rr.residual:.3e} | relerr {rr.relerr:.3e} | bad {rr.bad}"
                    )

                feasible = [
                    (nm, rr)
                    for nm, rr in rows
                    if rr.bad == 0 and rr.residual <= args.target_resid
                ]
                if feasible:
                    best_name, best_rr = min(feasible, key=lambda t: t[1].ms)
                    print(
                        f"BEST<=target({args.target_resid:.3g}): {best_name} @ {best_rr.ms:.3f} ms, resid={best_rr.residual:.3e}"
                    )
                else:
                    best_name, best_rr = min(
                        [(nm, rr) for nm, rr in rows if rr.bad == 0],
                        key=lambda t: t[1].ms,
                    )
                    print(
                        f"BEST overall: {best_name} @ {best_rr.ms:.3f} ms, resid={best_rr.residual:.3e}"
                    )
                print()


if __name__ == "__main__":
    main()
