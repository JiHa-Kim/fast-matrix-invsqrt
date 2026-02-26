from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

import torch


def median(xs: Sequence[float]) -> float:
    ys = sorted(float(x) for x in xs)
    if not ys:
        return float("nan")
    mid = len(ys) // 2
    if len(ys) % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


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


def parse_shapes(spec: str) -> List[int]:
    vals: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    if not vals:
        raise ValueError("empty shape list")
    return vals


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


def make_nonspd_cases(
    case: str,
    n: int,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
    g: torch.Generator,
) -> List[torch.Tensor]:
    mats: List[torch.Tensor] = []
    eye = torch.eye(n, device=device, dtype=dtype)
    inv_sqrt_n = 1.0 / float(n) ** 0.5

    for _ in range(int(count)):
        if case == "gaussian_shifted":
            # Well-conditioned dense non-symmetric matrix near identity.
            R = torch.randn(n, n, device=device, dtype=dtype, generator=g)
            A = eye + 0.35 * inv_sqrt_n * R
        elif case == "nonnormal_upper":
            # Strongly non-normal (strictly upper-triangular perturbation).
            U = torch.randn(n, n, device=device, dtype=dtype, generator=g).triu(
                diagonal=1
            )
            A = eye + 0.45 * inv_sqrt_n * U
        elif case == "similarity_posspec":
            # Positive real spectrum and moderate non-normality via a near-identity
            # non-orthogonal similarity transform.
            eigs = torch.linspace(0.2, 1.0, steps=n, device=device, dtype=dtype)
            Q, _ = torch.linalg.qr(
                torch.randn(n, n, device=device, dtype=dtype, generator=g),
                mode="reduced",
            )
            S = torch.randn(n, n, device=device, dtype=dtype, generator=g).triu(
                diagonal=1
            )
            P = Q @ (eye + 0.15 * inv_sqrt_n * S)
            P_inv = torch.linalg.inv(P)
            A = (P * eigs.unsqueeze(0)) @ P_inv
        elif case == "similarity_posspec_hard":
            # Harder non-normal variant with denser non-orthogonal similarity.
            eigs = torch.linspace(0.05, 1.0, steps=n, device=device, dtype=dtype)
            P = torch.randn(n, n, device=device, dtype=dtype, generator=g)
            P = P + 0.1 * eye
            try:
                P_inv = torch.linalg.inv(P)
            except RuntimeError:
                P = P + 0.2 * eye
                P_inv = torch.linalg.inv(P)
            A = (P * eigs.unsqueeze(0)) @ P_inv
        else:
            raise ValueError(f"unknown non-SPD case: {case}")

        mats.append(A)

    return mats


def maybe_compile(fn, enabled: bool):
    if not enabled:
        return fn
    try:
        return torch.compile(
            fn,
            mode="max-autotune",
            fullgraph=False,
            options={"triton.cudagraphs": False},
        )
    except Exception:
        return fn
