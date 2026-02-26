from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence, Tuple

import torch


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
