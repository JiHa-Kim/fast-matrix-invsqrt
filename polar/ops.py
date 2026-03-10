from __future__ import annotations

import math
import time
from typing import Tuple

import torch

Tensor = torch.Tensor


def bf16_target(mode: str) -> float:
    u = 2.0**-8
    if mode == "aggressive":
        return float(1.0 + u)
    if mode == "robust":
        return float((1.0 + u) / (1.0 - u))
    raise ValueError(mode)


def symmetrize(A: Tensor) -> Tensor:
    return 0.5 * (A + A.mT)


def cuda_time_ms(fn):
    if not torch.cuda.is_available():
        t0 = time.time()
        out = fn()
        return 1000.0 * (time.time() - t0), out
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), out


def safe_exp(x: float) -> float:
    if x >= 709.0:
        return float("inf")
    return float(math.exp(x))


def acosh_exp(logu: float) -> float:
    if logu <= 0.0:
        return 0.0
    if logu < 20.0:
        u = math.exp(logu)
        return float(math.acosh(max(u, 1.0)))
    return float(logu + math.log(2.0))


@torch.no_grad()
def gram_xtx(X: Tensor, accum_dtype: torch.dtype) -> Tensor:
    return symmetrize((X.to(dtype=accum_dtype)).mT @ X.to(dtype=accum_dtype))


@torch.no_grad()
def gram_xtx_fp64(X: Tensor) -> Tensor:
    X64 = X.float().to(torch.float64)
    return symmetrize(X64.mT @ X64)


@torch.no_grad()
def chol_with_jitter_fp64(
    A: Tensor, jitter_rel: float, max_tries: int = 8
) -> Tuple[Tensor, float]:
    # Ensure input is float64 and symmetric as per original
    A = symmetrize(A.to(torch.float64))
    if not torch.isfinite(A).all():
        raise RuntimeError("non-finite matrix before Cholesky")

    n = A.shape[0]
    # Keep trace and eye logic as per original to preserve exact numerical behavior
    I = torch.eye(n, device=A.device, dtype=torch.float64)
    scale = float((torch.trace(A).abs() / max(n, 1)).item())
    base = max(float(jitter_rel) * max(scale, 1.0), 1e-30)

    delta = 0.0
    for _ in range(max_tries):
        At = A if delta == 0.0 else (A + delta * I)
        L, info = torch.linalg.cholesky_ex(At)
        if int(info.item()) == 0:
            return L, float(delta)
        delta = base if delta == 0.0 else 2.0 * delta

    raise RuntimeError("Cholesky failed even after jitter escalation")


@torch.no_grad()
def cert_bound_trace_logdet(S: Tensor, jitter_rel: float) -> Tuple[float, float]:
    S = symmetrize(S.to(torch.float64))
    n = S.shape[0]

    L, shift = chol_with_jitter_fp64(S, jitter_rel=jitter_rel)
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum().item()

    a = max(float((torch.trace(S) / n).item()), 1e-300)
    g = safe_exp(logdet / n)
    r = max(a / max(g, 1e-300), 1.0)

    logu = 0.5 * n * math.log(r)
    eta_ub = acosh_exp(logu)
    return float(safe_exp(eta_ub)), float(shift)


@torch.no_grad()
def exact_eigvalsh(S: Tensor, eig_device: str = "auto") -> Tensor:
    S = symmetrize(S.to(torch.float64))
    n = S.shape[0]

    if eig_device == "cpu":
        use_cpu = True
    elif eig_device == "cuda":
        use_cpu = False
    else:
        use_cpu = (S.device.type != "cuda") or (n >= 4096)

    if use_cpu:
        evals = torch.linalg.eigvalsh(S.cpu())
        return evals.to(device=S.device)
    return torch.linalg.eigvalsh(S)


@torch.no_grad()
def apply_right(X: Tensor, U: Tensor, out_dtype: torch.dtype) -> Tensor:
    return (X.to(torch.float64) @ U.to(torch.float64)).to(dtype=out_dtype)


@torch.no_grad()
def apply_right_typed(
    X: Tensor,
    U: Tensor,
    matmul_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> Tensor:
    orig_precision = torch.get_float32_matmul_precision()
    if matmul_dtype == torch.float32:
        torch.set_float32_matmul_precision("high")

    try:
        return (X.to(dtype=matmul_dtype) @ U.to(dtype=matmul_dtype)).to(dtype=out_dtype)
    finally:
        torch.set_float32_matmul_precision(orig_precision)
