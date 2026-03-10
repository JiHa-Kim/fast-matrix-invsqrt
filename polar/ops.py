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
def gram_xtx_chunked(X: Tensor, chunk_rows: int, accum_dtype: torch.dtype) -> Tensor:
    m, n = X.shape
    S = torch.zeros((n, n), device=X.device, dtype=accum_dtype)
    for i in range(0, m, chunk_rows):
        Xi = X[i : i + chunk_rows].to(dtype=accum_dtype)
        S.addmm_(Xi.T, Xi)
    return symmetrize(S)


@torch.no_grad()
def gram_xtx_chunked_fp64(X: Tensor, chunk_rows: int) -> Tensor:
    m, n = X.shape
    S = torch.zeros((n, n), device=X.device, dtype=torch.float64)
    for i in range(0, m, chunk_rows):
        # The user wants to keep original precision logic.
        # This code explicitly converts to float64 for the MMM.
        Xi = X[i : i + chunk_rows].float().to(torch.float64)
        S.addmm_(Xi.T, Xi)
    return symmetrize(S)


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
def apply_right_small_chunked(
    X: Tensor, U: Tensor, rhs_chunk_rows: int, out_dtype: torch.dtype
) -> Tensor:
    m, n = X.shape
    X_next = torch.empty((m, n), device=X.device, dtype=out_dtype)
    # Always use float64 for the multiplication if U is float64 to maintain stability
    # especially when U is ill-conditioned (which happens in DWH/Zolo steps).
    U_work = U.to(torch.float64)

    for i in range(0, m, rhs_chunk_rows):
        Xi = X[i : i + rhs_chunk_rows].to(dtype=torch.float64)
        Zi = Xi @ U_work
        X_next[i : i + rhs_chunk_rows] = Zi.to(dtype=out_dtype)

    return X_next
