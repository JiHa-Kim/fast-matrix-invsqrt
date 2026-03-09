#!/usr/bin/env python3
import math
import random
import time
from typing import List, Tuple

import torch

Tensor = torch.Tensor


def symmetrize(A: Tensor) -> Tensor:
    # A.add_(A.mT) fails because A.mT is a view of A.
    # We use the standard form which is safe and reasonably fast.
    return 0.5 * (A + A.mT)


def pct(xs: List[float], p: float) -> float:
    ys = [float(x) for x in xs if math.isfinite(float(x))]
    if not ys:
        return float("nan")
    ys.sort()
    i = int(round(p * (len(ys) - 1)))
    i = max(0, min(len(ys) - 1, i))
    return float(ys[i])


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


def seed_all(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rel_fro(A: Tensor, B: Tensor) -> float:
    num = float(torch.linalg.matrix_norm(A - B, ord="fro").item())
    den = max(float(torch.linalg.matrix_norm(B, ord="fro").item()), 1e-300)
    return float(num / den)


def rel_spec(A: Tensor, B: Tensor) -> float:
    num = float(torch.linalg.matrix_norm(A - B, ord=2).item())
    den = max(float(torch.linalg.matrix_norm(B, ord=2).item()), 1e-300)
    return float(num / den)


@torch.no_grad()
def chol_with_jitter_fp64(
    A: Tensor,
    jitter_rel: float,
    max_tries: int = 8,
) -> Tuple[Tensor, float]:
    A = symmetrize(A.to(torch.float64))
    if not torch.isfinite(A).all():
        raise RuntimeError("non-finite matrix before Cholesky")

    n = A.shape[0]
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
def make_spd_honest_fp64(P: Tensor, jitter_rel: float) -> Tuple[Tensor, float]:
    P = symmetrize(P.to(torch.float64))
    _, shift = chol_with_jitter_fp64(P, jitter_rel=jitter_rel)
    if shift > 0.0:
        n = P.shape[0]
        I = torch.eye(n, device=P.device, dtype=torch.float64)
        P = symmetrize(P + shift * I)
    return P, float(shift)


@torch.no_grad()
def init_spectrum_exact_fp64(P: Tensor) -> Tuple[float, float]:
    evals = torch.linalg.eigvalsh(symmetrize(P.to(torch.float64)))
    lam_min = max(float(evals[0].item()), 1e-300)
    lam_max = max(float(evals[-1].item()), lam_min)
    return float(lam_min), float(lam_max)


@torch.no_grad()
def apply_right_chunked(
    Y: Tensor, Q: Tensor, chunk_rows: int, out_dtype: torch.dtype
) -> Tensor:
    m, n = Y.shape
    # If out_dtype is float32, we can usually afford to do the GEMM in float32.
    # Gawlik minimax iterations are robust. For p-th roots, the "small" side
    # MUST be float64, but applying it to the "large" side can often be float32.
    comp_dtype = Q.dtype
    if out_dtype == torch.float32 and Q.dtype == torch.float64:
        # On consumer GPUs, float64 is 32x slower than float32.
        # If the user chose float32 for iterations, they likely want speed.
        comp_dtype = torch.float32

    out = torch.empty((m, n), device=Y.device, dtype=out_dtype)
    Q_comp = Q.to(comp_dtype)
    
    # If m is small enough, avoid chunking overhead.
    # 32768 * 8192 * 4 bytes is 1GB, which fits in most GPUs.
    # Let's use a heuristic for chunking.
    if m <= chunk_rows:
        return (Y.to(comp_dtype) @ Q_comp).to(out_dtype)

    for i in range(0, m, chunk_rows):
        end = min(i + chunk_rows, m)
        Yi = Y[i:end].to(comp_dtype)
        Zi = Yi @ Q_comp
        out[i:end] = Zi.to(out_dtype)
    return out
