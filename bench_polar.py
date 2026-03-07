#!/usr/bin/env python3
"""
bench_polar_final4.py

Sensible, optimized benchmark suite for bf16/fp16 approximate polar/orthogonalization.

PE5 (Polar Express degree-5 odd polynomial) preconditioning options kept:
  - aol: Almost-Orthogonal Layer scaling (no extra scaling; already enforces ||X||2 <= 1)
         Implemented in a fused way when left-small (m<=n): we reuse A0 = X X^T and
         compute A1 = D A0 D so the first iteration doesn't recompute A.
  - minbound: global scaling by 1 / min(||X||F, sqrt(||X||1 * ||X||inf))
              (basin guarantee via provable upper bound on ||X||2).

Gram-first options:
  - gram_halley: Halley inverse sqrt using fp32 SPD Cholesky_ex + cholesky_solve
  - gram_ns: Newton-Schulz inverse sqrt on Gram (fp32 small-side) with basin-guaranteeing scaling

Average-case and worst-case spectra families included.

Outputs JSONL.

Example:
  python bench_polar_final4.py --device cuda --dtype bf16 --tf32 \
    --shapesets kimi_k2 glm5 sweep \
    --methods pe5 gram_halley gram_ns \
    --steps_list 1 2 3 4 5 6 \
    --pe_preconds aol minbound \
    --avg_profiles loguniform_kappa powerlaw twocluster one_tiny_outlier rankdef_plus_ridge \
    --worst_profiles geo_1e8 geo_1e12 half_big_half_tiny single_spike_single_tiny \
    --n_avg_seeds 20 \
    --target_delta_f 0.2 \
    --out results.jsonl
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple, Optional

import torch


_PE5 = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]


# ----------------------------
# Timing utilities
# ----------------------------
class CUDATimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def time_ms(
        self, fn: Callable[[], torch.Tensor], repeats: int, warmup: int
    ) -> Tuple[float, torch.Tensor]:
        last = None
        for _ in range(warmup):
            last = fn()
        torch.cuda.synchronize(self.device)
        times: List[float] = []
        for _ in range(repeats):
            self.start.record()
            last = fn()
            self.end.record()
            torch.cuda.synchronize(self.device)
            times.append(self.start.elapsed_time(self.end))
        times.sort()
        return float(times[len(times) // 2]), last


def _time_ms_cpu(
    fn: Callable[[], torch.Tensor], repeats: int, warmup: int
) -> Tuple[float, torch.Tensor]:
    last = None
    for _ in range(warmup):
        last = fn()
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.time()
        last = fn()
        t1 = time.time()
        times.append((t1 - t0) * 1e3)
    times.sort()
    return float(times[len(times) // 2]), last


# ----------------------------
# Helpers
# ----------------------------
def _sym(S: torch.Tensor) -> torch.Tensor:
    return 0.5 * (S + S.transpose(-2, -1))


def _eye(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.eye(n, device=device, dtype=dtype)


def _maybe_transpose_to_left_small(
    G: torch.Tensor, auto_transpose: bool
) -> Tuple[torch.Tensor, bool]:
    if (not auto_transpose) or (G.shape[-2] <= G.shape[-1]):
        return G, False
    return G.transpose(-2, -1).contiguous(), True


def _power_norm_sym(A: torch.Tensor, iters: int = 20) -> float:
    v = torch.ones((A.shape[-1],), device=A.device, dtype=A.dtype)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        w = A @ v
        v = w / (w.norm() + 1e-12)
    w = A @ v
    return float((v * w).sum().abs().item())


@torch.inference_mode()
def orth_metrics(Q: torch.Tensor, spectral_iters: int = 20) -> Dict[str, float]:
    m, n = Q.shape[-2], Q.shape[-1]
    Qt = Q.transpose(-2, -1)
    if m <= n:
        S = Q.float() @ Qt.float()
    else:
        S = Qt.float() @ Q.float()
    S = _sym(S)
    E = S - _eye(S.shape[-1], S.device, S.dtype)
    dF = float(torch.linalg.norm(E, ord="fro").item())
    r2 = _power_norm_sym(E, iters=spectral_iters)
    return {"delta_F": dF, "rho_2": r2, "small_dim": int(S.shape[-1])}


# ----------------------------
# MinBound scaling (provable upper bound for ||X||2)
# ----------------------------
@torch.inference_mode()
def spectral_ub_min_fro_1inf(X: torch.Tensor, eps: float = 1e-12) -> float:
    A = X.float()
    fro = float((A * A).sum().sqrt().clamp_min(eps).item())
    n1 = float(A.abs().sum(dim=0).max().clamp_min(eps).item())
    ninf = float(A.abs().sum(dim=1).max().clamp_min(eps).item())
    ub_1inf = math.sqrt(n1 * ninf)
    return float(min(fro, ub_1inf))


@torch.inference_mode()
def precond_minbound(X: torch.Tensor, eps: float, safety: float) -> torch.Tensor:
    ub = spectral_ub_min_fro_1inf(X, eps=eps)
    target = 1.0 / max(1.0, safety)
    if ub <= target:
        return X
    return X * (target / ub)


# ----------------------------
# AOL preconditioning (fused for left-small)
# ----------------------------
@torch.inference_mode()
def aol_fused_left_small(
    X: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Left-small X is (m,n) with m<=n. We want a cheap AOL-style spectral bound using the small Gram.
    Apply AOL to X^T (column scaling of X^T), which corresponds to row scaling of X:

      A0 = X X^T  (m,m)
      d_i = 1 / sqrt( sum_j |A0_{ij}| )      (row sums of abs Gram)
      X1 = D X0
      A1 = D A0 D  == X1 X1^T

    Returns:
      X1, A1 to reuse in first PE iteration (saves recomputing A).
    This enforces ||X1||2 <= 1 in exact arithmetic by Theorem 1 applied to P = X^T. :contentReference[oaicite:5]{index=5}
    """
    A0 = X @ X.transpose(-2, -1)  # (m,m)
    rs = A0.abs().sum(dim=-1).clamp_min(eps)  # (m,)
    d = torch.rsqrt(rs).to(dtype=X.dtype)
    X1 = d.unsqueeze(-1) * X
    # A1 = D A0 D
    A1 = (d.unsqueeze(-1) * A0) * d.unsqueeze(-2)
    return X1, A1


@torch.inference_mode()
def aol_generic(X: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Generic AOL as in Prach-Lampert / Turbo-Muon: scale columns using row sums of abs(X^T X).
    More expensive when X is left-small (since X^T X is big), so for benchmarking we use the fused
    left-small variant above when auto_transpose is enabled.
    """
    G = X.transpose(-2, -1) @ X  # (n,n)
    rs = G.abs().sum(dim=-1).clamp_min(eps)  # (n,)
    s = torch.rsqrt(rs).to(dtype=X.dtype)
    return X * s.unsqueeze(-2)


# ----------------------------
# PE5 method
# ----------------------------
@torch.inference_mode()
def pe5(
    G: torch.Tensor,
    steps: int,
    precond: str,  # 'aol' or 'minbound'
    eps: float,
    safety: float,
    auto_transpose: bool,
    aol_guard: bool,
    aol_guard_tol: float,
) -> torch.Tensor:
    """
    PE5 odd polynomial iteration on left-small orientation (m<=n):
      A = X X^T
      A2 = A A
      X <- a X + (b A + c A2) X

    Preconditioners:
      - aol: AOL scaling (no extra scaling). Fused to reuse A for first iter.
      - minbound: global scaling by min(Fro, sqrt(1*inf)) upper bound.

    Optional AOL guard:
      After AOL, compute a cheap ub and downscale only if ub > 1 + tol.
      This should almost never trigger; it is only to avoid rare bf16 edge cases.
    """
    X, tflag = _maybe_transpose_to_left_small(G, auto_transpose)
    X = X.contiguous()

    A = None
    if precond == "aol":
        # Use fused small-Gram AOL when left-small
        X, A = aol_fused_left_small(X, eps=eps)

        if aol_guard:
            ub = spectral_ub_min_fro_1inf(X, eps=eps)
            if ub > (1.0 + aol_guard_tol):
                X = X * (1.0 / ub)
                A = (1.0 / (ub * ub)) * A  # keep A consistent: A = X X^T

    elif precond == "minbound":
        X = precond_minbound(X, eps=eps, safety=safety)
    else:
        raise ValueError("pe_precond must be 'aol' or 'minbound'")

    coeffs = _PE5[:steps] + [_PE5[-1]] * max(0, steps - len(_PE5))
    for it, (a, b, c) in enumerate(coeffs):
        if it == 0 and A is not None:
            # Reuse A from AOL-fused preconditioner
            A0 = A
        else:
            A0 = X @ X.transpose(-2, -1)

        A2 = A0 @ A0
        B = b * A0 + c * A2
        X = a * X + (B @ X)

    if tflag:
        X = X.transpose(-2, -1).contiguous()
    return X


# ----------------------------
# Gram-first methods
# ----------------------------
@torch.inference_mode()
def gram_from_left_small(H: torch.Tensor) -> torch.Tensor:
    return _sym(H.float() @ H.transpose(-2, -1).float())


@torch.inference_mode()
def gram_scale_for_ns(A: torch.Tensor, eps: float) -> float:
    tr_over_m = float(A.diagonal(dim1=-2, dim2=-1).mean().clamp_min(eps).item())
    fro = float((A * A).sum().sqrt().clamp_min(eps).item())
    n1 = float(A.abs().sum(dim=0).max().clamp_min(eps).item())
    ninf = float(A.abs().sum(dim=1).max().clamp_min(eps).item())
    ub = min(fro, math.sqrt(n1 * ninf))
    # ensure ||A/alpha||2 <= 3
    alpha = max(tr_over_m, ub / 3.0, eps)
    return float(alpha)


@torch.inference_mode()
def gram_halley_cholesky(
    G: torch.Tensor,
    steps: int,
    lam: float,
    eps: float,
    auto_transpose: bool,
) -> torch.Tensor:
    """
    SPD-safe Halley inverse sqrt on Gram:
      B = H H^T + (lam+eps) I (fp32 SPD)
      Z_{k+1} = W_k Z_k,  W_k = (3I + S_k)(I + 3S_k)^{-1}, S_k = Z_k B Z_k^T
      Q = Z H

    Cholesky_ex ensures we avoid CPU sync; jitter retry handles rare fp issues.
    """
    H, tflag = _maybe_transpose_to_left_small(G, auto_transpose)
    H = H.contiguous()
    m = H.shape[-2]

    I_mat = _eye(m, H.device, torch.float32)
    B = gram_from_left_small(H) + (lam + eps) * I_mat

    # mild scaling (not required for SPD-safety, helps numerics)
    alpha = float(B.diagonal(dim1=-2, dim2=-1).mean().clamp_min(eps).item())
    Bn = B / alpha

    Z = I_mat
    for _ in range(steps):
        S = _sym(Z @ Bn @ Z.transpose(-2, -1))
        M = _sym(I_mat + 3.0 * S) + eps * I_mat
        L, info = torch.linalg.cholesky_ex(M, check_errors=False)
        if int(info.item()) != 0:
            L, info = torch.linalg.cholesky_ex(
                M + (10.0 * eps) * I_mat, check_errors=False
            )
            if int(info.item()) != 0:
                break
        RHS = 3.0 * I_mat + S
        W = torch.cholesky_solve(RHS.transpose(-2, -1), L).transpose(-2, -1)
        Z = _sym(W) @ Z

    Z = Z * (1.0 / math.sqrt(alpha))
    Q = Z.to(dtype=H.dtype) @ H
    if tflag:
        Q = Q.transpose(-2, -1).contiguous()
    return Q


@torch.inference_mode()
def gram_newton_schulz(
    G: torch.Tensor,
    steps: int,
    lam: float,
    eps: float,
    auto_transpose: bool,
) -> torch.Tensor:
    """
    GEMM-only Newton-Schulz inverse sqrt on Gram with basin-guaranteeing scaling.
    """
    H, tflag = _maybe_transpose_to_left_small(G, auto_transpose)
    H = H.contiguous()
    m = H.shape[-2]

    I_mat = _eye(m, H.device, torch.float32)
    A = gram_from_left_small(H) + (lam + eps) * I_mat
    alpha = gram_scale_for_ns(A, eps=eps)

    Y = A / alpha
    Z = I_mat
    for _ in range(steps):
        T = 0.5 * (3.0 * I_mat - (Z @ Y))
        Y = Y @ T
        Z = T @ Z

    Z = Z * (1.0 / math.sqrt(alpha))
    Q = Z.to(dtype=H.dtype) @ H
    if tflag:
        Q = Q.transpose(-2, -1).contiguous()
    return Q


# ----------------------------
# Synthetic spectra + matrix generator
# ----------------------------
@torch.inference_mode()
def rand_orth_cols(m: int, n: int, device: torch.device) -> torch.Tensor:
    A = torch.randn((m, n), device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q


@torch.inference_mode()
def make_matrix_from_singulars(
    m: int, n: int, s: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    r = min(m, n)
    assert s.numel() == r
    if m >= n:
        U = rand_orth_cols(m, n, device)
        V = rand_orth_cols(n, n, device)
        G = (U @ torch.diag(s.float()) @ V.transpose(-2, -1)).to(dtype=dtype)
    else:
        U = rand_orth_cols(m, m, device)
        V = rand_orth_cols(n, m, device)
        G = (U @ torch.diag(s.float()) @ V.transpose(-2, -1)).to(dtype=dtype)
    return G


def singulars_avg(r: int, kind: str, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    if kind == "loguniform_kappa":
        log10_kappa = float(
            torch.empty((), device=device).uniform_(0.0, 10.0, generator=g).item()
        )
        kappa = 10.0**log10_kappa
        t = torch.linspace(0, 1, r, device=device)
        s = kappa ** (-t)
        s = s / s.median()

    elif kind == "powerlaw":
        alpha = float(
            torch.empty((), device=device).uniform_(0.25, 3.0, generator=g).item()
        )
        i = torch.arange(1, r + 1, device=device, dtype=torch.float32)
        s = i.pow(-alpha)
        s = s / s.median()

    elif kind == "twocluster":
        hi = 10.0 ** float(
            torch.empty((), device=device).uniform_(0.0, 2.0, generator=g).item()
        )
        lo = 10.0 ** float(
            torch.empty((), device=device).uniform_(-10.0, -2.0, generator=g).item()
        )
        frac = float(
            torch.empty((), device=device).uniform_(0.02, 0.5, generator=g).item()
        )
        k = max(1, int(frac * r))
        s = torch.full((r,), lo, device=device)
        s[:k] = hi
        perm = torch.randperm(r, generator=g, device=device)
        s = s[perm]
        s = s / s[s > 0].median()

    elif kind == "one_tiny_outlier":
        kappa = 10.0 ** float(
            torch.empty((), device=device).uniform_(6.0, 12.0, generator=g).item()
        )
        s = torch.ones(r, device=device)
        s[-1] = 1.0 / kappa

    elif kind == "rankdef_plus_ridge":
        ridge = 10.0 ** float(
            torch.empty((), device=device).uniform_(-12.0, -6.0, generator=g).item()
        )
        frac = float(
            torch.empty((), device=device).uniform_(0.05, 0.3, generator=g).item()
        )
        z = max(1, int(frac * r))
        base = torch.logspace(0.0, -4.0, r, device=device)
        base[-z:] = ridge
        s = base / base[base > 0].median()

    else:
        raise ValueError(kind)

    return s


def singulars_worst(r: int, kind: str, device: torch.device) -> torch.Tensor:
    if kind == "geo_1e8":
        s = torch.logspace(0.0, -8.0, r, device=device)
        s = s / s.median()
    elif kind == "geo_1e12":
        s = torch.logspace(0.0, -12.0, r, device=device)
        s = s / s[s > 0].median()
    elif kind == "half_big_half_tiny":
        s = torch.ones(r, device=device)
        s[r // 2 :] = 1e-8
        s = s / s.median()
    elif kind == "single_spike_single_tiny":
        s = torch.ones(r, device=device)
        s[0] = 1e4
        s[-1] = 1e-12
        s = s / s[s > 0].median()
    else:
        raise ValueError(kind)
    return s


# ----------------------------
# Shapesets
# ----------------------------
def shapes_kimi_k2() -> List[Tuple[str, int, int]]:
    d, e = 7168, 2048
    return [
        ("k2_attn_qkv_fused", d, 3 * d),
        ("k2_attn_o", d, d),
        ("k2_moe_up", d, e),
        ("k2_moe_down", e, d),
    ]


def shapes_glm5() -> List[Tuple[str, int, int]]:
    d, ff, moe, ql, kv = 6144, 12288, 2048, 2048, 512
    return [
        ("glm5_attn_qkv_fused", d, 3 * d),
        ("glm5_attn_o", d, d),
        ("glm5_dense_up", d, ff),
        ("glm5_dense_down", ff, d),
        ("glm5_moe_up", d, moe),
        ("glm5_moe_down", moe, d),
        ("glm5_q_lora", d, ql),
        ("glm5_kv_lora", d, kv),
    ]


def shapes_sweep() -> List[Tuple[str, int, int]]:
    base = [1024, 2048, 4096, 6144, 7168, 8192]
    ratios = [1, 2, 4, 8, 16, 32]
    out: List[Tuple[str, int, int]] = []
    for d in base:
        for r in ratios:
            out.append((f"sweep_{d}x{r * d}", d, r * d))
            out.append((f"sweep_{r * d}x{d}", r * d, d))
    return out


_SHAPESETS = {"kimi_k2": shapes_kimi_k2, "glm5": shapes_glm5, "sweep": shapes_sweep}


# ----------------------------
# Specs + runner
# ----------------------------
@dataclass
class Spec:
    name: str
    method: str
    steps: int
    pe_precond: str = "aol"  # aol | minbound
    pe_safety: float = 1.0
    aol_guard: bool = False
    aol_guard_tol: float = 5e-3
    lam: float = 0.0


def build_specs(args) -> List[Spec]:
    specs: List[Spec] = []
    if "pe5" in args.methods:
        for st in args.steps_list:
            for p in args.pe_preconds:
                specs.append(
                    Spec(
                        name=f"pe5_{p}_s{st}"
                        + ("_g" if args.aol_guard and p == "aol" else ""),
                        method="pe5",
                        steps=st,
                        pe_precond=p,
                        pe_safety=float(args.pe_safety),
                        aol_guard=bool(args.aol_guard) if p == "aol" else False,
                        aol_guard_tol=float(args.aol_guard_tol),
                    )
                )
    if "gram_halley" in args.methods:
        for st in args.steps_list:
            specs.append(
                Spec(
                    name=f"gram_halley_s{st}_lam{args.lam}",
                    method="gram_halley",
                    steps=st,
                    lam=float(args.lam),
                )
            )
    if "gram_ns" in args.methods:
        for st in args.steps_list:
            specs.append(
                Spec(
                    name=f"gram_ns_s{st}_lam{args.lam}",
                    method="gram_ns",
                    steps=st,
                    lam=float(args.lam),
                )
            )
    return specs


@torch.inference_mode()
def apply_method(G: torch.Tensor, spec: Spec, args) -> torch.Tensor:
    if spec.method == "pe5":
        return pe5(
            G=G,
            steps=spec.steps,
            precond=spec.pe_precond,
            eps=args.eps,
            safety=spec.pe_safety,
            auto_transpose=args.auto_transpose,
            aol_guard=spec.aol_guard,
            aol_guard_tol=spec.aol_guard_tol,
        )
    if spec.method == "gram_halley":
        return gram_halley_cholesky(
            G, spec.steps, spec.lam, args.eps_chol, args.auto_transpose
        )
    if spec.method == "gram_ns":
        return gram_newton_schulz(
            G, spec.steps, spec.lam, args.eps_chol, args.auto_transpose
        )
    raise ValueError(spec.method)


def run_one(G: torch.Tensor, spec: Spec, timer: Optional[CUDATimer], args) -> Dict:
    def fn():
        return apply_method(G, spec, args)

    if G.is_cuda:
        ms, Q = timer.time_ms(fn, repeats=args.repeats, warmup=args.warmup)
    else:
        ms, Q = _time_ms_cpu(fn, repeats=args.repeats, warmup=args.warmup)
    mets = orth_metrics(Q, spectral_iters=args.spectral_iters)
    ok = (mets["delta_F"] <= args.target_delta_f) and (
        not (math.isnan(mets["delta_F"]) or math.isinf(mets["delta_F"]))
    )
    return {"spec": asdict(spec), "time_ms_median": ms, "ok": bool(ok), **mets}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"]
    )
    ap.add_argument("--tf32", action="store_true")

    ap.add_argument("--shapesets", nargs="+", default=["kimi_k2", "glm5", "sweep"])
    ap.add_argument("--auto_transpose", action="store_true")

    ap.add_argument(
        "--avg_profiles",
        nargs="+",
        default=[
            "loguniform_kappa",
            "powerlaw",
            "twocluster",
            "one_tiny_outlier",
            "rankdef_plus_ridge",
        ],
    )
    ap.add_argument(
        "--worst_profiles",
        nargs="+",
        default=[
            "geo_1e8",
            "geo_1e12",
            "half_big_half_tiny",
            "single_spike_single_tiny",
        ],
    )
    ap.add_argument("--n_avg_seeds", type=int, default=20)

    ap.add_argument("--methods", nargs="+", default=["pe5", "gram_halley", "gram_ns"])
    ap.add_argument("--steps_list", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])

    ap.add_argument("--pe_preconds", nargs="+", default=["aol", "minbound"])
    ap.add_argument("--pe_safety", type=float, default=1.0)

    ap.add_argument("--aol_guard", action="store_true")
    ap.add_argument("--aol_guard_tol", type=float, default=5e-3)

    ap.add_argument("--lam", type=float, default=0.0)

    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--target_delta_f", type=float, default=0.2)
    ap.add_argument("--spectral_iters", type=int, default=20)

    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--eps_chol", type=float, default=1e-8)

    ap.add_argument("--out", type=str, default="results.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        device = torch.device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        try:
            torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
        except Exception:
            pass
        timer = CUDATimer(device)
    else:
        device = torch.device("cpu")
        timer = None

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]

    shapes: List[Tuple[str, int, int]] = []
    for ss in args.shapesets:
        shapes.extend(_SHAPESETS[ss]())

    specs = build_specs(args)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for tag, m, n in shapes:
            for prof in args.avg_profiles:
                for k in range(args.n_avg_seeds):
                    seed = (
                        args.seed
                        + 10_000 * (hash(tag) % 997)
                        + 100 * (hash(prof) % 997)
                        + k
                    )
                    s = singulars_avg(min(m, n), prof, device, seed=seed)
                    G = make_matrix_from_singulars(m, n, s, device=device, dtype=dtype)
                    if G.is_cuda:
                        torch.cuda.synchronize(device)

                    for spec in specs:
                        rec = run_one(G, spec, timer, args)
                        rec["case"] = {
                            "bucket": "avg",
                            "tag": tag,
                            "m": m,
                            "n": n,
                            "profile": prof,
                            "seed": int(seed),
                            "dtype": args.dtype,
                            "device": str(device),
                        }
                        f.write(json.dumps(rec) + "\n")
                        f.flush()

            for prof in args.worst_profiles:
                s = singulars_worst(min(m, n), prof, device)
                for k in range(min(5, args.n_avg_seeds)):
                    seed = (
                        args.seed
                        + 50_000 * (hash(tag) % 997)
                        + 500 * (hash(prof) % 997)
                        + k
                    )
                    torch.manual_seed(seed)
                    G = make_matrix_from_singulars(m, n, s, device=device, dtype=dtype)
                    if G.is_cuda:
                        torch.cuda.synchronize(device)

                    for spec in specs:
                        rec = run_one(G, spec, timer, args)
                        rec["case"] = {
                            "bucket": "worst",
                            "tag": tag,
                            "m": m,
                            "n": n,
                            "profile": prof,
                            "seed": int(seed),
                            "dtype": args.dtype,
                            "device": str(device),
                        }
                        f.write(json.dumps(rec) + "\n")
                        f.flush()

    print("Wrote results to", args.out)


if __name__ == "__main__":
    main()
