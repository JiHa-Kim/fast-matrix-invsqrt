#!/usr/bin/env python3
# polar_dwh_bf16_predcap_gersh_power_suite.py
#
# BF16-friendly polar/orthogonalization preconditioner via direct x g(x^2)
# (DWH/QDWH-style), with GPU-fast extremal eigenvalue BOUNDS/ESTIMATES designed
# for bf16 stability.
#
# What this script does (high level):
#   - Stores tall iterand X in bf16.
#   - Forms small Gram S = X^T X in fp32, symmetrizes, trace-centers.
#   - Uses FAST GPU bounds for eigenvalues of S:
#       * Gershgorin bounds (O(n^2), purely elementwise+reductions, GPU-friendly).
#         We apply a conservative "sum fudge" so float32 reduction never underestimates
#         row sums, which could otherwise break the bound.
#       * Optional block power iteration to get a tighter *estimate* of lambda_max
#         using GEMM (S @ Q) with small block size, GPU-efficient.
#   - Chooses coefficients (a,b,c) by searching ell in [lo, 1] and minimizing
#     predicted kappa based on (lam_min_lb, lam_max_design), under caps.
#   - Factors only M = I + c S (never S), with jitter only on M, and optional fp64 fallback.
#
# Certification:
#   - For n <= eig_threshold: eigvalsh(S) for exact cert.
#   - For n > eig_threshold: rigorous-ish Gershgorin kappa upper bound:
#       kappa_ub(S) = lam_max_ub / max(lam_min_lb, floor)
#     and kappa_O_ub = sqrt(kappa_ub).
#
# IMPORTANT:
#   - Gershgorin bounds depend on accurate row sums of abs entries. In float32,
#     reductions can undercount. We compensate with a provable-style worst-case
#     relative fudge ~= O(n * eps32). This makes the bound conservative in practice.
#
# Suite:
#   - Runs "Kimi K2/GLM5-ish" shapes; skips OOM cases and continues.

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import time
from typing import List, Tuple

import numpy as np
import torch

Tensor = torch.Tensor


# ----------------------------- utilities -----------------------------------


def symmetrize(A: Tensor) -> Tensor:
    return 0.5 * (A + A.T)


def pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    i = int(round(p * (len(ys) - 1)))
    i = max(0, min(len(ys) - 1, i))
    return float(ys[i])


def cuda_time_ms(fn):
    if not torch.cuda.is_available():
        return 0.0, fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), out


# ----------------------- fast eigenvalue bounds/estimates -------------------


@torch.no_grad()
def gershgorin_bounds_symmetric(
    S: Tensor,
    sum_fudge: float = 0.0,
) -> Tuple[float, float]:
    """
    Gershgorin bounds for symmetric S:
      lam_min >= min_i (S_ii - sum_{j!=i} |S_ij|)
      lam_max <= max_i (S_ii + sum_{j!=i} |S_ij|)

    We compute row sums in float32 and apply a conservative multiplicative fudge
    to avoid underestimation from float32 reductions:
      offsum <- offsum * (1 + sum_fudge_effective)

    If sum_fudge is 0, we set:
      sum_fudge_effective = max(1e-4, 4 * n * eps32)
    which is intentionally conservative.
    """
    S = symmetrize(S)
    n = S.shape[0]
    diag = torch.diag(S)
    absS = torch.abs(S)
    # row_abs_sum includes diagonal
    row_abs_sum = torch.sum(absS, dim=1)
    offsum = row_abs_sum - torch.abs(diag)

    eps32 = torch.finfo(torch.float32).eps
    auto_fudge = max(1e-4, float(4.0 * n * eps32))
    fudge = float(sum_fudge) if sum_fudge > 0.0 else auto_fudge

    offsum = offsum * (1.0 + fudge)

    lb = torch.min(diag - offsum).item()
    ub = torch.max(diag + offsum).item()
    return float(lb), float(ub)


@torch.no_grad()
def block_power_lambda_max(
    S: Tensor,
    iters: int = 8,
    block: int = 8,
    seed: int = 0,
    reorth: bool = True,
) -> float:
    """
    GPU-efficient lambda_max estimate using block power iteration with GEMM.

    Returns a LOWER bound on lambda_max(S) (Rayleigh in the block subspace),
    which is useful to tighten design endpoints:
      lam_max_design = min(lam_max_ub, lam_max_est*(1+margin))

    This uses (n x n) @ (n x b) GEMM each iteration (fast on GPU).
    """
    S = symmetrize(S)
    n = S.shape[0]
    device = S.device
    dtype = S.dtype

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    Q = torch.randn((n, block), device=device, dtype=dtype, generator=gen)
    # Orthonormalize initial block
    Q, _ = torch.linalg.qr(Q, mode="reduced")

    for _ in range(iters):
        Z = S @ Q  # GEMM (n,n)x(n,b)
        Q, _ = torch.linalg.qr(Z, mode="reduced")
        if reorth:
            # one extra reorth pass helps numerical stability for small block
            Q, _ = torch.linalg.qr(Q, mode="reduced")

    # Rayleigh on subspace
    B = Q.T @ (S @ Q)  # (b,b)
    B = symmetrize(B).double()
    evals = torch.linalg.eigvalsh(B)
    lam_max_est = float(evals[-1].item())
    return lam_max_est


# ----------------------------- DWH coefficients ----------------------------


def dwh_coeffs(ell: float) -> Tuple[float, float, float]:
    ell = float(ell)
    ell = min(max(ell, 1e-12), 1.0)
    ell2 = ell * ell

    d = (4.0 * (1.0 - ell2) / (ell2 * ell2)) ** (1.0 / 3.0)
    h = math.sqrt(1.0 + d) + 0.5 * math.sqrt(
        8.0 - 4.0 * d + 8.0 * (2.0 - ell2) / (ell2 * math.sqrt(1.0 + d))
    )
    a = h
    b = (a - 1.0) * (a - 1.0) / 4.0
    c = a + b - 1.0
    return a, b, c


def phi_map(a: float, b: float, c: float, x: float) -> float:
    r = (a + b * x) / (1.0 + c * x)
    return x * (r * r)


def pick_capped_coeffs_by_pred(
    lam_min: float,
    lam_max: float,
    ell0: float,
    a_cap: float,
    b_cap: float,
    c_cap: float,
    grid: int = 48,
) -> Tuple[float, float, float, float]:
    """
    Search ell in [ell0, 1] (geometric grid) and pick DWH(a,b,c) that:
      - satisfies caps
      - minimizes predicted kappa after one step using endpoints lam_min, lam_max:
          k_pred = phi(lam_max) / phi(lam_min)

    Returns (a,b,c, ell_chosen). If nothing fits, returns mild step (3,1,3).
    """
    lo = min(max(float(ell0), 1e-12), 1.0)
    if lo >= 1.0:
        return 1.0, 0.0, 0.0, 1.0

    lam_min = max(float(lam_min), 0.0)
    lam_max = max(float(lam_max), lam_min)

    ts = np.linspace(0.0, 1.0, int(grid))
    ells = lo * (1.0 / lo) ** ts

    best = None
    best_k = None
    best_ell = None

    for ell in ells:
        a, b, c = dwh_coeffs(float(ell))
        if a > a_cap or b > b_cap or c > c_cap:
            continue
        y_min = phi_map(a, b, c, lam_min)
        y_max = phi_map(a, b, c, lam_max)
        if not (math.isfinite(y_min) and math.isfinite(y_max)):
            continue
        if y_min <= 0.0 or y_max <= 0.0:
            continue
        k = y_max / y_min
        if best_k is None or k < best_k:
            best_k = k
            best = (a, b, c)
            best_ell = float(ell)

    if best is None:
        return 3.0, 1.0, 3.0, lo
    return float(best[0]), float(best[1]), float(best[2]), float(best_ell)


# ----------------------------- Cholesky on M -------------------------------


@torch.no_grad()
def chol_on_M_with_jitter(
    M: Tensor,
    jitter_rel: float,
    max_tries: int = 10,
    allow_fp64_fallback: bool = True,
    fp64_eps_rel: float = 1e-12,
) -> Tuple[Tensor, float]:
    """
    Cholesky for M, with jitter on M only.
    If fp32+jitter fails, optional fp64 minimal shift via eigvalsh(M).
    Returns (L, used_shift_or_jitter). L may be fp32 or fp64.
    """
    M = symmetrize(M)
    if not torch.isfinite(M).all():
        raise RuntimeError("Non-finite entries in M before Cholesky")

    n = M.shape[0]
    I32 = torch.eye(n, device=M.device, dtype=M.dtype)

    tr = torch.trace(M).abs()
    base = float((jitter_rel * (tr / n)).item()) if jitter_rel > 0.0 else 0.0
    delta = base

    for _ in range(max_tries):
        Mt = M if delta == 0.0 else (M + delta * I32)
        L, info = torch.linalg.cholesky_ex(Mt)
        if int(info.item()) == 0:
            return L, float(delta)
        if jitter_rel <= 0.0:
            break
        delta = delta * 2.0 if delta > 0.0 else base

    if not allow_fp64_fallback:
        raise RuntimeError(
            "Cholesky failed on M (fp32+jitter) and fp64 fallback disabled"
        )

    Md = symmetrize(M.double())
    evals = torch.linalg.eigvalsh(Md)
    lam_min = float(evals[0].item())
    trd = float(torch.trace(Md).abs().item())
    eps = fp64_eps_rel * (trd / n if n > 0 else 1.0)
    shift = max(0.0, -lam_min + eps)
    I64 = torch.eye(n, device=M.device, dtype=torch.float64)
    Ld = torch.linalg.cholesky(Md + shift * I64)
    return Ld, float(shift)


# ----------------------------- synthetic matrices --------------------------


def make_matrix_from_singulars(
    m: int,
    singulars: Tensor,
    seed: int,
    device: str,
    storage_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    n = int(singulars.numel())
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    U, _ = torch.linalg.qr(
        torch.randn(m, n, generator=gen, dtype=torch.float32), mode="reduced"
    )
    V, _ = torch.linalg.qr(
        torch.randn(n, n, generator=gen, dtype=torch.float32), mode="reduced"
    )
    G = (U * singulars.float()) @ V.T
    return G.to(device=device, dtype=storage_dtype)


def make_spectrum_bank(
    n: int, kappa_G: float, bank_size: int, seed: int
) -> List[Tensor]:
    sig_max = 1.0
    sig_min = 1.0 / float(kappa_G)
    out: List[Tensor] = []

    out.append(
        torch.logspace(0.0, math.log10(sig_min), n, base=10.0, dtype=torch.float32)
    )

    t = torch.linspace(0.0, 1.0, n)
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        logs = math.log(sig_max) + (math.log(sig_min) - math.log(sig_max)) * (t**p)
        out.append(torch.exp(logs))
        logs = math.log(sig_max) + (math.log(sig_min) - math.log(sig_max)) * (
            1.0 - (1.0 - t) ** p
        )
        out.append(torch.exp(logs))

    for frac in [1 / n, 2 / n, 4 / n, 8 / n, 0.1, 0.25, 0.5, 0.75, 0.9]:
        r = max(1, min(n - 1, int(round(frac * n))))
        s = torch.full((n,), sig_min, dtype=torch.float32)
        s[:r] = sig_max
        out.append(s)

    rng = random.Random(seed)
    while len(out) < bank_size:
        u = sorted([rng.random() for _ in range(n)], reverse=True)
        logs = torch.tensor([math.log(sig_min) * x for x in u], dtype=torch.float32)
        s = torch.exp(logs)
        s[0] = sig_max
        s[-1] = sig_min
        out.append(s)

    return out[:bank_size]


# ----------------------------- suite shapes --------------------------------


def suite_shapes_kimi_glm5() -> List[Tuple[int, int]]:
    return [
        (2048, 256),
        (4096, 256),
        (8192, 256),
        (8192, 1024),
        (16384, 1024),
        (8192, 2048),
        (16384, 2048),
        (28672, 4096),
        (28672, 7168),
        (32768, 8192),
    ]


# ----------------------------- run core ------------------------------------


@dataclasses.dataclass
class RunSummary:
    success: bool
    final_kO_ub: float
    steps: int
    ms_gram: float
    ms_solve: float
    ms_upd: float
    ms_total: float


@torch.no_grad()
def run_one_case(
    G_storage: Tensor,
    target_kappa_O: float,
    max_steps: int,
    ell0: float,
    eps_scale: float,
    safety: float,
    jitter_rel: float,
    tf32: bool,
    psd_repair: bool,
    cert_floor_rel: float,
    sum_fudge: float,
    a_cap: float,
    b_cap: float,
    c_cap: float,
    ell_grid: int,
    use_block_power: bool,
    power_iters: int,
    power_block: int,
    power_margin: float,
    eig_threshold: int,
    allow_fp64_fallback: bool,
) -> RunSummary:
    device = G_storage.device
    m, n = G_storage.shape
    Id_mat = torch.eye(n, device=device, dtype=torch.float32)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high")

    Gf = G_storage.float()

    fro = torch.linalg.norm(Gf, ord="fro")
    denom = float(safety) * fro + float(eps_scale)
    X = (Gf / denom).to(torch.bfloat16)

    tau = 1.0 / float(safety)

    ms_gram_sum = 0.0
    ms_solve_sum = 0.0
    ms_upd_sum = 0.0

    for it in range(1, max_steps + 1):
        # Gram
        ms_gram, S = cuda_time_ms(lambda: symmetrize(X.float().T @ X.float()))
        ms_gram_sum += ms_gram

        # Trace center
        mu = torch.trace(S) / n
        mu_f = float(mu.item())
        if not math.isfinite(mu_f) or mu_f <= 0.0:
            return RunSummary(
                False,
                float("inf"),
                it,
                ms_gram_sum,
                ms_solve_sum,
                ms_upd_sum,
                float("inf"),
            )
        X = (X.float() / math.sqrt(mu_f)).to(torch.bfloat16)
        S = S / mu

        # Fast bounds (Gershgorin)
        lam_lb, lam_ub = gershgorin_bounds_symmetric(S, sum_fudge=sum_fudge)

        # Optional PSD repair of S using Gersh LB (cheapest possible repair)
        # If LB < 0, shift S so LB becomes small positive.
        if psd_repair and lam_lb < 0.0:
            eps = cert_floor_rel * max(lam_ub, 1.0)
            shift = -lam_lb + eps
            S = S + shift * Id_mat
            lam_lb += shift
            lam_ub += shift
            lam_lb = max(lam_lb, 0.0)

        # Certification:
        # - For small n, do exact eigvalsh(S) to accept.
        # - For large n, use Gersh kappa upper bound.
        if n <= eig_threshold:
            Sd = symmetrize(S).double()
            evals = torch.linalg.eigvalsh(Sd)
            lam_min = max(float(evals[0].item()), 0.0)
            lam_max = max(float(evals[-1].item()), 0.0)
            lam_min_safe = max(
                lam_min, cert_floor_rel * lam_max if lam_max > 0.0 else 0.0
            )
            kappa_cert = (
                1.0 if lam_max == 0.0 else (lam_max / max(lam_min_safe, 1e-300))
            )
            kO_ub = math.sqrt(kappa_cert)
            # For design endpoints:
            lam_min_design = lam_min
            lam_max_design = lam_max
        else:
            lam_min_safe = max(max(lam_lb, 0.0), cert_floor_rel * max(lam_ub, 0.0))
            kappa_ub = max(lam_ub, 0.0) / max(lam_min_safe, 1e-300)
            kO_ub = math.sqrt(kappa_ub)
            lam_min_design = max(lam_lb, 0.0)
            lam_max_design = max(lam_ub, lam_min_design)

        if kO_ub <= target_kappa_O:
            ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum
            return RunSummary(
                True, float(kO_ub), it, ms_gram_sum, ms_solve_sum, ms_upd_sum, ms_total
            )

        # Tighten lam_max for DESIGN using block power estimate (lower bound) to counter loose Gersh ub.
        if use_block_power:
            lam_max_est = block_power_lambda_max(
                S.float(),
                iters=power_iters,
                block=power_block,
                seed=1337 + it,
                reorth=True,
            )
            lam_max_design = min(
                lam_max_design, float(lam_max_est) * (1.0 + float(power_margin))
            )
            lam_max_design = max(lam_max_design, lam_min_design)

        # Choose bounded coefficients by predicted contraction on [lam_min_design, lam_max_design].
        a, b, c, ell_chosen = pick_capped_coeffs_by_pred(
            lam_min=lam_min_design,
            lam_max=lam_max_design,
            ell0=ell0,
            a_cap=a_cap,
            b_cap=b_cap,
            c_cap=c_cap,
            grid=ell_grid,
        )

        # Solve U = (aI + bS)(I + cS)^(-1)
        def solve():
            M = Id_mat + float(c) * S
            RHS = float(a) * Id_mat + float(b) * S
            L, _used = chol_on_M_with_jitter(
                M,
                jitter_rel=jitter_rel,
                max_tries=10,
                allow_fp64_fallback=allow_fp64_fallback,
                fp64_eps_rel=1e-12,
            )
            if L.dtype == torch.float64:
                U64 = torch.cholesky_solve(RHS.double(), L)
                return U64.float()
            return torch.cholesky_solve(RHS, L)

        ms_solve, U = cuda_time_ms(solve)
        ms_solve_sum += ms_solve

        # Fixed damping only (no cap-driven tau shrinking)
        U = (1.0 - tau) * Id_mat + tau * U

        ms_upd, X = cuda_time_ms(lambda: (X.float() @ U).to(torch.bfloat16))
        ms_upd_sum += ms_upd

    # If max_steps reached, report final upper bound kO_ub from last iteration.
    ms_total = ms_gram_sum + ms_solve_sum + ms_upd_sum
    return RunSummary(
        False, float(kO_ub), max_steps, ms_gram_sum, ms_solve_sum, ms_upd_sum, ms_total
    )


# ----------------------------- CLI -----------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=["demo", "bank", "suite"], default="suite")

    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--kappa_G", type=float, default=1e7)
    ap.add_argument("--target_kappa_O", type=float, default=1.22474)
    ap.add_argument("--max_steps", type=int, default=5)

    ap.add_argument("--ell0", type=float, default=1e-3)
    ap.add_argument("--eps_scale", type=float, default=1e-2)
    ap.add_argument("--safety", type=float, default=1.01)
    ap.add_argument("--jitter_rel", type=float, default=1e-10)
    ap.add_argument("--tf32", action="store_true")

    ap.add_argument("--psd_repair", action="store_true", default=True)
    ap.add_argument("--no_psd_repair", dest="psd_repair", action="store_false")
    ap.add_argument("--cert_floor_rel", type=float, default=1e-12)

    # Gershgorin reduction fudge. If 0, auto uses max(1e-4, 4*n*eps32).
    ap.add_argument("--sum_fudge", type=float, default=0.0)

    # Coefficient caps
    ap.add_argument("--a_cap", type=float, default=96.0)
    ap.add_argument("--b_cap", type=float, default=8192.0)
    ap.add_argument("--c_cap", type=float, default=8192.0)
    ap.add_argument("--ell_grid", type=int, default=48)

    # Optional block power tightening for lambda_max design
    ap.add_argument("--use_block_power", action="store_true", default=True)
    ap.add_argument("--no_block_power", dest="use_block_power", action="store_false")
    ap.add_argument("--power_iters", type=int, default=8)
    ap.add_argument("--power_block", type=int, default=8)
    ap.add_argument("--power_margin", type=float, default=0.02)

    # Exact eig cert for small n
    ap.add_argument("--eig_threshold", type=int, default=2048)

    # Fallbacks
    ap.add_argument(
        "--no_fp64_fallback",
        dest="allow_fp64_fallback",
        action="store_false",
        default=True,
    )

    # Bank/suite
    ap.add_argument("--bank_size", type=int, default=12)
    ap.add_argument("--suite_cases", type=int, default=6)
    ap.add_argument("--suite_shapes", choices=["kimi_glm5"], default="kimi_glm5")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    print(
        f"device={args.device}  mode={args.mode}  kappa_G<={args.kappa_G:.3g}  target_kappa(O)<={args.target_kappa_O:.6g}"
    )
    print(
        "knobs: "
        f"ell0={args.ell0:g} eps_scale={args.eps_scale:g} safety={args.safety:g} jitter_rel={args.jitter_rel:g} "
        f"tf32={args.tf32} psd_repair={args.psd_repair} cert_floor_rel={args.cert_floor_rel:g} "
        f"a_cap={args.a_cap:g} b_cap={args.b_cap:g} c_cap={args.c_cap:g} ell_grid={args.ell_grid} "
        f"use_block_power={args.use_block_power} power_iters={args.power_iters} power_block={args.power_block} power_margin={args.power_margin} "
        f"eig_threshold={args.eig_threshold} max_steps={args.max_steps}"
    )

    def make_case(m: int, n: int, case_seed: int) -> Tensor:
        spectra = make_spectrum_bank(n, args.kappa_G, bank_size=1, seed=case_seed + n)
        return make_matrix_from_singulars(
            m, spectra[0], seed=case_seed, device=args.device
        )

    def run_case(G: Tensor) -> RunSummary:
        return run_one_case(
            G_storage=G,
            target_kappa_O=args.target_kappa_O,
            max_steps=args.max_steps,
            ell0=args.ell0,
            eps_scale=args.eps_scale,
            safety=args.safety,
            jitter_rel=args.jitter_rel,
            tf32=args.tf32,
            psd_repair=args.psd_repair,
            cert_floor_rel=args.cert_floor_rel,
            sum_fudge=args.sum_fudge,
            a_cap=args.a_cap,
            b_cap=args.b_cap,
            c_cap=args.c_cap,
            ell_grid=args.ell_grid,
            use_block_power=args.use_block_power,
            power_iters=args.power_iters,
            power_block=args.power_block,
            power_margin=args.power_margin,
            eig_threshold=args.eig_threshold,
            allow_fp64_fallback=args.allow_fp64_fallback,
        )

    if args.mode == "demo":
        G = make_case(args.m, args.n, args.seed)
        res = run_case(G)
        print("")
        print(
            f"demo m={args.m} n={args.n}: success={res.success} final_kO_ub={res.final_kO_ub:.6g} steps={res.steps}"
        )
        print(
            f"  ms total={res.ms_total:.3f} (gram={res.ms_gram:.3f} solve={res.ms_solve:.3f} upd={res.ms_upd:.3f})"
        )
        return

    if args.mode == "bank":
        finals: List[float] = []
        steps: List[int] = []
        for i in range(args.bank_size):
            G = make_case(args.m, args.n, args.seed + 1000 + i)
            try:
                res = run_case(G)
                finals.append(res.final_kO_ub)
                steps.append(res.steps)
            except torch.cuda.OutOfMemoryError:
                finals.append(float("inf"))
                steps.append(0)
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        print("")
        print(f"bank summary (N={len(finals)}):")
        print(
            f"  success <= target: {sum(1 for x in finals if x <= args.target_kappa_O)}/{len(finals)}"
        )
        print(
            f"  worst kappa(O): {max(finals):.6g}  median: {pct(finals, 0.5):.6g}  p90: {pct(finals, 0.9):.6g}"
        )
        print(
            f"  steps median: {pct([float(x) for x in steps], 0.5):.6g}  p90: {pct([float(x) for x in steps], 0.9):.6g}"
        )
        return

    # suite
    shapes = (
        suite_shapes_kimi_glm5()
        if args.suite_shapes == "kimi_glm5"
        else [(args.m, args.n)]
    )

    for m, n in shapes:
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            print(
                f"\nshape m={m} n={n}  (cuda mem free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB)"
            )
        else:
            print(f"\nshape m={m} n={n}")

        finals: List[float] = []
        steps_used: List[int] = []
        ms_total: List[float] = []
        ms_gram: List[float] = []
        ms_solve: List[float] = []
        ms_upd: List[float] = []
        successes = 0

        t0 = time.time()
        for i in range(args.suite_cases):
            try:
                G = make_case(m, n, args.seed + 10000 + i)
                res = run_case(G)
                finals.append(res.final_kO_ub)
                steps_used.append(res.steps)
                successes += int(res.final_kO_ub <= args.target_kappa_O)
                ms_total.append(res.ms_total)
                ms_gram.append(res.ms_gram)
                ms_solve.append(res.ms_solve)
                ms_upd.append(res.ms_upd)
            except torch.cuda.OutOfMemoryError:
                print(f"  case {i:02d} OOM (skipping)")
                finals.append(float("inf"))
                steps_used.append(0)
                ms_total.append(float("inf"))
                ms_gram.append(float("inf"))
                ms_solve.append(float("inf"))
                ms_upd.append(float("inf"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as ex:
                print(f"  case {i:02d} FAILED: {type(ex).__name__}: {ex}")
                finals.append(float("inf"))
                steps_used.append(0)
                ms_total.append(float("inf"))
                ms_gram.append(float("inf"))
                ms_solve.append(float("inf"))
                ms_upd.append(float("inf"))

        dt = time.time() - t0
        print(f"  ran {args.suite_cases} cases in {dt:.2f}s")
        print(f"  success <= target: {successes}/{args.suite_cases}")
        print(
            f"  worst kappa(O): {max(finals):.6g}  median: {pct(finals, 0.5):.6g}  p90: {pct(finals, 0.9):.6g}"
        )
        print(
            f"  steps median: {pct([float(x) for x in steps_used], 0.5):.6g}  p90: {pct([float(x) for x in steps_used], 0.9):.6g}"
        )
        print(
            f"  ms total median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}"
        )
        print(
            f"    ms gram  median: {pct(ms_gram, 0.5):.3f}  p90: {pct(ms_gram, 0.9):.3f}"
        )
        print(
            f"    ms solve  median: {pct(ms_solve, 0.5):.3f}  p90: {pct(ms_solve, 0.9):.3f}"
        )
        print(
            f"    ms upd   median: {pct(ms_upd, 0.5):.3f}  p90: {pct(ms_upd, 0.9):.3f}"
        )


if __name__ == "__main__":
    main()
