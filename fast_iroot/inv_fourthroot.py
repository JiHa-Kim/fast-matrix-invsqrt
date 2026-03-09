#!/usr/bin/env python3
# action_invquarter_composite_minimax_actioncert.py
#
# Correctness-first action-only baseline for computing G P^(-1/4)
# with a principled composite rational iteration.
#
# Main design:
#   - P is any SPD matrix; we do NOT assume P = G^T G.
#   - We target the action directly by maintaining B_k = G Z_k.
#   - The small-side iteration is the explicit type-(1,0) rational minimax
#     p-th-root iteration of Gawlik, composed across steps.
#   - For p = 4, the inverse-root factor is
#         h(z, alpha) = 4 mu(alpha)^3 / (z + 3 mu(alpha)^4),
#     where
#         mu(alpha)^4 = (alpha - alpha^4) / (3 (1 - alpha)).
#   - This gives the updates
#         Z_{k+1} = h(M_k, alpha_k) Z_k,
#         B_{k+1} = B_k h(M_k, alpha_k),
#         M_{k+1} = h(M_k, alpha_k)^4 M_k,
#     with M_k = Z_k^4 (P / tau), tau = lambda_max(P).
#   - Final scaling follows Algorithm 5.1 of Gawlik (2019):
#         Z_tilde = tau^{-1/4} * (1 + alpha_k) / (2 alpha_k) * Z_k,
#         B_tilde = tau^{-1/4} * (1 + alpha_k) / (2 alpha_k) * B_k.
#   - Success is based ONLY on certified action-relative error, not condition number.
#
# This is the p = 4 composite-rational analogue of switching from a crude
# coefficient fit to a mathematically derived iteration.

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

P_ROOT = 4


# ----------------------------- utilities ------------------------------------


def symmetrize(A: Tensor) -> Tensor:
    return 0.5 * (A + A.T)


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


# ----------------------------- fp64 SPD ops ---------------------------------


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


# ----------------------------- composite rational step ----------------------


@torch.no_grad()
def mu_from_alpha_p4(alpha: float) -> float:
    alpha = float(min(max(alpha, 1e-300), 1.0))
    if alpha >= 1.0 - 1e-15:
        return 1.0
    # mu(alpha)^4 = (alpha - alpha^4) / (3 (1 - alpha)) = alpha (1 + alpha + alpha^2) / 3
    mu4 = alpha * (1.0 + alpha + alpha * alpha) / 3.0
    mu4 = min(max(mu4, 1e-300), 1.0)
    return float(mu4**0.25)


@torch.no_grad()
def alpha_next_p4(alpha: float, mu: float) -> float:
    alpha = float(alpha)
    mu = float(mu)
    num = 4.0 * alpha * (mu**3)
    den = 3.0 * (mu**4) + (alpha**4)
    out = num / max(den, 1e-300)
    return float(min(max(out, 1e-300), 1.0))


@torch.no_grad()
def build_w_from_M_p4(
    M: Tensor, alpha: float, solve_jitter_rel: float
) -> Tuple[Tensor, float, float, float]:
    """
    For the explicit type-(1,0) p=4 minimax step,
      h(z, alpha) = 4 mu^3 / (z + 3 mu^4).
    Returns W = h(M, alpha), the shift b = 3 mu^4, mu, and Cholesky jitter shift.
    """
    M = symmetrize(M.to(torch.float64))
    n = M.shape[0]
    I = torch.eye(n, device=M.device, dtype=torch.float64)

    mu = mu_from_alpha_p4(alpha)
    a1 = 4.0 * (mu**3)
    b1 = 3.0 * (mu**4)

    A = symmetrize(M + b1 * I)
    L, shift = chol_with_jitter_fp64(A, jitter_rel=solve_jitter_rel)
    W = a1 * torch.cholesky_solve(I, L)
    W = symmetrize(W)
    return W, float(b1), float(mu), float(shift)


@torch.no_grad()
def update_M_p4(M: Tensor, W: Tensor) -> Tensor:
    W2 = symmetrize(W @ W)
    W4 = symmetrize(W2 @ W2)
    return symmetrize(W4 @ M)


@torch.no_grad()
def apply_right_chunked(
    Y: Tensor, Q: Tensor, chunk_rows: int, out_dtype: torch.dtype
) -> Tensor:
    m, n = Y.shape
    out = torch.empty((m, n), device=Y.device, dtype=out_dtype)
    Q64 = Q.to(torch.float64)
    for i in range(0, m, chunk_rows):
        Yi = Y[i : i + chunk_rows].float().to(torch.float64)
        Zi = Yi @ Q64
        out[i : i + chunk_rows] = Zi.to(out_dtype)
    return out


# ----------------------------- action certification -------------------------


@dataclasses.dataclass
class ActionCert:
    action_rel_cert: float
    action_rel_exact: float
    resid_M_cert: float
    resid_M_exact: float


@torch.no_grad()
def cert_action_rel_from_M(
    M: Tensor,
    cert_mode: str,
    exact_threshold: int,
) -> ActionCert:
    M = symmetrize(M.to(torch.float64))
    n = M.shape[0]
    I = torch.eye(n, device=M.device, dtype=torch.float64)

    use_exact = (cert_mode == "exact") or (cert_mode == "auto" and n <= exact_threshold)

    if use_exact:
        evals = torch.linalg.eigvalsh(M)
        lam_min = max(float(evals[0].item()), 1e-300)
        lam_max = max(float(evals[-1].item()), lam_min)
        action_rel = max(1.0 - lam_min**0.25, lam_max**0.25 - 1.0)
        resid_M = max(abs(1.0 - lam_min), abs(lam_max - 1.0))
        return ActionCert(
            action_rel_cert=float(action_rel),
            action_rel_exact=float(action_rel),
            resid_M_cert=float(resid_M),
            resid_M_exact=float(resid_M),
        )

    E = M - I
    eta = float(torch.linalg.matrix_norm(E, ord="fro").item())
    if eta >= 1.0:
        action_rel_ub = float("inf")
    else:
        action_rel_ub = max(1.0 - (1.0 - eta) ** 0.25, (1.0 + eta) ** 0.25 - 1.0)
    return ActionCert(
        action_rel_cert=float(action_rel_ub),
        action_rel_exact=float("nan"),
        resid_M_cert=float(eta),
        resid_M_exact=float("nan"),
    )


# ----------------------------- oracle helpers ------------------------------


@torch.no_grad()
def exact_invquarter_fp64(P: Tensor) -> Tensor:
    evals, U = torch.linalg.eigh(symmetrize(P.to(torch.float64)))
    evals = torch.clamp(evals, min=1e-300)
    X = (U * evals.pow(-0.25)) @ U.T
    return symmetrize(X)


@torch.no_grad()
def exact_quarter_resid_fp64(X: Tensor, P: Tensor) -> float:
    evals, U = torch.linalg.eigh(symmetrize(P.to(torch.float64)))
    evals = torch.clamp(evals, min=1e-300)
    P18 = (U * evals.pow(0.125)) @ U.T
    S = symmetrize(P18 @ X.to(torch.float64) @ P18)
    e = torch.linalg.eigvalsh(S)
    lam_min = float(e[0].item())
    lam_max = float(e[-1].item())
    return float(max(abs(1.0 - lam_min), abs(lam_max - 1.0)))


# ----------------------------- synthetic inputs ----------------------------


def make_spd_from_eigs(
    eigs: Tensor,
    seed: int,
    device: str,
    storage_dtype: torch.dtype,
) -> Tensor:
    n = int(eigs.numel())
    seed_all(seed)
    Q, _ = torch.linalg.qr(
        torch.randn(n, n, device=device, dtype=torch.float64),
        mode="reduced",
    )
    P = (Q * eigs.to(device=device, dtype=torch.float64)) @ Q.T
    P = symmetrize(P)
    return P.to(dtype=storage_dtype)


def make_tall_random(
    m: int,
    n: int,
    seed: int,
    device: str,
    storage_dtype: torch.dtype,
) -> Tensor:
    seed_all(seed)
    G = torch.randn(m, n, device=device, dtype=torch.float32)
    return G.to(dtype=storage_dtype)


def make_eig_bank(n: int, kappa_P: float, bank_size: int, seed: int) -> List[Tensor]:
    lam_max = 1.0
    lam_min = 1.0 / float(kappa_P)
    out: List[Tensor] = []

    out.append(
        torch.logspace(0.0, math.log10(lam_min), n, base=10.0, dtype=torch.float64)
    )

    t = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        logs1 = math.log(lam_max) + (math.log(lam_min) - math.log(lam_max)) * (t**p)
        logs2 = math.log(lam_max) + (math.log(lam_min) - math.log(lam_max)) * (
            1.0 - (1.0 - t) ** p
        )
        out.append(torch.exp(logs1))
        out.append(torch.exp(logs2))

    for frac in [1 / n, 2 / n, 4 / n, 8 / n, 0.1, 0.25, 0.5, 0.75, 0.9]:
        r = max(1, min(n - 1, int(round(frac * n))))
        d = torch.full((n,), lam_min, dtype=torch.float64)
        d[:r] = lam_max
        out.append(d)

    rng = random.Random(seed)
    while len(out) < bank_size:
        u = sorted([rng.random() for _ in range(n)], reverse=True)
        logs = torch.tensor([math.log(lam_min) * x for x in u], dtype=torch.float64)
        d = torch.exp(logs)
        d[0] = lam_max
        d[-1] = lam_min
        out.append(d)

    return out[:bank_size]


def suite_shapes_default() -> List[Tuple[int, int]]:
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


# ----------------------------- core run ------------------------------------


@dataclasses.dataclass
class RunSummary:
    success: bool
    action_rel_cert: float
    resid_M_cert: float
    resid_M_exact: float
    alpha_final: float
    alpha_pred_action_rel: float
    steps: int
    guards: int
    oracle_action_rel_fro: float
    oracle_action_rel_spec: float
    oracle_replay_rel_fro: float
    oracle_root_rel_fro: float
    oracle_quarter_resid: float
    ms_scale: float
    ms_small: float
    ms_apply: float
    ms_cert: float
    ms_oracle: float
    ms_total: float


@torch.no_grad()
def run_one_case(
    G_storage: Tensor,
    P_storage: Tensor,
    target_action_rel: float,
    max_steps: int,
    iter_dtype: torch.dtype,
    cert_mode: str,
    exact_threshold: int,
    rhs_chunk_rows: int,
    solve_jitter_rel: float,
    oracle_mode: str,
    oracle_n_max: int,
) -> RunSummary:
    P_honest, spd_shift = make_spd_honest_fp64(P_storage, jitter_rel=solve_jitter_rel)
    n = P_honest.shape[0]

    ms_scale, (lam_min, lam_max) = cuda_time_ms(
        lambda: init_spectrum_exact_fp64(P_honest)
    )
    tau = float(lam_max)
    alpha = float((lam_min / lam_max) ** 0.25)

    M = symmetrize(P_honest / tau)
    Z = torch.eye(n, device=P_honest.device, dtype=torch.float64)
    B = G_storage.to(iter_dtype)

    ms_small_sum = 0.0
    ms_apply_sum = 0.0
    ms_cert_sum = 0.0
    guards = int(spd_shift > 0.0)

    final_action_rel_cert = float("inf")
    final_resid_M_cert = float("inf")
    final_resid_M_exact = float("nan")

    for it in range(1, max_steps + 1):
        ms_small, (W, _b1, mu, chol_shift) = cuda_time_ms(
            lambda: build_w_from_M_p4(
                M=M, alpha=alpha, solve_jitter_rel=solve_jitter_rel
            )
        )
        ms_small_sum += ms_small
        guards += int(chol_shift > 0.0)

        ms_apply, B = cuda_time_ms(
            lambda: apply_right_chunked(B, W, rhs_chunk_rows, iter_dtype)
        )
        ms_apply_sum += ms_apply

        # Small-side exact state.
        Z = symmetrize(W @ Z)
        M = update_M_p4(M, W)
        alpha = alpha_next_p4(alpha, mu)

        def cert_step():
            # Theorem-based certificate from Gawlik's Corollary 3.4 for Z_tilde A^{1/p} - I.
            eps = (1.0 - alpha) / (1.0 + alpha)
            cert_theory = float(eps / max(1.0 - eps, 1e-300))

            # A posteriori exact/bound diagnostic from M_tilde = Z_tilde^4 P.
            scale = (tau**-0.25) * ((1.0 + alpha) / (2.0 * alpha))
            Zt = scale * Z
            Z2 = symmetrize(Zt @ Zt)
            Z4 = symmetrize(Z2 @ Z2)
            Mtilde = symmetrize(Z4 @ P_honest)
            cert_post = cert_action_rel_from_M(
                Mtilde, cert_mode=cert_mode, exact_threshold=exact_threshold
            )
            return cert_theory, cert_post, float(eps)

        ms_cert, (cert_theory, cert_post, eps) = cuda_time_ms(cert_step)
        ms_cert_sum += ms_cert

        final_action_rel_cert = cert_theory
        final_resid_M_cert = float(cert_post.resid_M_cert)
        final_resid_M_exact = float(cert_post.resid_M_exact)

        if final_action_rel_cert <= target_action_rel:
            break

    alpha_pred_action_rel = float((1.0 - alpha) / (1.0 + alpha))
    scale = (tau**-0.25) * ((1.0 + alpha) / (2.0 * alpha))
    Z_tilde = scale * Z
    B_tilde = B.float().to(torch.float64) * scale

    ms_oracle = 0.0
    oracle_action_rel_fro = float("nan")
    oracle_action_rel_spec = float("nan")
    oracle_replay_rel_fro = float("nan")
    oracle_root_rel_fro = float("nan")
    oracle_quarter_resid = float("nan")

    do_oracle = oracle_mode == "on" or (oracle_mode == "auto" and n <= oracle_n_max)

    if do_oracle:

        def oracle_step():
            X_exact = exact_invquarter_fp64(P_honest)
            Y_exact = G_storage.float().to(torch.float64) @ X_exact
            Y_replay = G_storage.float().to(torch.float64) @ Z_tilde
            return (
                rel_fro(B_tilde, Y_exact),
                rel_spec(B_tilde, Y_exact),
                rel_fro(B_tilde, Y_replay),
                rel_fro(Z_tilde, X_exact),
                exact_quarter_resid_fp64(Z_tilde, P_honest),
            )

        ms_oracle, oracle_vals = cuda_time_ms(oracle_step)
        (
            oracle_action_rel_fro,
            oracle_action_rel_spec,
            oracle_replay_rel_fro,
            oracle_root_rel_fro,
            oracle_quarter_resid,
        ) = oracle_vals

    ms_total = ms_scale + ms_small_sum + ms_apply_sum + ms_cert_sum + ms_oracle
    return RunSummary(
        success=final_action_rel_cert <= target_action_rel,
        action_rel_cert=final_action_rel_cert,
        resid_M_cert=final_resid_M_cert,
        resid_M_exact=final_resid_M_exact,
        alpha_final=float(alpha),
        alpha_pred_action_rel=float(alpha_pred_action_rel),
        steps=it,
        guards=guards,
        oracle_action_rel_fro=float(oracle_action_rel_fro),
        oracle_action_rel_spec=float(oracle_action_rel_spec),
        oracle_replay_rel_fro=float(oracle_replay_rel_fro),
        oracle_root_rel_fro=float(oracle_root_rel_fro),
        oracle_quarter_resid=float(oracle_quarter_resid),
        ms_scale=ms_scale,
        ms_small=ms_small_sum,
        ms_apply=ms_apply_sum,
        ms_cert=ms_cert_sum,
        ms_oracle=ms_oracle,
        ms_total=ms_total,
    )


# ----------------------------- CLI ------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=["demo", "bank", "suite"], default="suite")

    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--kappa_P", type=float, default=1e7)
    ap.add_argument("--target_action_rel", type=float, default=1e-3)
    ap.add_argument("--max_steps", type=int, default=6)

    ap.add_argument(
        "--input_dtype", choices=["float64", "float32", "bfloat16"], default="float32"
    )
    ap.add_argument("--iter_dtype", choices=["float32", "bfloat16"], default="float32")

    ap.add_argument("--cert_mode", choices=["auto", "exact", "bound"], default="auto")
    ap.add_argument("--exact_threshold", type=int, default=1024)
    ap.add_argument("--rhs_chunk_rows", type=int, default=2048)
    ap.add_argument("--solve_jitter_rel", type=float, default=1e-15)

    ap.add_argument("--oracle_mode", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--oracle_n_max", type=int, default=512)

    ap.add_argument("--bank_size", type=int, default=12)
    ap.add_argument("--suite_cases", type=int, default=6)
    ap.add_argument("--suite_shapes", choices=["default"], default="default")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if args.input_dtype == "float64":
        input_dtype = torch.float64
    elif args.input_dtype == "float32":
        input_dtype = torch.float32
    else:
        input_dtype = torch.bfloat16

    iter_dtype = torch.float32 if args.iter_dtype == "float32" else torch.bfloat16

    print(
        f"device={args.device}  mode={args.mode}  target=G P^(-1/4)  "
        f"kappa_P<={args.kappa_P:.3g}  target_action_rel<={args.target_action_rel:.6g}"
    )
    print(
        "knobs: "
        f"max_steps={args.max_steps} input_dtype={args.input_dtype} iter_dtype={args.iter_dtype} "
        f"cert_mode={args.cert_mode} exact_threshold={args.exact_threshold} "
        f"rhs_chunk_rows={args.rhs_chunk_rows} oracle_mode={args.oracle_mode} oracle_n_max={args.oracle_n_max} "
        f"solve_jitter_rel={args.solve_jitter_rel:g}"
    )
    print("method: explicit composite type-(1,0) rational minimax iteration for p=4")

    def make_case(
        m: int, n: int, case_seed: int, case_idx: int
    ) -> Tuple[Tensor, Tensor]:
        bank = make_eig_bank(
            n,
            args.kappa_P,
            bank_size=max(args.bank_size, args.suite_cases, 8),
            seed=case_seed + 17 * n,
        )
        eigs = bank[case_idx % len(bank)]
        P = make_spd_from_eigs(
            eigs=eigs, seed=case_seed, device=args.device, storage_dtype=input_dtype
        )
        G = make_tall_random(
            m=m, n=n, seed=case_seed + 1, device=args.device, storage_dtype=input_dtype
        )
        return G, P

    def run_case(G: Tensor, P: Tensor) -> RunSummary:
        return run_one_case(
            G_storage=G,
            P_storage=P,
            target_action_rel=args.target_action_rel,
            max_steps=args.max_steps,
            iter_dtype=iter_dtype,
            cert_mode=args.cert_mode,
            exact_threshold=args.exact_threshold,
            rhs_chunk_rows=args.rhs_chunk_rows,
            solve_jitter_rel=args.solve_jitter_rel,
            oracle_mode=args.oracle_mode,
            oracle_n_max=args.oracle_n_max,
        )

    if args.mode == "demo":
        G, P = make_case(args.m, args.n, args.seed, 0)
        res = run_case(G, P)
        print("")
        print(
            f"demo m={args.m} n={args.n}: success={res.success} "
            f"action_rel_cert={res.action_rel_cert:.6g} "
            f"resid(M)_cert={res.resid_M_cert:.6g} alpha_final={res.alpha_final:.6g} "
            f"alpha_pred_action_rel={res.alpha_pred_action_rel:.6g} "
            f"steps={res.steps} guards={res.guards}"
        )
        if math.isfinite(res.oracle_action_rel_fro):
            print(
                f"  oracle action rel fro={res.oracle_action_rel_fro:.6g} spec={res.oracle_action_rel_spec:.6g} "
                f"replay={res.oracle_replay_rel_fro:.6g} root={res.oracle_root_rel_fro:.6g} "
                f"quarter_resid={res.oracle_quarter_resid:.6g}"
            )
        print(
            f"  ms total={res.ms_total:.3f} "
            f"(scale={res.ms_scale:.3f} small={res.ms_small:.3f} apply={res.ms_apply:.3f} cert={res.ms_cert:.3f} oracle={res.ms_oracle:.3f})"
        )
        return

    if args.mode == "bank":
        finals = []
        residuals = []
        alpha_preds = []
        steps = []
        guards = []
        ms_total = []
        oracle_action_fro = []
        oracle_action_spec = []
        oracle_replay = []
        oracle_root = []
        oracle_quarter = []

        for i in range(args.bank_size):
            try:
                G, P = make_case(args.m, args.n, args.seed + 1000 + i, i)
                res = run_case(G, P)
                finals.append(res.action_rel_cert)
                residuals.append(res.resid_M_cert)
                alpha_preds.append(res.alpha_pred_action_rel)
                steps.append(res.steps)
                guards.append(res.guards)
                ms_total.append(res.ms_total)
                oracle_action_fro.append(res.oracle_action_rel_fro)
                oracle_action_spec.append(res.oracle_action_rel_spec)
                oracle_replay.append(res.oracle_replay_rel_fro)
                oracle_root.append(res.oracle_root_rel_fro)
                oracle_quarter.append(res.oracle_quarter_resid)
                del G, P
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                finals.append(float("inf"))
                residuals.append(float("inf"))
                alpha_preds.append(float("inf"))
                steps.append(0)
                guards.append(0)
                ms_total.append(float("inf"))
                oracle_action_fro.append(float("nan"))
                oracle_action_spec.append(float("nan"))
                oracle_replay.append(float("nan"))
                oracle_root.append(float("nan"))
                oracle_quarter.append(float("nan"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        print("")
        print(f"bank summary (N={len(finals)}):")
        print(
            f"  success <= target action rel: {sum(1 for x in finals if x <= args.target_action_rel)}/{len(finals)}"
        )
        print(
            f"  worst action rel cert: {max(finals):.6g}  median: {pct(finals, 0.5):.6g}  p90: {pct(finals, 0.9):.6g}"
        )
        print(
            f"  resid(M)_cert median: {pct(residuals, 0.5):.6g}  p90: {pct(residuals, 0.9):.6g}"
        )
        print(
            f"  alpha-pred action rel median: {pct(alpha_preds, 0.5):.6g}  p90: {pct(alpha_preds, 0.9):.6g}"
        )
        print(f"  steps median: {pct(steps, 0.5):.6g}  p90: {pct(steps, 0.9):.6g}")
        print(f"  guards median: {pct(guards, 0.5):.6g}  p90: {pct(guards, 0.9):.6g}")
        if any(math.isfinite(x) for x in oracle_action_fro):
            print(
                f"  oracle action rel_fro median: {pct(oracle_action_fro, 0.5):.6g}  p90: {pct(oracle_action_fro, 0.9):.6g}"
            )
            print(
                f"  oracle action rel_spec median: {pct(oracle_action_spec, 0.5):.6g}  p90: {pct(oracle_action_spec, 0.9):.6g}"
            )
            print(
                f"  oracle replay rel_fro median: {pct(oracle_replay, 0.5):.6g}  p90: {pct(oracle_replay, 0.9):.6g}"
            )
            print(
                f"  oracle root rel_fro median: {pct(oracle_root, 0.5):.6g}  p90: {pct(oracle_root, 0.9):.6g}"
            )
            print(
                f"  oracle quarter resid median: {pct(oracle_quarter, 0.5):.6g}  p90: {pct(oracle_quarter, 0.9):.6g}"
            )
        print(
            f"  ms total median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}"
        )
        return

    shapes = (
        suite_shapes_default() if args.suite_shapes == "default" else [(args.m, args.n)]
    )

    for m, n in shapes:
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            print(
                f"\nshape m={m} n={n}  (cuda mem free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB)"
            )
        else:
            print(f"\nshape m={m} n={n}")

        finals = []
        residuals = []
        alpha_preds = []
        steps = []
        guards = []
        ms_total = []
        ms_scale = []
        ms_small = []
        ms_apply = []
        ms_cert = []
        ms_oracle = []
        oracle_action_fro = []
        oracle_action_spec = []
        oracle_replay = []
        oracle_root = []
        oracle_quarter = []
        successes = 0

        t0 = time.time()
        for i in range(args.suite_cases):
            try:
                G, P = make_case(m, n, args.seed + 10000 + i, i)
                res = run_case(G, P)
                finals.append(res.action_rel_cert)
                residuals.append(res.resid_M_cert)
                alpha_preds.append(res.alpha_pred_action_rel)
                steps.append(res.steps)
                guards.append(res.guards)
                successes += int(res.success)
                ms_total.append(res.ms_total)
                ms_scale.append(res.ms_scale)
                ms_small.append(res.ms_small)
                ms_apply.append(res.ms_apply)
                ms_cert.append(res.ms_cert)
                ms_oracle.append(res.ms_oracle)
                oracle_action_fro.append(res.oracle_action_rel_fro)
                oracle_action_spec.append(res.oracle_action_rel_spec)
                oracle_replay.append(res.oracle_replay_rel_fro)
                oracle_root.append(res.oracle_root_rel_fro)
                oracle_quarter.append(res.oracle_quarter_resid)
                del G, P
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  case {i:02d} OOM (skipping)")
                finals.append(float("inf"))
                residuals.append(float("inf"))
                alpha_preds.append(float("inf"))
                steps.append(0)
                guards.append(0)
                ms_total.append(float("inf"))
                ms_scale.append(float("inf"))
                ms_small.append(float("inf"))
                ms_apply.append(float("inf"))
                ms_cert.append(float("inf"))
                ms_oracle.append(float("inf"))
                oracle_action_fro.append(float("nan"))
                oracle_action_spec.append(float("nan"))
                oracle_replay.append(float("nan"))
                oracle_root.append(float("nan"))
                oracle_quarter.append(float("nan"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as ex:
                print(f"  case {i:02d} FAILED: {type(ex).__name__}: {ex}")
                finals.append(float("inf"))
                residuals.append(float("inf"))
                alpha_preds.append(float("inf"))
                steps.append(0)
                guards.append(0)
                ms_total.append(float("inf"))
                ms_scale.append(float("inf"))
                ms_small.append(float("inf"))
                ms_apply.append(float("inf"))
                ms_cert.append(float("inf"))
                ms_oracle.append(float("inf"))
                oracle_action_fro.append(float("nan"))
                oracle_action_spec.append(float("nan"))
                oracle_replay.append(float("nan"))
                oracle_root.append(float("nan"))
                oracle_quarter.append(float("nan"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        dt = time.time() - t0
        print(f"  ran {args.suite_cases} cases in {dt:.2f}s")
        print(f"  success <= target action rel: {successes}/{args.suite_cases}")
        print(
            f"  action rel cert median: {pct(finals, 0.5):.6g}  p90: {pct(finals, 0.9):.6g}"
        )
        print(
            f"  resid(M)_cert median: {pct(residuals, 0.5):.6g}  p90: {pct(residuals, 0.9):.6g}"
        )
        print(
            f"  alpha-pred action rel median: {pct(alpha_preds, 0.5):.6g}  p90: {pct(alpha_preds, 0.9):.6g}"
        )
        print(f"  steps median: {pct(steps, 0.5):.6g}  p90: {pct(steps, 0.9):.6g}")
        print(f"  guards median: {pct(guards, 0.5):.6g}  p90: {pct(guards, 0.9):.6g}")
        if any(math.isfinite(x) for x in oracle_action_fro):
            print(
                f"  oracle action rel_fro median: {pct(oracle_action_fro, 0.5):.6g}  p90: {pct(oracle_action_fro, 0.9):.6g}"
            )
            print(
                f"  oracle action rel_spec median: {pct(oracle_action_spec, 0.5):.6g}  p90: {pct(oracle_action_spec, 0.9):.6g}"
            )
            print(
                f"  oracle replay rel_fro median: {pct(oracle_replay, 0.5):.6g}  p90: {pct(oracle_replay, 0.9):.6g}"
            )
            print(
                f"  oracle root rel_fro median: {pct(oracle_root, 0.5):.6g}  p90: {pct(oracle_root, 0.9):.6g}"
            )
            print(
                f"  oracle quarter resid median: {pct(oracle_quarter, 0.5):.6g}  p90: {pct(oracle_quarter, 0.9):.6g}"
            )
        else:
            print("  oracle metrics: skipped on all cases")
        print(
            f"  ms total median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}"
        )
        print(
            f"    ms scale median: {pct(ms_scale, 0.5):.3f}  p90: {pct(ms_scale, 0.9):.3f}"
        )
        print(
            f"    ms small median: {pct(ms_small, 0.5):.3f}  p90: {pct(ms_small, 0.9):.3f}"
        )
        print(
            f"    ms apply median: {pct(ms_apply, 0.5):.3f}  p90: {pct(ms_apply, 0.9):.3f}"
        )
        print(
            f"    ms cert  median: {pct(ms_cert, 0.5):.3f}  p90: {pct(ms_cert, 0.9):.3f}"
        )
        print(
            f"    ms oracle median: {pct(ms_oracle, 0.5):.3f}  p90: {pct(ms_oracle, 0.9):.3f}"
        )


if __name__ == "__main__":
    main()
