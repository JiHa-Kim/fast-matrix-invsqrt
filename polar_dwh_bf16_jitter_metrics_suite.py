#!/usr/bin/env python3
"""
polar_dwh_bf16_jitter_metrics_suite.py

Extends the BF16-friendly direct polar (DWH/QDWH-style) iteration with:
  - rich orthogonalization + polar-quality metrics
  - a standard stress/benchmark suite over many (m,n) shapes ("Kimi K2/GLM5 level")
  - CSV/JSONL logging for later analysis

Core algorithm (unchanged):
  - X stored bf16, S = X^T X formed fp32
  - trace-centering each iter
  - Cholesky only on M = I + cS
  - jitter (+delta I) only on M and only if Cholesky fails

This script is meant to be run on CUDA.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import time
from typing import Dict, List, Tuple

import torch

Tensor = torch.Tensor


# ----------------------------- utilities -----------------------------------


def symmetrize(A: Tensor) -> Tensor:
    return 0.5 * (A + A.T)


def cuda_time_ms(fn):
    if not torch.cuda.is_available():
        out = fn()
        return 0.0, out
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), out


def chol_with_jitter_on_M(
    M: Tensor, jitter_rel: float, max_tries: int = 12
) -> Tuple[Tensor, float]:
    """
    Robust Cholesky on SPD-ish matrix M by adding diagonal jitter only if needed.

    IMPORTANT: This is "jitter" for factorization robustness, not problem regularization.
    It is applied ONLY to M (the matrix being factorized), and ONLY if Cholesky fails.

    Strategy:
      1) Try fp32 cholesky_ex with delta = jitter_rel * tr(M)/n, doubling delta on failure.
      2) If fp32 still fails, do an eigvalsh-based minimal shift in fp64:
           shift = max(0, -lambda_min(M)) + eps
         then Cholesky in fp64 and return fp64 factor (caller can solve in fp64 and cast back).

    Returns:
      (L, used_delta) where L has same dtype as M unless fp64 fallback is used by caller.
    """
    M = symmetrize(M)
    if not torch.isfinite(M).all():
        raise RuntimeError("Non-finite entries in M before Cholesky")

    n = M.shape[0]
    Id_mat = torch.eye(n, device=M.device, dtype=M.dtype)

    tr = torch.trace(M).abs()
    base = (tr / n) * float(jitter_rel) if jitter_rel > 0.0 else tr.new_tensor(0.0)
    delta = float(base.item()) if jitter_rel > 0.0 else 0.0

    # fp32 attempts with doubling jitter
    for _ in range(max_tries):
        Mt = M if delta == 0.0 else (M + delta * Id_mat)
        L, info = torch.linalg.cholesky_ex(Mt)
        if int(info.item()) == 0:
            return L, delta
        if jitter_rel <= 0.0:
            break
        delta = delta * 2.0 if delta > 0.0 else float(base.item())

    # Signal failure; caller will take the fp64 eig-shift fallback.
    raise RuntimeError("chol_fp32_failed")


def kappa_from_gram(
    S: Tensor,
    psd_clip: bool = True,
    cert_floor_rel: float = 1e-12,
) -> Tuple[float, float, float, float]:
    Sd = symmetrize(S).double()
    evals = torch.linalg.eigvalsh(Sd)
    lam_min_raw = float(evals[0].item())
    lam_max_raw = float(evals[-1].item())

    lam_max = max(lam_max_raw, 0.0)
    lam_min = max(lam_min_raw, 0.0) if psd_clip else lam_min_raw
    lam_min_safe = max(lam_min, cert_floor_rel * lam_max if lam_max > 0.0 else 0.0)

    if lam_max == 0.0:
        return 1.0, lam_min_raw, lam_max_raw, lam_min_safe

    kappa = lam_max / lam_min_safe
    return float(kappa), lam_min_raw, lam_max_raw, lam_min_safe


def psd_shift_gram_inplace(
    S: Tensor,
    cert_floor_rel: float = 1e-12,
    psd_clip: bool = True,
) -> Tuple[Tensor, float, float, float, float]:
    """
    Compute eigvals of S in fp64 and (optionally) apply a minimal PSD shift to S in fp32:

      if lam_min_raw < 0 and psd_clip:
          shift = -lam_min_raw + eps
          S <- S + shift * I

    eps is scaled to lam_max to avoid exactly-zero lam_min after shift.

    Returns:
      (S_shifted, lam_min_raw, lam_max_raw, lam_min_safe, psd_shift)

    lam_min_safe is computed after shift + floor:
      lam_min_safe = max(lam_min_raw + shift, cert_floor_rel * (lam_max_raw + shift))
    """
    S = symmetrize(S)
    n = S.shape[0]
    Sd = S.double()
    evals = torch.linalg.eigvalsh(Sd)
    lam_min_raw = float(evals[0].item())
    lam_max_raw = float(evals[-1].item())

    psd_shift = 0.0
    if psd_clip and lam_min_raw < 0.0:
        # eps scaled to lam_max to keep lam_min_safe positive but tiny
        eps = cert_floor_rel * max(lam_max_raw, 1.0)
        psd_shift = float(-lam_min_raw + eps)
        Id_matd_matd_matd_matd_matd_matd_matd_mat = torch.eye(
            n, device=S.device, dtype=S.dtype
        )
        S = S + psd_shift * Id_matd_matd_matd_matd_matd_matd_matd_mat

    lam_min_adj = lam_min_raw + psd_shift
    lam_max_adj = lam_max_raw + psd_shift
    lam_max_adj = max(lam_max_adj, 0.0)
    lam_min_adj = max(lam_min_adj, 0.0) if psd_clip else lam_min_adj
    lam_min_safe = max(
        lam_min_adj, cert_floor_rel * lam_max_adj if lam_max_adj > 0.0 else 0.0
    )

    return S, lam_min_raw, lam_max_raw, float(lam_min_safe), float(psd_shift)


def gram_metrics(S: Tensor) -> Dict[str, float]:
    """Metrics based on S = X^T X (assumed symmetric). Uses fp64 eigs for some items."""
    n = S.shape[0]
    Id_matd_matd_matd_mat = torch.eye(n, device=S.device, dtype=S.dtype)
    E = symmetrize(S) - Id_matd_matd_matd_mat

    # Frobenius metrics in fp32
    fro_E = float(torch.linalg.norm(E, ord="fro").item())
    fro_S = float(torch.linalg.norm(S, ord="fro").item())
    offdiag = E - torch.diag(torch.diag(E))
    offdiag_fro = float(torch.linalg.norm(offdiag, ord="fro").item())
    max_abs_diag_dev = float((torch.abs(torch.diag(E))).max().item())
    trace_dev = float((torch.trace(S) / n - 1.0).item())

    # Spectral metrics via eigs in fp64
    evals = torch.linalg.eigvalsh(symmetrize(S).double())
    lam_min = float(evals[0].item())
    lam_max = float(evals[-1].item())
    # ||S-I||_2 = max |lambda_i - 1|
    norm2_E = float(torch.max(torch.abs(evals - 1.0)).item())

    # logdet (mean) - PSD clipped for stability
    evals_clip = torch.clamp(evals, min=1e-300)
    mean_logdet = float(torch.log(evals_clip).mean().item())

    return {
        "fro_E": fro_E,
        "fro_S": fro_S,
        "offdiag_fro": offdiag_fro,
        "max_abs_diag_dev": max_abs_diag_dev,
        "trace_dev": trace_dev,
        "lam_min_eig": lam_min,
        "lam_max_eig": lam_max,
        "norm2_E": norm2_E,
        "mean_logdet": mean_logdet,
    }


def polar_first_order_metrics(G: Tensor, X: Tensor) -> Dict[str, float]:
    """
    Polar first-order condition: for exact polar Q, Q^T G is symmetric PSD.
    Use H = X^T G. If X ~ Q, H should be ~ symmetric.
    """
    Gf = G.float()
    Xf = X.float()
    H = Xf.T @ Gf
    Hs = symmetrize(H)
    num = torch.linalg.norm(H - H.T, ord="fro")
    den = torch.linalg.norm(H, ord="fro") + 1e-30
    sym_err = float((num / den).item())

    # Decomposition residual: G ?= X * sym(H)
    R = Gf - Xf @ Hs
    decomp_res = float(
        (
            torch.linalg.norm(R, ord="fro") / (torch.linalg.norm(Gf, ord="fro") + 1e-30)
        ).item()
    )
    return {"sym_err": sym_err, "decomp_res": decomp_res}


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


# ----------------------------- synthetic matrices --------------------------


def make_matrix_from_singulars(
    m: int, singulars: Tensor, seed: int, device: str, storage_dtype=torch.bfloat16
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


# ----------------------------- iteration + metrics -------------------------


@dataclasses.dataclass
class IterLog:
    it: int
    kappa_O: float
    kappa_cert: float
    ell_used: float
    a: float
    b: float
    c: float
    lam_min_raw: float
    lam_max_raw: float
    lam_min_safe: float
    used_jitter: float
    psd_shift: float
    ms_gram: float
    ms_solve: float
    ms_upd: float
    # gram metrics
    norm2_E: float
    fro_E: float
    offdiag_fro: float
    max_abs_diag_dev: float
    trace_dev: float
    mean_logdet: float
    # polar first-order metrics
    sym_err: float
    decomp_res: float


@torch.no_grad()
def run_polar_dwh_with_metrics(
    G_storage: Tensor,
    target_kappa_O: float,
    max_steps: int,
    ell0: float,
    eps_scale: float,
    safety: float,
    jitter_rel: float,
    tf32: bool,
    psd_clip: bool,
    cert_floor_rel: float,
) -> Tuple[Tensor, List[IterLog]]:
    device = G_storage.device
    m, n = G_storage.shape
    Id_matd_mat = torch.eye(n, device=device, dtype=torch.float32)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high")

    Gf = G_storage.float()
    fro = torch.linalg.norm(Gf, ord="fro")
    denom = float(safety) * fro + float(eps_scale)
    X = (Gf / denom).to(torch.bfloat16)

    logs: List[IterLog] = []

    for it in range(1, max_steps + 1):
        ms_gram, S = cuda_time_ms(lambda: symmetrize(X.float().T @ X.float()))

        mu = torch.trace(S) / n
        mu_f = float(mu.item())
        X = (X.float() / math.sqrt(mu_f)).to(torch.bfloat16)
        S = S / mu

        S, lam_min_raw, lam_max_raw, lam_min_safe, psd_shift = psd_shift_gram_inplace(
            S, cert_floor_rel=cert_floor_rel, psd_clip=psd_clip
        )
        # kappa computed from adjusted (shifted) endpoints
        lam_max_adj = max(lam_max_raw + psd_shift, 0.0)
        kappa_cert = (
            1.0 if lam_max_adj == 0.0 else (lam_max_adj / max(lam_min_safe, 1e-300))
        )
        kappa_O = math.sqrt(kappa_cert)

        gm = gram_metrics(S.float())
        pm = polar_first_order_metrics(G_storage, X)

        if kappa_O <= target_kappa_O:
            logs.append(
                IterLog(
                    it=it,
                    kappa_O=kappa_O,
                    kappa_cert=kappa_cert,
                    ell_used=float("nan"),
                    a=1.0,
                    b=0.0,
                    c=0.0,
                    lam_min_raw=lam_min_raw,
                    lam_max_raw=lam_max_raw,
                    lam_min_safe=lam_min_safe,
                    used_jitter=0.0,
                    psd_shift=float(psd_shift),
                    ms_gram=ms_gram,
                    ms_solve=0.0,
                    ms_upd=0.0,
                    norm2_E=gm["norm2_E"],
                    fro_E=gm["fro_E"],
                    offdiag_fro=gm["offdiag_fro"],
                    max_abs_diag_dev=gm["max_abs_diag_dev"],
                    trace_dev=gm["trace_dev"],
                    mean_logdet=gm["mean_logdet"],
                    sym_err=pm["sym_err"],
                    decomp_res=pm["decomp_res"],
                )
            )
            break

        ell_est = math.sqrt(max(lam_min_safe, 0.0))
        ell_used = max(float(ell0), ell_est)
        a, b, c = dwh_coeffs(ell_used)

        def solve():
            M = Id_matd_mat + float(c) * S
            RHS = float(a) * Id_matd_mat + float(b) * S

            try:
                L, used_delta = chol_with_jitter_on_M(M, jitter_rel=jitter_rel)
                U = torch.cholesky_solve(RHS, L)
                return U, used_delta
            except RuntimeError as e:
                # fp64 minimal shift via eigvalsh: shift >= -lambda_min(M) + eps
                if str(e) != "chol_fp32_failed":
                    raise
                Md = symmetrize(M.double())
                evals = torch.linalg.eigvalsh(Md)
                lam_min = float(evals[0].item())
                # eps scaled to mean diag magnitude
                trd = float(torch.trace(Md).abs().item())
                eps = 1e-12 * (trd / n if n > 0 else 1.0)
                shift = max(0.0, -lam_min + eps)

                Id = torch.eye(n, device=M.device, dtype=torch.float64)
                Ld = torch.linalg.cholesky(Md + shift * Id)
                Ud = torch.cholesky_solve(RHS.double(), Ld)
                return Ud.float(), float(shift)

        ms_solve, (U, used_delta) = cuda_time_ms(solve)

        tau = 1.0 / float(safety)
        U = (1.0 - tau) * Id_matd_mat + tau * U
        ms_upd, X = cuda_time_ms(lambda: (X.float() @ U).to(torch.bfloat16))

        logs.append(
            IterLog(
                it=it,
                kappa_O=kappa_O,
                kappa_cert=kappa_cert,
                ell_used=float(ell_used),
                a=float(a),
                b=float(b),
                c=float(c),
                lam_min_raw=float(lam_min_raw),
                lam_max_raw=float(lam_max_raw),
                lam_min_safe=float(lam_min_safe),
                used_jitter=float(used_delta),
                psd_shift=float(psd_shift),
                ms_gram=float(ms_gram),
                ms_solve=float(ms_solve),
                ms_upd=float(ms_upd),
                norm2_E=float(gm["norm2_E"]),
                fro_E=float(gm["fro_E"]),
                offdiag_fro=float(gm["offdiag_fro"]),
                max_abs_diag_dev=float(gm["max_abs_diag_dev"]),
                trace_dev=float(gm["trace_dev"]),
                mean_logdet=float(gm["mean_logdet"]),
                sym_err=float(pm["sym_err"]),
                decomp_res=float(pm["decomp_res"]),
            )
        )

    return X, logs


def pct(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = int(round(p * (len(xs) - 1)))
    i = max(0, min(len(xs) - 1, i))
    return xs[i]


# ----------------------------- suite shapes --------------------------------


def suite_shapes_kimi_glm5() -> List[Tuple[int, int]]:
    """
    A practical suite spanning:
      - medium: (m,n) around 2k x 256
      - large: 8k-32k tall with 1k-8k width (Kimi K2/GLM5-ish)
    Adjust for your GPU memory if needed.
    """
    shapes: List[Tuple[int, int]] = []
    shapes += [(2048, 256), (4096, 256), (8192, 256)]
    shapes += [(8192, 1024), (16384, 1024)]
    shapes += [(8192, 2048), (16384, 2048)]
    shapes += [(28672, 4096), (28672, 7168)]  # common "giant MLP/attn" widths
    shapes += [(32768, 8192)]
    return shapes


# ----------------------------- CLI -----------------------------------------


def print_header():
    print(
        "it  kappa(O)    norm2_E     fro_E   offdiagF  max|dii-1|  sym_err  decomp_res  "
        "ell_used      a          b          c       jitter    ms_gram  ms_solve  ms_upd"
    )


def print_row(r: IterLog):
    print(
        f"{r.it:2d}  "
        f"{r.kappa_O:8.4f}  "
        f"{r.norm2_E:9.3g}  {r.fro_E:9.3g}  {r.offdiag_fro:9.3g}  {r.max_abs_diag_dev:10.3g}  "
        f"{r.sym_err:7.3g}  {r.decomp_res:10.3g}  "
        f"{r.ell_used:7.3g}  "
        f"{r.a:9.3g}  {r.b:9.3g}  {r.c:9.3g}  "
        f"{r.used_jitter:8.2g}  {r.psd_shift:7.2g}  "
        f"{r.ms_gram:7.2f}  {r.ms_solve:7.2f}  {r.ms_upd:7.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=["demo", "bank", "suite"], default="suite")
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--kappa_G", type=float, default=1e7)
    ap.add_argument("--target_kappa_O", type=float, default=math.sqrt(1.5))
    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument("--bank_size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)

    # knobs
    ap.add_argument("--ell0", type=float, default=1e-3)
    ap.add_argument("--eps_scale", type=float, default=1e-2)
    ap.add_argument("--safety", type=float, default=1.01)
    ap.add_argument("--jitter_rel", type=float, default=1e-10)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--psd_clip", action="store_true", default=True)
    ap.add_argument("--no_psd_clip", dest="psd_clip", action="store_false")
    ap.add_argument("--cert_floor_rel", type=float, default=1e-12)

    # suite controls
    ap.add_argument(
        "--suite_cases", type=int, default=6, help="spectra per shape in suite mode"
    )
    ap.add_argument("--suite_shapes", choices=["kimi_glm5"], default="kimi_glm5")
    ap.add_argument(
        "--csv", type=str, default="", help="optional path to write CSV summary"
    )
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    print(
        f"device={args.device}  mode={args.mode}  kappa_G<={args.kappa_G:.3g}  "
        f"target_kappa(O)<={args.target_kappa_O:.6g}"
    )
    print(
        f"knobs: ell0={args.ell0:g} eps_scale={args.eps_scale:g} safety={args.safety:g} "
        f"jitter_rel={args.jitter_rel:g} tf32={args.tf32} psd_clip={args.psd_clip} cert_floor_rel={args.cert_floor_rel:g}"
    )

    if args.mode == "demo":
        spectra = make_spectrum_bank(args.n, args.kappa_G, bank_size=1, seed=args.seed)
        G = make_matrix_from_singulars(
            args.m, spectra[0], seed=args.seed, device=args.device
        )
        _, logs = run_polar_dwh_with_metrics(
            G,
            args.target_kappa_O,
            args.max_steps,
            args.ell0,
            args.eps_scale,
            args.safety,
            args.jitter_rel,
            args.tf32,
            args.psd_clip,
            args.cert_floor_rel,
        )
        print("")
        print_header()
        for r in logs:
            print_row(r)
        print(f"\nfinal kappa(O)={logs[-1].kappa_O:.6g} steps={logs[-1].it}")
        return

    if args.mode == "bank":
        spectra = make_spectrum_bank(
            args.n, args.kappa_G, bank_size=args.bank_size, seed=args.seed
        )
        finals = []
        steps = []
        for i, s in enumerate(spectra):
            G = make_matrix_from_singulars(
                args.m, s, seed=args.seed + 1000 + i, device=args.device
            )
            _, logs = run_polar_dwh_with_metrics(
                G,
                args.target_kappa_O,
                args.max_steps,
                args.ell0,
                args.eps_scale,
                args.safety,
                args.jitter_rel,
                args.tf32,
                args.psd_clip,
                args.cert_floor_rel,
            )
            finals.append(logs[-1].kappa_O)
            steps.append(logs[-1].it)
        print("")
        print(f"bank summary (N={len(finals)}):")
        print(
            f"  success <= target: {sum(1 for x in finals if x <= args.target_kappa_O)}/{len(finals)}"
        )
        print(f"  worst kappa(O): {max(finals):.6g}")
        print(f"  median kappa(O): {pct(finals, 0.5):.6g}")
        print(f"  p90 kappa(O): {pct(finals, 0.9):.6g}")
        print(f"  max steps used: {max(steps)}")
        return

    # suite mode
    if args.suite_shapes == "kimi_glm5":
        shapes = suite_shapes_kimi_glm5()
    else:
        shapes = [(args.m, args.n)]

    # optional CSV
    csv_lines: List[str] = []
    if args.csv:
        csv_lines.append(
            "m,n,cases,successes,worst_kO,median_kO,p90_kO,median_steps,p90_steps,"
            "median_ms_total,p90_ms_total,median_ms_gram,p90_ms_gram,median_ms_solve,p90_ms_solve,median_ms_upd,p90_ms_upd"
        )

    for m, n in shapes:
        spectra = make_spectrum_bank(
            n, args.kappa_G, bank_size=args.suite_cases, seed=args.seed + n
        )
        finals: List[float] = []
        steps_used: List[int] = []
        ms_total: List[float] = []
        ms_gram: List[float] = []
        ms_solve: List[float] = []
        ms_upd: List[float] = []
        successes = 0

        # quick memory info
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            print(
                f"\nshape m={m} n={n}  (cuda mem free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB)"
            )

        t0 = time.time()
        for i, s in enumerate(spectra):
            G = make_matrix_from_singulars(
                m, s, seed=args.seed + 10000 + i, device=args.device
            )
            try:
                _, logs = run_polar_dwh_with_metrics(
                    G,
                    args.target_kappa_O,
                    args.max_steps,
                    args.ell0,
                    args.eps_scale,
                    args.safety,
                    args.jitter_rel,
                    args.tf32,
                    args.psd_clip,
                    args.cert_floor_rel,
                )
                finals.append(logs[-1].kappa_O)
                steps_used.append(logs[-1].it)
                successes += int(logs[-1].kappa_O <= args.target_kappa_O)

                # timing totals: sum per-iter (already includes sync)
                ms_total.append(sum(r.ms_gram + r.ms_solve + r.ms_upd for r in logs))
                ms_gram.append(sum(r.ms_gram for r in logs))
                ms_solve.append(sum(r.ms_solve for r in logs))
                ms_upd.append(sum(r.ms_upd for r in logs))
            except Exception as ex:
                # Record failure but keep the suite running.
                finals.append(float("inf"))
                steps_used.append(0)
                ms_total.append(float("inf"))
                ms_gram.append(float("inf"))
                ms_solve.append(float("inf"))
                ms_upd.append(float("inf"))
                print(f"  case {i:02d} FAILED: {type(ex).__name__}: {ex}")

        dt = time.time() - t0
        print(f"  ran {len(spectra)} cases in {dt:.2f}s")
        print(f"  success <= target: {successes}/{len(spectra)}")
        print(
            f"  worst kappa(O): {max(finals):.6g}  median: {pct(finals, 0.5):.6g}  p90: {pct(finals, 0.9):.6g}"
        )
        print(
            f"  steps median: {pct(steps_used, 0.5):.6g}  p90: {pct(steps_used, 0.9):.6g}"
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

        if args.csv:
            csv_lines.append(
                f"{m},{n},{len(spectra)},{successes},"
                f"{max(finals):.6g},{pct(finals, 0.5):.6g},{pct(finals, 0.9):.6g},"
                f"{pct(steps_used, 0.5):.6g},{pct(steps_used, 0.9):.6g},"
                f"{pct(ms_total, 0.5):.3f},{pct(ms_total, 0.9):.3f},"
                f"{pct(ms_gram, 0.5):.3f},{pct(ms_gram, 0.9):.3f},"
                f"{pct(ms_solve, 0.5):.3f},{pct(ms_solve, 0.9):.3f},"
                f"{pct(ms_upd, 0.5):.3f},{pct(ms_upd, 0.9):.3f}"
            )

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write("\n".join(csv_lines))
        print(f"\nwrote CSV summary to: {args.csv}")


if __name__ == "__main__":
    main()
