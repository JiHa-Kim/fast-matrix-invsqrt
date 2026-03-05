#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import List

import numpy as np

try:
    import torch
except ImportError:
    torch = None

_APPLY_MONO_FN = None
_APPLY_CHEB_FN = None


def load_mats(npz_path: str) -> List[np.ndarray]:
    """
    Expect either:
      - Bs: shape (K,n,n)
      - or a list of arrays saved as B0,B1,...
    """
    z = np.load(npz_path)
    if "Bs" in z:
        Bs = z["Bs"]
        return [Bs[i] for i in range(Bs.shape[0])]
    out = []
    for k in sorted(z.files):
        out.append(z[k])
    return out


def fro_norm(a: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(a, ord="fro")


def jacobi_init(B: torch.Tensor, jacobi_eps: float) -> torch.Tensor:
    d = torch.diagonal(B).to(torch.float32)
    inv_sqrt = torch.rsqrt(d + jacobi_eps)
    Z = torch.diag(inv_sqrt).to(torch.bfloat16)
    return Z


def choose_beta(S: torch.Tensor, mode: str = "fro") -> torch.Tensor:
    # Must be the same for all methods being compared.
    if mode == "fro":
        # Upper bound-ish for lambda_max: ||S||_F >= ||S||_2
        return fro_norm(S).clamp_min(1.0)
    if mode == "trace":
        n = S.shape[0]
        return (torch.trace(S).abs() / n).clamp_min(1.0)
    if mode == "maxdiag":
        return torch.max(torch.diagonal(S)).clamp_min(1.0)
    raise ValueError("beta mode must be fro|trace|maxdiag")


def apply_poly_right_mono(
    Z: torch.Tensor, S: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    """
    Compute Z q(S) with monomial Horner:
      Y = a[d] Z
      for k=d-1..0: Y = Y S + a[k] Z
    """
    d = a.numel() - 1
    Y = a[d] * Z
    for k in range(d - 1, -1, -1):
        Y = Y @ S
        Y = Y + a[k] * Z
    return Y


def apply_poly_right_cheb(
    Z: torch.Tensor, S: torch.Tensor, c: torch.Tensor, ell: float
) -> torch.Tensor:
    """
    Evaluate q(S) = sum_{k=0}^d c[k] T_k(t(S)) on [ell,1], but apply on the right to Z.
    We avoid forming t explicitly by using:
      t = alpha S + beta I,  alpha = 2/(1-ell), beta = -(1+ell)/(1-ell)

    Forward recurrence on ZT_k:
      ZT0 = Z
      ZT1 = Z t = alpha ZS + beta Z
      ZTk = 2 (ZTk-1 t) - ZTk-2
    Accumulate: sum c[k] * ZTk
    """
    d = c.numel() - 1
    alpha = 2.0 / (1.0 - ell)
    beta = -(1.0 + ell) / (1.0 - ell)

    ZT0 = Z
    out = c[0] * ZT0
    if d == 0:
        return out

    ZS = Z @ S
    ZT1 = alpha * ZS + beta * Z
    out = out + c[1] * ZT1

    for k in range(2, d + 1):
        # ZT1 t = alpha (ZT1 S) + beta ZT1
        ZT1S = ZT1 @ S
        ZT2 = 2.0 * (alpha * ZT1S + beta * ZT1) - ZT0
        out = out + c[k] * ZT2
        ZT0, ZT1 = ZT1, ZT2

    return out


@dataclass
class RunOut:
    basis: str
    deg: int
    iters: int
    total_ms: float
    deltaF_final: float
    deltaF_traj: List[float]


def run_precond(
    B: torch.Tensor,
    basis: str,
    deg: int,
    iters: int,
    ell: float,
    coeffs: torch.Tensor,
    ridge: float,
    jacobi_eps: float,
    beta_mode: str,
) -> RunOut:
    n = B.shape[0]
    I_mat = torch.eye(n, device=B.device, dtype=torch.float32)

    # ridge in original coordinates: B <- B + ridge I
    if ridge != 0.0:
        B = B + ridge * I_mat

    Z = jacobi_init(B, jacobi_eps=jacobi_eps)  # bf16
    delta_traj: List[float] = []

    # Get the correctly compiled (or eager) function pointers
    global _APPLY_MONO_FN, _APPLY_CHEB_FN
    apply_mono = _APPLY_MONO_FN if _APPLY_MONO_FN is not None else apply_poly_right_mono
    apply_cheb = _APPLY_CHEB_FN if _APPLY_CHEB_FN is not None else apply_poly_right_cheb

    # Warmup one S build outside timing if you want (kept simple here)
    if B.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        # S = Z^T B Z in fp32
        Zf = Z.to(torch.float32)
        tmp = B @ Zf
        S = Zf.T @ tmp

        deltaF = float(fro_norm(S - I_mat).item())
        delta_traj.append(deltaF)

        beta = choose_beta(S, mode=beta_mode)
        Shat = (S / beta).to(torch.float32)

        # Apply polynomial in fp32 (matmuls dominate anyway), then cast back to bf16 Z
        if basis == "mono":
            Y = apply_mono(Zf, Shat, coeffs)
        else:
            Y = apply_cheb(Zf, Shat, coeffs, ell=ell)

        Znew = (Y / torch.sqrt(beta)).to(torch.bfloat16)
        Z = Znew

    if B.is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return RunOut(
        basis=basis,
        deg=deg,
        iters=iters,
        total_ms=1000.0 * (t1 - t0),
        deltaF_final=delta_traj[-1],
        deltaF_traj=delta_traj,
    )


def microbench_apply(
    n: int,
    basis: str,
    deg: int,
    ell: float,
    coeffs: torch.Tensor,
    device: str,
    reps: int = 200,
) -> float:
    """
    Time only Z <- Z q(S) (no S build), to isolate polynomial evaluation overhead.
    """
    assert torch is not None
    dev = torch.device(device)
    Z = torch.randn(n, n, device=dev, dtype=torch.float32)
    S = torch.randn(n, n, device=dev, dtype=torch.float32)
    S = (S + S.T) * 0.5
    S = S @ S.T  # SPD-ish
    S = S / choose_beta(S, mode="fro")

    global _APPLY_MONO_FN, _APPLY_CHEB_FN
    apply_mono = _APPLY_MONO_FN if _APPLY_MONO_FN is not None else apply_poly_right_mono
    apply_cheb = _APPLY_CHEB_FN if _APPLY_CHEB_FN is not None else apply_poly_right_cheb

    if dev.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        if basis == "mono":
            _ = apply_mono(Z, S, coeffs)
        else:
            _ = apply_cheb(Z, S, coeffs, ell=ell)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return 1e6 * (t1 - t0) / reps  # microseconds per apply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--ell", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument(
        "--beta-mode", type=str, default="fro", choices=["fro", "trace", "maxdiag"]
    )
    ap.add_argument("--ridge", type=float, default=0.0)
    ap.add_argument("--jacobi-eps", type=float, default=0.0)

    ap.add_argument("--deg-list", type=str, default="2,3,4,5")
    ap.add_argument("--iter-list", type=str, default="3,4,5")

    ap.add_argument(
        "--coeff-dir",
        type=str,
        required=True,
        help="Directory with coeff json files named like basis_deg.json, e.g. mono_3.json",
    )
    ap.add_argument("--out-csv", type=str, default="bench_out.csv")
    ap.add_argument("--do-micro", action="store_true")
    ap.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile to optimize the evaluation kernels",
    )
    args = ap.parse_args()

    if torch is None:
        raise RuntimeError("This script needs torch installed.")

    global _APPLY_MONO_FN, _APPLY_CHEB_FN
    if args.compile:
        print("Compiling functions with torch.compile()...")
        _APPLY_MONO_FN = torch.compile(apply_poly_right_mono)
        _APPLY_CHEB_FN = torch.compile(apply_poly_right_cheb)
    else:
        _APPLY_MONO_FN = None
        _APPLY_CHEB_FN = None

    Bs_np = load_mats(args.npz)
    Bs = [torch.tensor(B, dtype=torch.float32, device=args.device) for B in Bs_np]

    degs = [int(x) for x in args.deg_list.split(",")]
    iters_list = [int(x) for x in args.iter_list.split(",")]

    # --- Warmup ---
    if torch.device(args.device).type == "cuda":
        print("Warming up GPU...")
        # Use first basis/deg/coeffs to warm up
        w_basis = "mono"
        w_deg = degs[0]
        w_coeff_path = os.path.join(args.coeff_dir, f"{w_basis}_{w_deg}.json")
        with open(w_coeff_path, "r", encoding="utf-8") as f:
            w_dat = json.load(f)
        w_coeffs = torch.tensor(
            w_dat["coeffs"], dtype=torch.float32, device=args.device
        )

        # Run a few dummy iterations
        for _ in range(5):
            _ = run_precond(
                B=Bs[0],
                basis=w_basis,
                deg=w_deg,
                iters=5,
                ell=args.ell,
                coeffs=w_coeffs,
                ridge=args.ridge,
                jacobi_eps=args.jacobi_eps,
                beta_mode=args.beta_mode,
            )
        torch.cuda.synchronize()
        print("Warmup complete.")

    rows = []
    for basis in ["mono", "cheb"]:
        for deg in degs:
            coeff_path = os.path.join(args.coeff_dir, f"{basis}_{deg}.json")
            with open(coeff_path, "r", encoding="utf-8") as f:
                dat = json.load(f)
            coeffs = torch.tensor(
                dat["coeffs"], dtype=torch.float32, device=args.device
            )

            if args.do_micro:
                n = Bs[0].shape[0]
                us = microbench_apply(n, basis, deg, args.ell, coeffs, args.device)
                rows.append(
                    {
                        "kind": "micro",
                        "basis": basis,
                        "deg": deg,
                        "iters": 0,
                        "matrix_id": -1,
                        "total_ms": 0.0,
                        "apply_us": us,
                        "deltaF_final": 0.0,
                        "deltaF_traj": "",
                    }
                )

            for iters in iters_list:
                for mid, B in enumerate(Bs):
                    out = run_precond(
                        B=B,
                        basis=basis,
                        deg=deg,
                        iters=iters,
                        ell=args.ell,
                        coeffs=coeffs,
                        ridge=args.ridge,
                        jacobi_eps=args.jacobi_eps,
                        beta_mode=args.beta_mode,
                    )
                    rows.append(
                        {
                            "kind": "full",
                            "basis": out.basis,
                            "deg": out.deg,
                            "iters": out.iters,
                            "matrix_id": mid,
                            "total_ms": out.total_ms,
                            "apply_us": 0.0,
                            "deltaF_final": out.deltaF_final,
                            "deltaF_traj": ";".join(
                                [f"{x:.6g}" for x in out.deltaF_traj]
                            ),
                        }
                    )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
