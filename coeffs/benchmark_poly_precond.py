#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np

# Ensure local module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fast_iroot.eval import (
    apply_poly_right_cheb,
    apply_poly_right_mono,
    choose_beta,
    fro_norm,
    jacobi_init,
    symmetrize,
)

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
    symmetrize_input: bool = True,
) -> RunOut:
    n = B.shape[0]
    
    # Always ensure input B is structurally symmetric in bf16
    if symmetrize_input:
        B = symmetrize(B)

    I_mat = torch.eye(n, device=B.device, dtype=torch.bfloat16)

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
    with torch.inference_mode():
        for _ in range(iters):
            # S = Z^T B Z in bf16
            tmp = B @ Z
            S = Z.T @ tmp

            # Symmetrize to prevent complex eigenvalue drift in bf16
            S = symmetrize(S)

            deltaF = float(fro_norm(S - I_mat).item())
            delta_traj.append(deltaF)

            beta = choose_beta(S, mode=beta_mode)
            Shat = S / beta

            # Apply polynomial in bf16
            if basis == "mono":
                Y = apply_mono(Z, Shat, coeffs)
            else:
                Y = apply_cheb(Z, Shat, coeffs, a_dom=ell)

            Z = (Y / torch.sqrt(beta)).to(torch.bfloat16)

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
    Z = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    S = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    S = (S + S.T) * 0.5
    S = S @ S.T  # SPD-ish
    S = S / choose_beta(S, mode="fro")

    global _APPLY_MONO_FN, _APPLY_CHEB_FN
    apply_mono = _APPLY_MONO_FN if _APPLY_MONO_FN is not None else apply_poly_right_mono       
    apply_cheb = _APPLY_CHEB_FN if _APPLY_CHEB_FN is not None else apply_poly_right_cheb       

    if dev.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
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
        _APPLY_MONO_FN = torch.compile(apply_poly_right_mono, mode="max-autotune")
        _APPLY_CHEB_FN = torch.compile(apply_poly_right_cheb, mode="max-autotune")
    else:
        _APPLY_MONO_FN = None
        _APPLY_CHEB_FN = None

    Bs_np = load_mats(args.npz)
    Bs = [torch.tensor(B, dtype=torch.bfloat16, device=args.device) for B in Bs_np]

    degs = [int(x) for x in args.deg_list.split(",")]
    iters_list = [int(x) for x in args.iter_list.split(",")]

    # --- Warmup ---
    if torch.device(args.device).type == "cuda":
        print("Warming up GPU and compiling all kernels...")
        for w_basis in ["mono", "cheb"]:
            for w_deg in degs:
                w_coeff_path = os.path.join(args.coeff_dir, f"{w_basis}_{w_deg}.json")
                with open(w_coeff_path, "r", encoding="utf-8") as f:
                    w_dat = json.load(f)
                w_coeffs = torch.tensor(
                    w_dat["coeffs"], dtype=torch.bfloat16, device=args.device
                )

                # Run a few dummy iterations to trigger compilation
                for _ in range(3):
                    _ = run_precond(
                        B=Bs[0],
                        basis=w_basis,
                        deg=w_deg,
                        iters=3,
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
                dat["coeffs"], dtype=torch.bfloat16, device=args.device
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
