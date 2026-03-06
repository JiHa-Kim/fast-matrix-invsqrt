#!/usr/bin/env python3
"""
Phase 2 Local Polynomial Designer (Chebyshev, r=2.0)

Implements the mathematically rigorous 2-step local refinement protocol:
- Uses a hybrid grid (continuous + all bf16 representables) to prevent Runge's Phenomenon.
- Solves the constrained Minimax problem |1 - x * q(x)^2| via Linear Programming.
- Supports exactly evaluating and verifying the Chebyshev coefficients in pure bf16 logic.
"""

import argparse
import json
import numpy as np
from scipy.optimize import linprog

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from design_bf16_poly import (
    bf16_round_f32,
    all_bf16_values_in_interval,
    x_to_t,
    cheb_vander_t,
    cheb_eval_bf16,
)


def solve_local_lp(
    a_dom: float, b_dom: float, deg: int, x_proxy: np.ndarray
) -> tuple[np.ndarray, float]:
    """Linearized LP: |1 - sqrt(x)*q(x)| <= delta"""
    x = x_proxy.astype(np.float64)
    s = np.sqrt(x).reshape(-1, 1)

    t = x_to_t(x, a_dom, b_dom).astype(np.float64)
    V = cheb_vander_t(t, deg)
    G = s * V

    # Objective: Minimize max error delta
    c_obj = np.zeros(deg + 2, dtype=np.float64)
    c_obj[-1] = 1.0

    # Constraints
    A1 = np.hstack([G, -np.ones((G.shape[0], 1))])
    A2 = np.hstack([-G, -np.ones((G.shape[0], 1))])
    A_ub = np.vstack([A1, A2])

    b1 = np.ones(G.shape[0])
    b2 = -np.ones(G.shape[0])
    b_ub = np.concatenate([b1, b2])

    res = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * (deg + 1) + [(0.0, None)],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    return res.x[: deg + 1].astype(np.float64), float(res.x[-1])


def verify_bf16(a_dom: float, b_dom: float, coeffs: np.ndarray) -> float:
    """Evaluate certificate max purely in bf16 logic: S_new = bf16(x * bf16(q * q))"""
    xs = all_bf16_values_in_interval(a_dom, b_dom)
    q = cheb_eval_bf16(xs, a_dom, b_dom, coeffs.astype(np.float32))

    q_pow2 = bf16_round_f32(q * q)
    S_new = bf16_round_f32(xs.astype(np.float32) * q_pow2)

    return float(np.max(np.abs(1.0 - S_new)))


def design_local(rho: float, deg: int) -> dict:
    a_dom = max(1e-6, 1.0 - rho)
    b_dom = 1.0 + rho

    # 1. Hybrid Grid (Phase 2 constraint)
    x_cont = np.linspace(a_dom, b_dom, 3000, dtype=np.float64)
    x_bf16 = all_bf16_values_in_interval(a_dom, b_dom).astype(np.float64)
    x_proxy = np.unique(np.concatenate([x_cont, x_bf16]))

    # 2. Optimal Minimax via LP
    coeffs, lp_delta = solve_local_lp(a_dom, b_dom, deg, x_proxy)

    # 3. Hardware verification
    bf16_err = verify_bf16(a_dom, b_dom, coeffs)

    return {
        "rho": float(rho),
        "a_dom": float(a_dom),
        "b_dom": float(b_dom),
        "deg": int(deg),
        "theoretical_proxy_delta": float(lp_delta),
        "bf16_max_cert_err": float(bf16_err),
        "coeffs": coeffs.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 2 Local Chebyshev Minimax Designer")
    ap.add_argument(
        "--rho", type=float, required=True, help="Radius of local interval around 1.0"
    )
    ap.add_argument("--deg", type=int, default=3, help="Polynomial Degree")
    ap.add_argument("--out", type=str, required=True, help="JSON output file")
    args = ap.parse_args()

    out = design_local(rho=args.rho, deg=args.deg)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"Phase 2 Local Polynomial generated -> {args.out}")
    print(f"Interval: [{out['a_dom']:.4f}, {out['b_dom']:.4f}]")
    print(f"LP proxy delta: {out['theoretical_proxy_delta']:.6g}")
    print(f"BF16 maximum cert error |1 - S_new|: {out['bf16_max_cert_err']:.6g}")


if __name__ == "__main__":
    main()
