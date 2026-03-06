#!/usr/bin/env python3
"""
Design bf16-optimal local minimax polynomials for Phase 2 preconditioning.

Objective: Minimize the maximum residual |1 - x * q(x)^r| over x in [1-rho, 1+rho].
"""

import argparse
import json
import numpy as np
from scipy.optimize import linprog
from numpy.polynomial import Chebyshev, Polynomial

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from design_bf16_poly import (
    bf16_round_f32,
    all_bf16_values_in_interval,
    x_to_t,
    cheb_vander_t,
    cheb_eval_bf16,
    monomial_eval_bf16_horner,
    build_proxy_set
)

def solve_local_lp(
    a_dom: float,
    b_dom: float,
    deg: int,
    x_proxy: np.ndarray,
    coef_bound: float,
    r: float,
) -> Tuple[np.ndarray, float]:
    x = x_proxy.astype(np.float64)
    s = (x ** (1.0 / r)).reshape(-1, 1)

    t = x_to_t(x, a_dom, b_dom).astype(np.float64)
    V = cheb_vander_t(t, deg)

    # We are linearizing |1 - x q^r| to |1 - x^{1/r} q(x)|
    G = s * V

    nvar = deg + 2
    c_obj = np.zeros(nvar, dtype=np.float64)
    c_obj[-1] = 1.0  # minimize delta

    # G @ c - delta <= 1
    # -G @ c - delta <= -1
    A1 = np.hstack([G, -np.ones((G.shape[0], 1))])
    b1 = np.ones(G.shape[0])

    A2 = np.hstack([-G, -np.ones((G.shape[0], 1))])
    b2 = -np.ones(G.shape[0])

    A_ub = np.vstack([A1, A2])
    b_ub = np.concatenate([b1, b2])

    bounds = [(-coef_bound, coef_bound)] * (deg + 1) + [(0.0, None)]
    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    coeffs = res.x[: deg + 1].astype(np.float64)
    delta = float(res.x[-1])
    return coeffs, delta

def verify_bf16_local(
    a_dom: float, b_dom: float, coeffs: np.ndarray, basis: str, r: float
) -> float:
    xs = all_bf16_values_in_interval(a_dom, b_dom)
    
    if basis == "cheb":
        q = cheb_eval_bf16(xs, a_dom, b_dom, coeffs.astype(np.float32))
    else:
        q = monomial_eval_bf16_horner(xs, coeffs.astype(np.float32))
        
    q_f32 = q.astype(np.float32)
    x_f32 = xs.astype(np.float32)
    
    # Compute certificate x * q^r in bf16
    q_pow = q_f32
    for _ in range(int(r) - 1):
        q_pow = bf16_round_f32(q_pow * q_f32)
    
    S_new = bf16_round_f32(x_f32 * q_pow)
    err = np.max(np.abs(1.0 - S_new))
    return float(err)

def refine_bf16_local(
    a_dom: float, b_dom: float, initial_coeffs: np.ndarray, basis: str, r: float
) -> Tuple[np.ndarray, float]:
    current_c = initial_coeffs.copy()
    current_err = verify_bf16_local(a_dom, b_dom, current_c, basis, r)
    
    delta = 0.05 * np.max(np.abs(current_c))
    min_delta = 1e-8
    
    iters = 0
    while delta > min_delta and iters < 500:
        improved = False
        for i in range(len(current_c)):
            for sign in [1.0, -1.0]:
                c_prop = current_c.copy()
                c_prop[i] += sign * delta
                
                err_prop = verify_bf16_local(a_dom, b_dom, c_prop, basis, r)
                if err_prop < current_err:
                    current_c = c_prop
                    current_err = err_prop
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            delta *= 0.5
        iters += 1
        
    return current_c, current_err

def design_local(
    rho: float,
    deg: int,
    basis: str,
    r: float,
    proxy_log: int,
    proxy_lin: int,
    coef_bound: float,
    refine_bf16: bool
) -> dict:
    a_dom = max(1e-6, 1.0 - rho)
    b_dom = 1.0 + rho
    
    x_proxy = build_proxy_set(a_dom, proxy_log, proxy_lin)
    # Add actual bf16 values to proxy for better LP
    x_bf16 = all_bf16_values_in_interval(a_dom, b_dom).astype(np.float64)
    x_proxy = np.unique(np.concatenate([x_proxy, x_bf16]))
    
    coeffs, lp_delta = solve_local_lp(a_dom, b_dom, deg, x_proxy, coef_bound, r)
    
    if basis == "mono":
        poly = Chebyshev(coeffs, domain=[a_dom, b_dom]).convert(kind=Polynomial)
        c_eval = poly.coef
        if len(c_eval) < deg + 1:
            c_eval = np.pad(c_eval, (0, deg + 1 - len(c_eval)))
    else:
        c_eval = coeffs
        
    bf16_err_base = verify_bf16_local(a_dom, b_dom, c_eval, basis, r)
    
    if refine_bf16:
        c_eval, bf16_err_refined = refine_bf16_local(a_dom, b_dom, c_eval, basis, r)
        final_err = bf16_err_refined
        kind = "phase2_local_refined"
    else:
        final_err = bf16_err_base
        kind = "phase2_local"
        
    return {
        "kind": kind,
        "basis": basis,
        "rho": float(rho),
        "a_dom": float(a_dom),
        "b_dom": float(b_dom),
        "deg": int(deg),
        "r": float(r),
        "proxy_linear_delta": float(lp_delta),
        "bf16_max_cert_err": float(final_err),
        "coeffs": c_eval.tolist()
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, required=True, help="Radius of local interval around 1.0")
    ap.add_argument("--deg", type=int, required=True)
    ap.add_argument("--basis", type=str, choices=["mono", "cheb"], required=True)
    ap.add_argument("--r", type=float, default=2.0)
    
    ap.add_argument("--proxy-log", type=int, default=1000)
    ap.add_argument("--proxy-lin", type=int, default=2000)
    ap.add_argument("--coef-bound", type=float, default=1e6)
    ap.add_argument("--refine-bf16", action="store_true")
    ap.add_argument("--out", type=str, required=True)
    
    args = ap.parse_args()
    
    out = design_local(
        rho=args.rho,
        deg=args.deg,
        basis=args.basis,
        r=args.r,
        proxy_log=args.proxy_log,
        proxy_lin=args.proxy_lin,
        coef_bound=args.coef_bound,
        refine_bf16=args.refine_bf16
    )
    
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        
    print(f"wrote {args.out}")
    print(f"Interval: [{out['a_dom']:.4f}, {out['b_dom']:.4f}]")
    print(f"LP proxy delta: {out['proxy_linear_delta']:.6g}")
    print(f"BF16 certificate max error |1 - S_new|: {out['bf16_max_cert_err']:.6g}")

if __name__ == "__main__":
    main()
