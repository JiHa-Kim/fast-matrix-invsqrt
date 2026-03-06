#!/usr/bin/env python3
"""
Design bf16-safe one-sided polynomials on x in [ell, 1].

Goal: maximize m = min_x g(x) where g(x) = sqrt(x) q(x)
Subject to (proxy constraints): m <= g(x) <= 1 - mu
Then verify in bf16 arithmetic on ALL bf16 representables in [ell,1]:
  g_bf16(x) <= 1.0  (no overshoot)

Supports two bases:
  - basis=mono: q(x) = sum_k a_k x^k      (bf16 Horner eval)
  - basis=cheb: q(x) = sum_k c_k T_k(t)  (bf16 Chebyshev recurrence on t(x))
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import linprog
from numpy.polynomial import Chebyshev, Polynomial


def bf16_round_f32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    u = x.view(np.uint32)
    is_special = (u & 0x7F800000) == 0x7F800000
    lsb = (u >> 16) & 1
    bias = np.uint32(0x7FFF) + lsb.astype(np.uint32)
    ur = (u + bias) & np.uint32(0xFFFF0000)
    out = ur.view(np.float32)
    out[is_special] = x[is_special]
    return out


def all_bf16_values_in_interval(a: float, b: float) -> np.ndarray:
    bits16 = np.arange(0, 1 << 16, dtype=np.uint16)
    bits32 = bits16.astype(np.uint32) << 16
    vals = bits32.view(np.float32)

    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= np.float32(a)) & (vals <= np.float32(b))]
    vals = np.unique(vals.astype(np.float32))
    return vals


def x_to_t(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return (2.0 * x - (a + b)) / (b - a)


def cheb_vander_t(t: np.ndarray, deg: int) -> np.ndarray:
    return np.polynomial.chebyshev.chebvander(t, deg).astype(np.float64)


def cheb_eval_bf16(x: np.ndarray, a: float, b: float, c_cheb: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    c_cheb = c_cheb.astype(np.float32, copy=False)
    d = c_cheb.size - 1

    t = bf16_round_f32(x_to_t(x, a, b).astype(np.float32))
    T0 = bf16_round_f32(np.ones_like(t, dtype=np.float32))
    if d == 0:
        return bf16_round_f32(c_cheb[0] * T0)

    T1 = t
    q = bf16_round_f32(c_cheb[0] * T0 + c_cheb[1] * T1)
    for k in range(2, d + 1):
        Tk = bf16_round_f32(2.0 * t * T1 - T0)
        q = bf16_round_f32(q + c_cheb[k] * Tk)
        T0, T1 = T1, Tk
    return q


def monomial_eval_bf16_horner(x: np.ndarray, a_mono: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    a_mono = a_mono.astype(np.float32, copy=False)
    d = a_mono.size - 1

    q = bf16_round_f32(np.full_like(x, a_mono[d], dtype=np.float32))
    for k in range(d - 1, -1, -1):
        q = bf16_round_f32(bf16_round_f32(q * x) + a_mono[k])
    return q


@dataclass
class DesignResult:
    coeffs: np.ndarray
    g_min_proxy: float
    mu: float


def build_proxy_set(ell: float, n_log: int, n_lin: int) -> np.ndarray:
    a, b = float(ell), 1.0
    xs = []
    if n_log > 0:
        xs.append(np.geomspace(a, b, n_log, dtype=np.float64))
    if n_lin > 0:
        xs.append(np.linspace(a, b, n_lin, dtype=np.float64))
    return np.unique(np.concatenate(xs))


def solve_one_sided_lp(
    ell: float,
    deg: int,
    mu: float,
    x_proxy: np.ndarray,
    coef_bound: float,
) -> DesignResult:
    a_dom, b_dom = float(ell), 1.0
    x = x_proxy.astype(np.float64)
    s = np.sqrt(x).reshape(-1, 1)

    # ALWAYS optimize using the mathematically well-conditioned Chebyshev basis
    t = x_to_t(x, a_dom, b_dom).astype(np.float64)
    V = cheb_vander_t(t, deg)

    # g(x) = sqrt(x) * q(x) = (s * V) @ coeffs
    G = s * V

    # variables z = [coeffs(0..deg), m]
    nvar = (deg + 1) + 1
    c_obj = np.zeros(nvar, dtype=np.float64)
    c_obj[-1] = -1.0  # maximize m

    # g_i >= m  <=>  -g_i + m <= 0
    A1 = np.hstack([-G, np.ones((G.shape[0], 1), dtype=np.float64)])
    b1 = np.zeros(G.shape[0], dtype=np.float64)

    # g_i <= 1 - mu
    A2 = np.hstack([G, np.zeros((G.shape[0], 1), dtype=np.float64)])
    b2 = (1.0 - mu) * np.ones(G.shape[0], dtype=np.float64)

    A_ub = np.vstack([A1, A2])
    b_ub = np.concatenate([b1, b2])

    bounds = [(-coef_bound, coef_bound)] * (deg + 1) + [(0.0, 1.0)]
    res = linprog(c=c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    coeffs = res.x[: deg + 1].astype(np.float64)
    m = float(res.x[-1])
    return DesignResult(coeffs=coeffs, g_min_proxy=m, mu=mu)


def verify_bf16_no_overshoot(
    ell: float, coeffs: np.ndarray, basis: str
) -> Tuple[bool, float, float]:
    a_dom, b_dom = float(ell), 1.0
    xs = all_bf16_values_in_interval(a_dom, b_dom)
    xs = xs[xs > 0]

    if basis == "cheb":
        q = cheb_eval_bf16(xs, a_dom, b_dom, coeffs.astype(np.float32))
    else:
        q = monomial_eval_bf16_horner(xs, coeffs.astype(np.float32))

    g = bf16_round_f32(np.sqrt(xs.astype(np.float32)) * q)
    max_g = float(np.max(g))
    min_g = float(np.min(g))
    ok = max_g <= 1.0
    return ok, max_g, min_g


def refine_bf16_coeffs(ell: float, initial_coeffs: np.ndarray, basis: str) -> Tuple[np.ndarray, float, float]:
    def eval_g(c: np.ndarray) -> Tuple[float, float]:
        _, max_g, min_g = verify_bf16_no_overshoot(ell, c, basis)
        return max_g, min_g

    def get_alpha_and_min(c: np.ndarray) -> Tuple[float, float, float]:
        alpha_lo, alpha_hi = 0.0, 1.0
        for _ in range(20):
            max_g, _ = eval_g(c * alpha_hi)
            if max_g <= 1.0:
                alpha_lo = alpha_hi
                alpha_hi *= 2.0
            else:
                break
        
        best_alpha = alpha_lo
        for _ in range(40):
            mid = 0.5 * (alpha_lo + alpha_hi)
            max_g, _ = eval_g(c * mid)
            if max_g <= 1.0:
                best_alpha = mid
                alpha_lo = mid
            else:
                alpha_hi = mid
                
        max_g, min_g = eval_g(c * best_alpha)
        return best_alpha, max_g, min_g

    current_c = initial_coeffs.copy()
    alpha, _, current_m = get_alpha_and_min(current_c)
    current_c = current_c * alpha

    delta = 0.05 * np.max(np.abs(current_c))
    min_delta = 1e-8 * np.max(np.abs(current_c))
    if min_delta == 0:
        min_delta = 1e-8

    iters = 0
    while delta > min_delta and iters < 500:
        improved = False
        for i in range(len(current_c)):
            for sign in [1.0, -1.0]:
                c_prop = current_c.copy()
                c_prop[i] += sign * delta
                
                alpha_prop, max_g_prop, m_prop = get_alpha_and_min(c_prop)
                if m_prop > current_m:
                    current_c = c_prop * alpha_prop
                    current_m = m_prop
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            delta *= 0.5
        iters += 1

    max_g, min_g = eval_g(current_c)
    return current_c, max_g, min_g


def design(
    ell: float,
    deg: int,
    basis: str,
    mu_hi: float,
    mu_iters: int,
    proxy_log: int,
    proxy_lin: int,
    coef_bound: float,
    include_bf16_in_proxy: bool,
    refine_bf16: bool = False,
) -> dict:
    x_proxy = build_proxy_set(ell, proxy_log, proxy_lin)
    if include_bf16_in_proxy:
        x_bf16 = all_bf16_values_in_interval(ell, 1.0).astype(np.float64)
        x_proxy = np.unique(np.concatenate([x_proxy, x_bf16]))

    lo, hi = 0.0, float(mu_hi)
    best = None

    for _ in range(mu_iters):
        mu = 0.5 * (lo + hi)
        try:
            sol = solve_one_sided_lp(ell, deg, mu, x_proxy, coef_bound)
        except RuntimeError:
            hi = mu
            continue

        c_eval = sol.coeffs
        if basis == "mono":
            poly = Chebyshev(c_eval, domain=[ell, 1.0]).convert(kind=Polynomial)
            c_eval = poly.coef
            # zero pad if conversion dropped highest order zero term
            if len(c_eval) < deg + 1:
                c_eval = np.pad(c_eval, (0, deg + 1 - len(c_eval)))

        ok, max_g, min_g = verify_bf16_no_overshoot(ell, c_eval, basis)
        if ok:
            best = (sol, c_eval, max_g, min_g)
            hi = mu
        else:
            lo = mu

    if best is None:
        raise RuntimeError("No feasible mu found. Increase mu_hi or coef_bound.")

    sol, c_eval, max_g, min_g = best
    
    kind = "phase1_bf16_safe"
    if refine_bf16:
        c_eval, max_g, min_g = refine_bf16_coeffs(ell, c_eval, basis)
        kind = "phase1_bf16_safe_refined"

    return {
        "kind": kind,
        "basis": basis,
        "ell": ell,
        "deg": int(deg),
        "mu_star": float(sol.mu),
        "proxy_min_g": float(sol.g_min_proxy),
        "bf16_eval_max_g": float(max_g),
        "bf16_eval_min_g": float(min_g),
        "coeffs": c_eval.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ell", type=float, required=True)
    ap.add_argument("--deg", type=int, required=True)
    ap.add_argument("--basis", type=str, choices=["mono", "cheb"], required=True)

    ap.add_argument("--mu-hi", type=float, default=0.05)
    ap.add_argument("--mu-iters", type=int, default=24)

    ap.add_argument("--proxy-log", type=int, default=4000)
    ap.add_argument("--proxy-lin", type=int, default=4000)
    ap.add_argument("--include-bf16-in-proxy", action="store_true")
    ap.add_argument("--refine-bf16", action="store_true", help="Refine coefficients using exact bf16-in-the-loop pattern search")

    ap.add_argument("--coef-bound", type=float, default=1e4)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    out = design(
        ell=args.ell,
        deg=args.deg,
        basis=args.basis,
        mu_hi=args.mu_hi,
        mu_iters=args.mu_iters,
        proxy_log=args.proxy_log,
        proxy_lin=args.proxy_lin,
        coef_bound=args.coef_bound,
        include_bf16_in_proxy=args.include_bf16_in_proxy,
        refine_bf16=args.refine_bf16,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"wrote {args.out}")
    print(f"mu_star = {out['mu_star']:.8g}, proxy_min_g = {out['proxy_min_g']:.8g}")
    print(
        f"bf16 max_g = {out['bf16_eval_max_g']:.8g}, bf16 min_g = {out['bf16_eval_min_g']:.8g}"
    )


if __name__ == "__main__":
    main()
