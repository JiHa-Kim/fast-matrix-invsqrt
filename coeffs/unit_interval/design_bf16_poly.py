#!/usr/bin/env python3
"""
Design bf16-screened one-sided Chebyshev polynomials on x in [ell, 1].

We design q(x) in Chebyshev basis on t(x) = (2x - (ell + 1)) / (1 - ell):
    q(x) = sum_{k=0}^deg c_k T_k(t(x))

Proxy design goal on a sampled set x_proxy:
    maximize m
    subject to  m <= g(x) <= 1 - mu
    where g(x) = x^(1/r) q(x)

Then verify with a bf16-like Clenshaw evaluator on all bf16 representables in [ell, 1]:
    g_bf16(x) <= 1

Important:
- This is only a scalar discrete-screening procedure.
- It is not a rigorous matrix-level safety certificate.
- The LP constraints are only enforced on x_proxy, not the full interval.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import itertools

import numpy as np
from scipy.optimize import linprog


def bf16_round_f32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    is_special = (u & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    ur = (u + bias) & np.uint32(0xFFFF0000)
    out = ur.view(np.float32)
    out[is_special] = x[is_special]
    return out


def bf16_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return bf16_round_f32(bf16_round_f32(a) + bf16_round_f32(b))


def bf16_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return bf16_round_f32(bf16_round_f32(a) - bf16_round_f32(b))


def bf16_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return bf16_round_f32(bf16_round_f32(a) * bf16_round_f32(b))


def all_bf16_values_in_interval(a: float, b: float) -> np.ndarray:
    bits16 = np.arange(0, 1 << 16, dtype=np.uint16)
    bits32 = bits16.astype(np.uint32) << np.uint32(16)
    vals = bits32.view(np.float32)
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= np.float32(a)) & (vals <= np.float32(b))]
    vals = np.unique(vals.astype(np.float32))
    return vals


def x_to_t(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return (2.0 * x - (a + b)) / (b - a)


def cheb_vander_t(t: np.ndarray, deg: int) -> np.ndarray:
    return np.polynomial.chebyshev.chebvander(t, deg).astype(np.float64)


def cheb_eval_bf16_clenshaw(
    x: np.ndarray, a: float, b: float, coeffs: np.ndarray
) -> np.ndarray:
    """
    Evaluate q(x) = sum_k coeffs[k] T_k(t(x)) using bf16-like Clenshaw.

    Recurrence for standard Chebyshev T_k:
      b_{d+1} = b_{d+2} = 0
      b_k = c_k + 2 t b_{k+1} - b_{k+2},  k = d, ..., 1
      q   = c_0 + t b_1 - b_2
    """
    x = np.asarray(x, dtype=np.float32)
    coeffs = np.asarray(coeffs, dtype=np.float32)
    d = coeffs.size - 1

    t = bf16_round_f32(x_to_t(x, a, b).astype(np.float32))

    if d == 0:
        return bf16_round_f32(coeffs[0] * np.ones_like(t, dtype=np.float32))

    bkp1 = bf16_round_f32(np.zeros_like(t, dtype=np.float32))  # b_{k+1}
    bkp2 = bf16_round_f32(np.zeros_like(t, dtype=np.float32))  # b_{k+2}

    two_t = bf16_mul(np.float32(2.0) * np.ones_like(t, dtype=np.float32), t)

    for k in range(d, 0, -1):
        term = bf16_mul(two_t, bkp1)
        bk = bf16_add(
            coeffs[k] * np.ones_like(t, dtype=np.float32), bf16_sub(term, bkp2)
        )
        bkp2 = bkp1
        bkp1 = bk

    out = bf16_add(
        coeffs[0] * np.ones_like(t, dtype=np.float32), bf16_sub(bf16_mul(t, bkp1), bkp2)
    )
    return out


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
    if not xs:
        return np.array([], dtype=np.float64)
    return np.unique(np.concatenate(xs))


def solve_one_sided_lp(
    ell: float,
    deg: int,
    mu: float,
    x_proxy: np.ndarray,
    coef_bound: float,
    r: float,
) -> DesignResult:
    a_dom, b_dom = float(ell), 1.0
    x = np.asarray(x_proxy, dtype=np.float64)
    s = (x ** (1.0 / r)).reshape(-1, 1)

    t = x_to_t(x, a_dom, b_dom).astype(np.float64)
    V = cheb_vander_t(t, deg)

    # g(x) = x^(1/r) q(x) = (s * V) @ coeffs
    G = s * V

    # z = [c_0, ..., c_deg, m]
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
    ell: float,
    coeffs: np.ndarray,
    r: float,
) -> Tuple[bool, float, float]:
    a_dom, b_dom = float(ell), 1.0
    xs = all_bf16_values_in_interval(a_dom, b_dom)
    xs = xs[xs > 0]

    q = cheb_eval_bf16_clenshaw(xs, a_dom, b_dom, coeffs.astype(np.float32))
    xr = bf16_round_f32(xs.astype(np.float32))
    sr = bf16_round_f32(xr ** np.float32(1.0 / r))
    g = bf16_mul(sr, q)

    max_g = float(np.max(g))
    min_g = float(np.min(g))
    ok = max_g <= 1.0
    return ok, max_g, min_g


def refine_bf16_coeffs(
    ell: float,
    initial_coeffs: np.ndarray,
    r: float,
) -> Tuple[np.ndarray, float, float]:
    def eval_g(c: np.ndarray) -> Tuple[float, float]:
        _, max_g, min_g = verify_bf16_no_overshoot(ell, c, r)
        return max_g, min_g

    def scale_to_boundary(c: np.ndarray) -> Tuple[np.ndarray, float, float]:
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

        scaled = c * best_alpha
        max_g, min_g = eval_g(scaled)
        return scaled, max_g, min_g

    current_c, max_g, current_m = scale_to_boundary(initial_coeffs.copy())

    scale = np.max(np.abs(current_c))
    delta = 0.05 * scale
    min_delta = max(1e-8 * scale, 1e-8)

    iters = 0
    while delta > min_delta and iters < 500:
        improved = False
        for i in range(len(current_c)):
            for sign in (1.0, -1.0):
                c_prop = current_c.copy()
                c_prop[i] += sign * delta

                c_scaled, max_g_prop, m_prop = scale_to_boundary(c_prop)
                if m_prop > current_m:
                    current_c = c_scaled
                    current_m = m_prop
                    max_g = max_g_prop
                    improved = True
                    break
            if improved:
                break

        if not improved:
            delta *= 0.5
        iters += 1

    return current_c, max_g, current_m


def design(
    ell: float,
    deg: int,
    mu_hi: float,
    mu_iters: int,
    proxy_log: int,
    proxy_lin: int,
    coef_bound: float,
    include_bf16_in_proxy: bool,
    refine_bf16: bool = False,
    r: float = 2.0,
) -> dict:
    x_proxy = build_proxy_set(ell, proxy_log, proxy_lin)
    if include_bf16_in_proxy:
        x_bf16 = all_bf16_values_in_interval(ell, 1.0).astype(np.float64)
        x_proxy = np.unique(np.concatenate([x_proxy, x_bf16]))

    # We are pushing toward the boundary g <= 1, so we search for the
    # smallest safe mu, not the largest one.
    lo, hi = 0.0, float(mu_hi)
    best = None

    for _ in range(mu_iters):
        mu = 0.5 * (lo + hi)
        try:
            sol = solve_one_sided_lp(ell, deg, mu, x_proxy, coef_bound, r)
        except RuntimeError:
            hi = mu
            continue

        ok, max_g, min_g = verify_bf16_no_overshoot(ell, sol.coeffs, r)
        if ok:
            best = (sol, max_g, min_g)
            hi = mu
        else:
            lo = mu

    if best is None:
        raise RuntimeError("No feasible mu found. Increase mu_hi or coef_bound.")

    sol, max_g, min_g = best

    out = {
        "kind": "phase1_bf16_safe",
        "basis": "cheb_clenshaw",
        "ell": float(ell),
        "deg": int(deg),
        "r": float(r),
        "mu_star_pre_refine": float(sol.mu),
        "proxy_min_g_pre_refine": float(sol.g_min_proxy),
        "bf16_eval_max_g_pre_refine": float(max_g),
        "bf16_eval_min_g_pre_refine": float(min_g),
        "coeffs_pre_refine": sol.coeffs.tolist(),
    }

    coeffs_final = sol.coeffs.copy()
    max_g_final = max_g
    min_g_final = min_g

    if refine_bf16:
        coeffs_final, max_g_final, min_g_final = refine_bf16_coeffs(
            ell, coeffs_final, r
        )
        out["kind"] = "phase1_bf16_safe_refined"

    out["bf16_eval_max_g"] = float(max_g_final)
    out["bf16_eval_min_g"] = float(min_g_final)
    out["coeffs"] = coeffs_final.tolist()

    return out


def main(deg: int, r: float, results_name: str) -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--ell", type=float, default=0.001)
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "json" / f"{results_name}.json"),
    )
    ap.add_argument("--deg", type=int, default=deg)
    ap.add_argument(
        "--r", type=float, default=r, help="Root degree, e.g. 2 for inv sqrt"
    )

    ap.add_argument("--mu-hi", type=float, default=0.05)
    ap.add_argument("--mu-iters", type=int, default=24)

    ap.add_argument("--proxy-log", type=int, default=4000)
    ap.add_argument("--proxy-lin", type=int, default=4000)
    ap.add_argument("--include-bf16-in-proxy", action="store_true")
    ap.add_argument("--refine-bf16", action="store_true")

    ap.add_argument("--coef-bound", type=float, default=1e4)
    args = ap.parse_args()

    out = design(
        ell=args.ell,
        deg=args.deg,
        mu_hi=args.mu_hi,
        mu_iters=args.mu_iters,
        proxy_log=args.proxy_log,
        proxy_lin=args.proxy_lin,
        coef_bound=args.coef_bound,
        include_bf16_in_proxy=args.include_bf16_in_proxy,
        refine_bf16=args.refine_bf16,
        r=args.r,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"wrote {args.out}")
    print(
        f"pre-refine: mu_star = {out['mu_star_pre_refine']:.8g}, "
        f"proxy_min_g = {out['proxy_min_g_pre_refine']:.8g}"
    )
    print(
        f"pre-refine bf16: max_g = {out['bf16_eval_max_g_pre_refine']:.8g}, "
        f"min_g = {out['bf16_eval_min_g_pre_refine']:.8g}"
    )
    print(
        f"final bf16: max_g = {out['bf16_eval_max_g']:.8g}, "
        f"min_g = {out['bf16_eval_min_g']:.8g}"
    )


if __name__ == "__main__":
    for deg, r in itertools.product(range(2, 6), [2.0, 4.0]):
        main(deg=deg, r=r, results_name=f"root{r}/cheb_clenshaw/deg{deg}")
