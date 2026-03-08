#!/usr/bin/env python3
# design_zolo_primary_and_fallback.py
#
# Offline design for:
#   - Primary: 1-step scaled Zolotarev map, type (2r+1,2r) (one solve)
#   - Fallback: 2-step scaled Zolotarev map, types (2r1+1,2r1) then (2r2+1,2r2) (two solves)
#
# Output: coefficients for q(t) = alpha * P(t) / Q(t), deg(P)=deg(Q)=r.
# The polar / Gram-side update is Z_+ = Z q(S), with S = Z^T B Z.
#
# For a target certificate condition kappa_star, we target singular values in:
#   [1/sqrt(kappa_star), sqrt(kappa_star)].
# Under scaling sigma_max=1, this means sigma_min >= 1/sqrt(kappa_star).
#
# Primary design:
#   Find ell such that ell_plus = Zhat_{2r+1}(ell; ell) == 1/sqrt(kappa_star).
#
# Fallback 2-step design:
#   Find ell0 such that ell2 == 1/sqrt(kappa_star), where
#     ell1 = Zhat_{2r1+1}(ell0; ell0)
#     ell2 = Zhat_{2r2+1}(ell1; ell1)
#
# Notes for bf16 pipelines:
# - Build S, P(S), Q(S) in fp32 (bf16 inputs, fp32 accumulate), symmetrize.
# - Cholesky factorize Q(S) once per step, solve many RHS.
# - Keep r small for stability (primary r=3; fallback r=2 is very safe).

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Tuple

import mpmath as mp


def _zolo_c_list(ell: mp.mpf, r: int) -> List[mp.mpf]:
    # c_i = ell^2 * sn^2(iK/(2r+1)) / cn^2(iK/(2r+1)),
    # with elliptic parameter m = 1 - ell^2.
    m = 1 - ell**2
    K = mp.ellipk(m)
    cs: List[mp.mpf] = []
    for i in range(1, 2 * r + 1):
        u = i * K / (2 * r + 1)
        sn = mp.ellipfun("sn", u, m)
        cn = mp.ellipfun("cn", u, m)
        cs.append(ell**2 * (sn**2) / (cn**2))
    return cs


def _zolo_scaled(x: mp.mpf, ell: mp.mpf, r: int) -> mp.mpf:
    # Zhat(x; ell) normalized so that Zhat(1; ell) = 1.
    cs = _zolo_c_list(ell, r)

    num = x
    den = mp.mpf(1)

    for j in range(1, r + 1):
        c_odd = cs[2 * j - 2]  # c_{2j-1}
        c_even = cs[2 * j - 1]  # c_{2j}
        num *= (x**2 + c_even) / (x**2 + c_odd)
        den *= (1 + c_even) / (1 + c_odd)

    return num / den


def _F_endpoint(ell: mp.mpf, r: int) -> mp.mpf:
    # ell -> ell_plus under one Zolotarev step of degree r
    return _zolo_scaled(ell, ell, r)


def _poly_mul_linear_desc(coeffs: List[mp.mpf], c: mp.mpf) -> List[mp.mpf]:
    # coeffs descending: sum_{k=0}^d a[k] t^{d-k}
    d = len(coeffs) - 1
    out = [mp.mpf(0)] * (d + 2)
    out[0] = coeffs[0]
    for i in range(1, d + 1):
        out[i] = coeffs[i] + c * coeffs[i - 1]
    out[d + 1] = c * coeffs[d]
    return out


def _build_q_polys(ell: mp.mpf, r: int) -> Dict[str, object]:
    cs = _zolo_c_list(ell, r)
    c_odd = [cs[2 * j] for j in range(r)]  # c1,c3,...,c_{2r-1}
    c_even = [cs[2 * j + 1] for j in range(r)]  # c2,c4,...,c_{2r}

    # q(t) = alpha * prod (t + c_even[j]) / prod (t + c_odd[j])
    # choose alpha so that q(1) = 1
    alpha = mp.mpf(1)
    for j in range(r):
        alpha *= (1 + c_odd[j]) / (1 + c_even[j])

    P = [alpha]
    for ce in c_even:
        P = _poly_mul_linear_desc(P, ce)

    Q = [mp.mpf(1)]
    for co in c_odd:
        Q = _poly_mul_linear_desc(Q, co)

    return {
        "ell": ell,
        "r": r,
        "alpha": alpha,
        "c_odd": c_odd,
        "c_even": c_even,
        "P_desc": P,
        "Q_desc": Q,
    }


def _solve_ell_one_step(kappa_star: float, r: int, iters: int = 240) -> mp.mpf:
    target = mp.mpf(1) / mp.sqrt(mp.mpf(kappa_star))

    # Bisection in log space for ell in (0, target].
    lo = mp.mpf("1e-40")
    hi = target
    for _ in range(iters):
        mid = mp.sqrt(lo * hi)
        val = _F_endpoint(mid, r)
        if val < target:
            lo = mid
        else:
            hi = mid
    return hi


def _solve_ell_two_step(
    kappa_star: float, r1: int, r2: int, iters: int = 260
) -> Tuple[mp.mpf, mp.mpf, mp.mpf]:
    target = mp.mpf(1) / mp.sqrt(mp.mpf(kappa_star))

    # We search ell0 on a safe numeric range; for kappa_star=1.5 and r1=r2=2,
    # ell0 lands around 1e-14 (well within this bracket).
    lo = mp.mpf("1e-30")
    hi = target

    for _ in range(iters):
        mid = mp.sqrt(lo * hi)
        ell1 = _F_endpoint(mid, r1)
        ell2 = _F_endpoint(ell1, r2)
        if ell2 < target:
            lo = mid
        else:
            hi = mid

    ell0 = hi
    ell1 = _F_endpoint(ell0, r1)
    ell2 = _F_endpoint(ell1, r2)
    return ell0, ell1, ell2


def _mp_list_to_float(xs: List[mp.mpf]) -> List[float]:
    return [float(x) for x in xs]


def _pack_step(label: str, info: Dict[str, object]) -> Dict[str, object]:
    ell = info["ell"]
    r = int(info["r"])
    alpha = info["alpha"]
    return {
        "label": label,
        "r": r,
        "ell": float(ell),
        "alpha": float(alpha),
        "c_odd": _mp_list_to_float(info["c_odd"]),
        "c_even": _mp_list_to_float(info["c_even"]),
        "P_desc": _mp_list_to_float(info["P_desc"]),
        "Q_desc": _mp_list_to_float(info["Q_desc"]),
    }


def main() -> None:
    mp.mp.dps = 120

    p = argparse.ArgumentParser(
        description="Design primary 1-step and fallback 2-step Zolotarev coefficients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kappa-star", type=float, default=1.5)
    p.add_argument(
        "--r-primary",
        type=int,
        default=3,
        help="Primary one-step degree r (type (2r+1,2r)).",
    )
    p.add_argument(
        "--r-fallback-1", type=int, default=2, help="Fallback step1 degree r1."
    )
    p.add_argument(
        "--r-fallback-2", type=int, default=2, help="Fallback step2 degree r2."
    )
    p.add_argument("--out", type=str, default="zolo_policy.json")

    args = p.parse_args()

    kappa_star = float(args.kappa_star)
    if not (math.isfinite(kappa_star) and kappa_star > 1.0):
        raise ValueError("kappa_star must be > 1")

    r_primary = int(args.r_primary)
    r1 = int(args.r_fallback_1)
    r2 = int(args.r_fallback_2)

    # Primary: 1-step
    ell_primary = _solve_ell_one_step(kappa_star, r_primary)
    step_primary = _build_q_polys(ell_primary, r_primary)
    ell_plus_primary = _F_endpoint(ell_primary, r_primary)

    # Fallback: 2-step (minimax optimal, re-parameterized by ell after step1)
    ell0_fb, ell1_fb, ell2_fb = _solve_ell_two_step(kappa_star, r1, r2)
    step_fb1 = _build_q_polys(ell0_fb, r1)
    step_fb2 = _build_q_polys(ell1_fb, r2)

    def kappaG(ell: float) -> float:
        return 1.0 / ell

    def kappaS(ell: float) -> float:
        return 1.0 / (ell * ell)

    result = {
        "kappa_star": kappa_star,
        "target_sigma_min": float(1.0 / math.sqrt(kappa_star)),
        "primary_1step": {
            "coverage_kappaG_max_model": kappaG(float(ell_primary)),
            "coverage_kappaS_max_model": kappaS(float(ell_primary)),
            "ell_in": float(ell_primary),
            "ell_out": float(ell_plus_primary),
            "step": _pack_step("primary", step_primary),
        },
        "fallback_2step": {
            "coverage_kappaG_max_model": kappaG(float(ell0_fb)),
            "coverage_kappaS_max_model": kappaS(float(ell0_fb)),
            "ell0_in": float(ell0_fb),
            "ell1_mid": float(ell1_fb),
            "ell2_out": float(ell2_fb),
            "step1": _pack_step("fallback_step1", step_fb1),
            "step2": _pack_step("fallback_step2", step_fb2),
        },
        "implementation_notes": {
            "step_form": "q(t)=alpha*P(t)/Q(t), deg P=deg Q=r. Apply as Z <- Z*P(S) then right-solve with Q(S).",
            "bf16_safety": "Build S,P(S),Q(S) in fp32 (bf16 inputs, fp32 accumulate), symmetrize, ridge Q if needed, then fp32 Cholesky(Q).",
            "divergence_triggers": "Cholesky fail on Q, NaN/Inf, or spread proxy increases. If so, run fallback 2-step.",
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {args.out}")
    print(
        "Primary kappaG_max_model:",
        result["primary_1step"]["coverage_kappaG_max_model"],
    )
    print(
        "Fallback kappaG_max_model:",
        result["fallback_2step"]["coverage_kappaG_max_model"],
    )


if __name__ == "__main__":
    main()
