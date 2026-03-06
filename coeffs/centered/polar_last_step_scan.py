#!/usr/bin/env python3
"""
Scan polynomial degrees for the first reverse-engineered local polar step.

We solve, for each degree d,

    maximize rho
    subject to sup_{x in [1-rho, 1+rho]} |x q(x)^2 - 1| <= tau

where q is a degree-d polynomial in x, represented during optimization in a
Chebyshev basis on the current interval [1-rho, 1+rho].

This is the exact-arithmetic scalar design problem from the project plan.
The script also includes an optional scalar bf16 deployment check using
Chebyshev + Clenshaw with bf16 rounding at every arithmetic operation.

Notes
-----
- The exact optimization is done on a dense grid, so this is a practical
  discrete minimax solver rather than a theorem-level Remez implementation.
- For a fixed degree, the polynomial function class is basis-independent.
  Chebyshev is used here because it is the intended deployment basis for d>=3.
- "Best" depends on objective. This script reports:
    * widest exact basin rho
    * rho / d               (very simple cost proxy)
    * atanh(rho) / d        (conditioning-style cost proxy)
  The raw winner and the cost-adjusted winner may differ.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.optimize import minimize


@dataclass
class DegreeResult:
    degree: int
    tau: float
    rho_exact: float
    exact_err_at_rho: float
    rho_over_degree: float
    atanh_rho_over_degree: float
    rho_bf16_same_rho: float | None
    bf16_err_at_exact_rho: float | None
    coeffs_cheb: list[float]


def smooth_max_abs(values: np.ndarray, p: int = 48) -> float:
    return float((np.mean(np.abs(values) ** p)) ** (1.0 / p))


def dense_error_grid(
    coeffs: np.ndarray, rho: float, ngrid: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(max(1.0 - rho, 1.0e-12), 1.0 + rho, ngrid, dtype=np.float64)
    if rho == 0.0:
        ts = np.zeros_like(xs)
    else:
        ts = (xs - 1.0) / rho
    qx = chebval(ts, coeffs)
    err = xs * qx * qx - 1.0
    return xs, ts, err


def fit_q_for_rho(
    degree: int,
    rho: float,
    ngrid: int = 2001,
    init: np.ndarray | None = None,
    smooth_p: int = 48,
    maxiter: int = 4000,
) -> tuple[float, np.ndarray]:
    """Return (discrete_inf_error, coeffs) for a fixed rho."""
    xs = np.linspace(max(1.0 - rho, 1.0e-12), 1.0 + rho, ngrid, dtype=np.float64)
    ts = np.zeros_like(xs) if rho == 0.0 else (xs - 1.0) / rho
    target = xs ** (-0.5)

    if init is None:
        coeffs0 = chebfit(ts, target, degree)
    else:
        coeffs0 = np.array(init, dtype=np.float64)
        if len(coeffs0) < degree + 1:
            coeffs0 = np.pad(coeffs0, (0, degree + 1 - len(coeffs0)))
        coeffs0 = coeffs0[: degree + 1]

    def err_vec(coeffs: np.ndarray) -> np.ndarray:
        qx = chebval(ts, coeffs)
        return xs * qx * qx - 1.0

    def err_inf(coeffs: np.ndarray) -> float:
        return float(np.max(np.abs(err_vec(coeffs))))

    def objective(coeffs: np.ndarray) -> float:
        return smooth_max_abs(err_vec(coeffs), p=smooth_p)

    best_coeffs = coeffs0.copy()
    best_err = err_inf(best_coeffs)

    for method, options in (
        ("Powell", {"maxiter": maxiter, "xtol": 1e-10, "ftol": 1e-12}),
        ("Nelder-Mead", {"maxiter": maxiter, "xatol": 1e-10, "fatol": 1e-12}),
    ):
        res = minimize(objective, best_coeffs, method=method, options=options)
        cand = np.array(res.x, dtype=np.float64)
        cand_err = err_inf(cand)
        if cand_err < best_err:
            best_coeffs = cand
            best_err = cand_err

    return best_err, best_coeffs


def max_rho_for_degree(
    degree: int,
    tau: float,
    rho_hi: float,
    rho_tol: float = 1.0e-6,
    fit_ngrid: int = 2001,
    verify_ngrid: int = 12001,
) -> tuple[float, float, np.ndarray]:
    """Return (rho_exact, exact_err_at_rho, coeffs)."""
    lo = 0.0
    hi = rho_hi
    best_coeffs = None

    # Warm start the feasible side while bisecting.
    while hi - lo > rho_tol:
        mid = 0.5 * (lo + hi)
        err_mid, coeffs_mid = fit_q_for_rho(
            degree=degree,
            rho=mid,
            ngrid=fit_ngrid,
            init=best_coeffs,
        )
        if err_mid <= tau:
            lo = mid
            best_coeffs = coeffs_mid
        else:
            hi = mid

    if best_coeffs is None:
        # This should only happen for an extremely tiny tau and tiny rho_hi.
        best_coeffs = np.array([1.0] + [0.0] * degree, dtype=np.float64)

    # Re-fit at the final rho using the best warm start, then verify on a denser grid.
    fit_err, best_coeffs = fit_q_for_rho(
        degree=degree,
        rho=lo,
        ngrid=max(fit_ngrid, verify_ngrid // 4),
        init=best_coeffs,
    )
    _, _, err_dense = dense_error_grid(best_coeffs, lo, verify_ngrid)
    exact_err = float(np.max(np.abs(err_dense)))
    return lo, exact_err, best_coeffs


# ---- bf16 scalar emulation for Chebyshev + Clenshaw ----


def round_bf16(x: np.ndarray | float | np.float32) -> np.ndarray | np.float32:
    arr = np.asarray(x, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    rounded = bits + bias
    out = (rounded & np.uint32(0xFFFF0000)).view(np.float32)
    if np.isscalar(x):
        return np.float32(out.item())
    return out.astype(np.float32, copy=False)


def clenshaw_bf16_scalar(t: float, coeffs: Iterable[float]) -> float:
    coeffs32 = round_bf16(np.asarray(list(coeffs), dtype=np.float32))
    t32 = round_bf16(np.float32(t))
    if len(coeffs32) == 1:
        return float(coeffs32[0])

    b1 = np.float32(0.0)
    b2 = np.float32(0.0)
    two_t = round_bf16(np.float32(2.0) * t32)
    for ak in coeffs32[:0:-1]:
        prod = round_bf16(two_t * b1)
        tmp = round_bf16(prod - b2)
        b0 = round_bf16(ak + tmp)
        b2, b1 = b1, b0

    prod = round_bf16(t32 * b1)
    q = round_bf16(coeffs32[0] + round_bf16(prod - b2))
    return float(q)


def bf16_error_for_coeffs(coeffs: np.ndarray, rho: float, ngrid: int = 4001) -> float:
    xs = np.linspace(max(1.0 - rho, 1.0e-12), 1.0 + rho, ngrid, dtype=np.float64)
    errs = np.empty_like(xs)
    for i, x in enumerate(xs):
        t = 0.0 if rho == 0.0 else (x - 1.0) / rho
        q = clenshaw_bf16_scalar(t, coeffs)
        xb = round_bf16(np.float32(x))
        qq = round_bf16(np.float32(q) * np.float32(q))
        y = round_bf16(xb * qq)
        errs[i] = abs(float(round_bf16(np.float32(y - 1.0))))
    return float(np.max(errs))


def max_rho_bf16_for_coeffs(
    coeffs: np.ndarray,
    tau: float,
    rho_hi: float,
    rho_tol: float = 1.0e-5,
    ngrid: int = 2001,
) -> float:
    lo, hi = 0.0, rho_hi
    while hi - lo > rho_tol:
        mid = 0.5 * (lo + hi)
        err_mid = bf16_error_for_coeffs(coeffs, mid, ngrid=ngrid)
        if err_mid <= tau:
            lo = mid
        else:
            hi = mid
    return lo


def scan_degrees(
    degrees: list[int],
    tau: float,
    rho_hi: float,
    fit_ngrid: int,
    verify_ngrid: int,
    do_bf16_check: bool,
) -> list[DegreeResult]:
    results: list[DegreeResult] = []
    for degree in degrees:
        rho_exact, exact_err, coeffs = max_rho_for_degree(
            degree=degree,
            tau=tau,
            rho_hi=rho_hi,
            fit_ngrid=fit_ngrid,
            verify_ngrid=verify_ngrid,
        )
        if degree > 0:
            rho_over_degree = rho_exact / degree
            atanh_score = math.atanh(min(rho_exact, 1.0 - 1.0e-12)) / degree
        else:
            rho_over_degree = float("nan")
            atanh_score = float("nan")

        rho_bf16_same_rho = None
        bf16_err_at_exact_rho = None
        if do_bf16_check:
            bf16_err_at_exact_rho = bf16_error_for_coeffs(coeffs, rho_exact)
            rho_bf16_same_rho = max_rho_bf16_for_coeffs(
                coeffs, tau=tau, rho_hi=rho_exact
            )

        results.append(
            DegreeResult(
                degree=degree,
                tau=tau,
                rho_exact=rho_exact,
                exact_err_at_rho=exact_err,
                rho_over_degree=rho_over_degree,
                atanh_rho_over_degree=atanh_score,
                rho_bf16_same_rho=rho_bf16_same_rho,
                bf16_err_at_exact_rho=bf16_err_at_exact_rho,
                coeffs_cheb=[float(v) for v in coeffs],
            )
        )
    return results


def print_summary(results: list[DegreeResult], print_degree: int) -> None:
    print()
    print("degree scan for first local reverse step")
    print("=" * 92)
    header = (
        f"{'deg':>3}  {'rho_exact':>12}  {'exact_err':>12}  {'rho/d':>12}  "
        f"{'atanh(rho)/d':>14}"
    )
    if results and results[0].bf16_err_at_exact_rho is not None:
        header += f"  {'bf16_err@rho':>14}  {'bf16_rho':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        row = (
            f"{r.degree:3d}  {r.rho_exact:12.6f}  {r.exact_err_at_rho:12.6e}  "
            f"{r.rho_over_degree:12.6f}  {r.atanh_rho_over_degree:14.6f}"
        )
        if r.bf16_err_at_exact_rho is not None:
            row += f"  {r.bf16_err_at_exact_rho:14.6e}  {r.rho_bf16_same_rho:12.6f}"
        print(row)

    exact_best = max(results, key=lambda r: r.rho_exact)
    cost_best = max(
        [r for r in results if r.degree > 0], key=lambda r: r.atanh_rho_over_degree
    )
    print()
    print(
        f"widest exact basin : degree {exact_best.degree} (rho = {exact_best.rho_exact:.6f})"
    )
    print(
        f"best cost proxy    : degree {cost_best.degree} "
        f"(atanh(rho)/d = {cost_best.atanh_rho_over_degree:.6f})"
    )
    print()
    print(f"degree-{print_degree} coeffs (Chebyshev on its own optimal interval):")
    target = next((r for r in results if r.degree == print_degree), None)
    if target is not None:
        print(json.dumps(target.coeffs_cheb))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tau", type=float, default=2.0**-7, help="target terminal radius / tolerance"
    )
    parser.add_argument("--min-degree", type=int, default=1)
    parser.add_argument("--max-degree", type=int, default=8)
    parser.add_argument(
        "--rho-hi", type=float, default=0.95, help="upper search limit for rho"
    )
    parser.add_argument("--fit-ngrid", type=int, default=2001)
    parser.add_argument("--verify-ngrid", type=int, default=12001)
    parser.add_argument("--skip-bf16", action="store_true")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--print-degree", type=int, default=3)
    args = parser.parse_args()

    degrees = list(range(args.min_degree, args.max_degree + 1))
    results = scan_degrees(
        degrees=degrees,
        tau=args.tau,
        rho_hi=args.rho_hi,
        fit_ngrid=args.fit_ngrid,
        verify_ngrid=args.verify_ngrid,
        do_bf16_check=not args.skip_bf16,
    )
    print_summary(results, args.print_degree)

    if args.json_out:
        payload = {
            "tau": args.tau,
            "rho_hi": args.rho_hi,
            "fit_ngrid": args.fit_ngrid,
            "verify_ngrid": args.verify_ngrid,
            "results": [asdict(r) for r in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print()
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
