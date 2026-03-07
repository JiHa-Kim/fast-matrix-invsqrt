#!/usr/bin/env python3
# design_two_step_coeffs.py
#
# Deterministic offline design of 2-step rational coefficients (c1, c2)
# for Gram-side inverse square root / damped polar pipelines.
#
# Model:
#   x -> phi_c(x) = x * ((x + c) / (c x + 1))^2
#   eta_plus(c, eta) = | log(phi_c(exp(eta))) |
#   two-step width: eta2(eta) = eta_plus(c2, eta_plus(c1, eta))
#
# Objective (min-max):
#   Given kappa0 = d / tau, eta0_max = 0.5 * log(kappa0),
#   minimize max_{eta in [0, eta0_max]} eta2(eta).
#
# Notes:
# - This uses the log-centered scalar model for contraction design.
# - In implementation you will typically use damping + trace-centering for robustness.
# - This script is for reproducibility: fixed grids, no randomness.

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ----------------------------
# Scalar map helpers
# ----------------------------


def phi_c(x: np.ndarray, c: float) -> np.ndarray:
    # x > 0, c > 0
    return x * ((x + c) / (c * x + 1.0)) ** 2


def eta_plus_vec(c: float, eta: np.ndarray) -> np.ndarray:
    # returns |log(phi_c(exp(eta)))| for vector eta
    x = np.exp(eta)
    y = phi_c(x, c)
    return np.abs(np.log(y))


def eta2_vec(c1: float, c2: float, eta: np.ndarray) -> np.ndarray:
    return eta_plus_vec(c2, eta_plus_vec(c1, eta))


# ----------------------------
# Search configuration
# ----------------------------


@dataclass(frozen=True)
class SearchConfig:
    c1_min: float
    c1_max: float
    c2_min: float
    c2_max: float
    c1_grid: int
    c2_grid: int
    eta_grid: int
    levels: int
    shrink: float
    topk: int


def _parse_tau(s: str) -> float:
    # Allows literals like "2**-8" or "0.0039".
    # We keep eval on a restricted namespace for convenience.
    allowed = {"__builtins__": {}}
    allowed.update({"math": math})
    allowed.update({"pow": pow})
    allowed.update({"abs": abs})
    allowed.update({"float": float})
    allowed.update({"int": int})
    allowed.update({"np": None})
    try:
        tau = eval(s, allowed, {})  # noqa: S307
    except Exception:
        tau = float(s)
    tau = float(tau)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"Invalid tau: {s}")
    return tau


def _dims_from_args(args: argparse.Namespace) -> List[int]:
    if args.dims:
        return [int(x) for x in args.dims]
    if args.dim_powers:
        p0, p1 = int(args.dim_powers[0]), int(args.dim_powers[1])
        return [2**p for p in range(p0, p1 + 1)]
    if args.dim_range:
        lo, hi, step = (
            int(args.dim_range[0]),
            int(args.dim_range[1]),
            int(args.dim_range[2]),
        )
        return list(range(lo, hi + 1, step))
    raise ValueError("Provide --dims, --dim-powers, or --dim-range.")


# ----------------------------
# Core optimizer
# ----------------------------


def _grid_logspace(lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(np.linspace(math.log(lo), math.log(hi), n, dtype=np.float64))


def _worst_eta2_for_pair(c1: float, c2: float, etas: np.ndarray) -> float:
    return float(np.max(eta2_vec(c1, c2, etas)))


def optimize_two_step_for_kappa0(kappa0: float, cfg: SearchConfig) -> Dict[str, float]:
    if not (math.isfinite(kappa0) and kappa0 >= 1.0):
        raise ValueError(f"Invalid kappa0: {kappa0}")

    eta0_max = 0.5 * math.log(kappa0)
    etas = np.linspace(0.0, eta0_max, cfg.eta_grid, dtype=np.float64)

    # Work in log space for stable zoom-in refinement
    log_c1_lo, log_c1_hi = math.log(cfg.c1_min), math.log(cfg.c1_max)
    log_c2_lo, log_c2_hi = math.log(cfg.c2_min), math.log(cfg.c2_max)

    best = {
        "worst_eta2": float("inf"),
        "c1": float("nan"),
        "c2": float("nan"),
    }

    # We will keep top-k candidates each level, then zoom around the best.
    for level in range(cfg.levels):
        c1s = np.exp(np.linspace(log_c1_lo, log_c1_hi, cfg.c1_grid, dtype=np.float64))
        c2s = np.exp(np.linspace(log_c2_lo, log_c2_hi, cfg.c2_grid, dtype=np.float64))

        # Evaluate in a semi-vectorized way:
        # For each c1, compute y1 = eta_plus(c1, etas) (length E),
        # then for all c2 at once compute eta_plus(c2, y1) via broadcasting.
        candidates: List[Tuple[float, float, float]] = []  # (worst, c1, c2)

        for c1 in c1s:
            y1 = eta_plus_vec(float(c1), etas)  # shape (E,)
            # Broadcast c2s over y1:
            # y2[i, :] = eta_plus(c2s[i], y1)
            x = np.exp(y1)[None, :]  # (1, E)
            c = c2s[:, None]  # (C2, 1)
            y = x * ((x + c) / (c * x + 1.0)) ** 2
            y2 = np.abs(np.log(y))  # (C2, E)
            worsts = np.max(y2, axis=1)  # (C2,)

            j = int(np.argmin(worsts))
            worst = float(worsts[j])
            c2 = float(c2s[j])

            candidates.append((worst, float(c1), c2))

        # Keep top-k by worst_eta2
        candidates.sort(key=lambda t: t[0])
        candidates = candidates[: max(1, cfg.topk)]

        # Update global best
        if candidates[0][0] < best["worst_eta2"]:
            best["worst_eta2"] = float(candidates[0][0])
            best["c1"] = float(candidates[0][1])
            best["c2"] = float(candidates[0][2])

        # Zoom around the current best (in log space)
        # Window size: current span * shrink
        span1 = (log_c1_hi - log_c1_lo) * cfg.shrink
        span2 = (log_c2_hi - log_c2_lo) * cfg.shrink
        log_c1_center = math.log(best["c1"])
        log_c2_center = math.log(best["c2"])
        log_c1_lo, log_c1_hi = log_c1_center - 0.5 * span1, log_c1_center + 0.5 * span1
        log_c2_lo, log_c2_hi = log_c2_center - 0.5 * span2, log_c2_center + 0.5 * span2

        # Clamp to original global ranges (so we never leave bounds)
        log_c1_lo = max(log_c1_lo, math.log(cfg.c1_min))
        log_c1_hi = min(log_c1_hi, math.log(cfg.c1_max))
        log_c2_lo = max(log_c2_lo, math.log(cfg.c2_min))
        log_c2_hi = min(log_c2_hi, math.log(cfg.c2_max))

    worst_eta2 = float(best["worst_eta2"])
    return {
        "kappa0": float(kappa0),
        "eta0_max": float(eta0_max),
        "c1": float(best["c1"]),
        "c2": float(best["c2"]),
        "worst_eta2": worst_eta2,
        "kappa_out_model": float(math.exp(2.0 * worst_eta2)),
    }


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Design 2-step rational coefficients (c1,c2) for damped Gram-side inverse sqrt / polar.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--tau",
        type=str,
        default="2**-8",
        help="Damping factor tau. Accepts floats or expressions like 2**-8.",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dims", nargs="+", help="List of dims, e.g. --dims 256 512 7168"
    )
    group.add_argument(
        "--dim-powers",
        nargs=2,
        help="Range of powers for dims, e.g. --dim-powers 8 14 for 2^8..2^14",
    )
    group.add_argument(
        "--dim-range", nargs=3, help="Range lo hi step, e.g. --dim-range 256 8192 256"
    )

    p.add_argument(
        "--out", type=str, default="coeffs.json", help="Output JSON filename."
    )

    # Search config
    p.add_argument("--c1-min", type=float, default=2.0)
    p.add_argument("--c1-max", type=float, default=300.0)
    p.add_argument("--c2-min", type=float, default=1.2)
    p.add_argument("--c2-max", type=float, default=10.0)

    p.add_argument("--c1-grid", type=int, default=200)
    p.add_argument("--c2-grid", type=int, default=200)
    p.add_argument("--eta-grid", type=int, default=800)

    p.add_argument(
        "--levels", type=int, default=5, help="Refinement levels (zoom-in iterations)."
    )
    p.add_argument(
        "--shrink",
        type=float,
        default=0.5,
        help="Zoom shrink factor per level in log-space.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=1,
        help="Keep top-k candidates each level (currently used only for ranking).",
    )

    args = p.parse_args()

    tau = _parse_tau(args.tau)
    dims = _dims_from_args(args)

    cfg = SearchConfig(
        c1_min=float(args.c1_min),
        c1_max=float(args.c1_max),
        c2_min=float(args.c2_min),
        c2_max=float(args.c2_max),
        c1_grid=int(args.c1_grid),
        c2_grid=int(args.c2_grid),
        eta_grid=int(args.eta_grid),
        levels=int(args.levels),
        shrink=float(args.shrink),
        topk=int(args.topk),
    )

    dim_results = []
    for d in dims:
        d = int(d)
        kappa0 = float(d) / float(tau)
        r = optimize_two_step_for_kappa0(kappa0, cfg)
        dim_results.append((d, r))
        print(
            f"d={d:6d}  kappa0={r['kappa0']:.6g}  "
            f"c1={r['c1']:.6g}  c2={r['c2']:.6g}  "
            f"kappa_out_model={r['kappa_out_model']:.6g}"
        )

    # order dims numerically
    dim_results.sort(key=lambda t: t[0])

    results = {
        "tau": float(tau),
        "definition": {
            "kappa0": "kappa0 = d / tau (coverage cap for normalized damped Gram, conservative)",
            "eta0_max": "0.5 * log(kappa0)",
            "objective": "minimize max_{eta in [0,eta0_max]} eta_plus(c2, eta_plus(c1, eta))",
            "kappa_out_model": "exp(2 * worst_eta2)",
        },
        "dims": {str(d): r for d, r in dim_results},
    }

    with open(args.out, "w", encoding="utf-8") as f:
        # DO NOT sort keys; we already built dims in numeric order.
        json.dump(results, f, indent=2)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
