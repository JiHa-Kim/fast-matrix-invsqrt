import math
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


def smooth_max(x, tau=80.0):
    m = x.max()
    return (torch.log(torch.mean(torch.exp(tau * (x - m)))) / tau) + m


def build_grid(lo, hi, n=8192):
    lo = max(float(lo), 1e-12)
    hi = max(float(hi), lo * 1.0001)
    return torch.exp(torch.linspace(math.log(lo), math.log(hi), n, dtype=torch.float64))


def fit_affine(
    lo,
    hi,
    init=None,
    q_floor=1e-3,
    pos_penalty=5e5,
    tau=80.0,
    iters=200,
    restarts=6,
    seed=0,
    p_val=2,
):
    ys = build_grid(lo, hi)
    rng = np.random.default_rng(seed)

    if init is None:
        # LS fit to y^(-1/p)
        y_np = ys.cpu().numpy()
        V = np.stack([np.ones_like(y_np), y_np], axis=1)
        target = y_np ** (-1.0 / p_val)
        init = np.linalg.lstsq(V, target, rcond=None)[0]

    base = torch.tensor(init, dtype=torch.float64)
    best, best_val = None, None

    for r in range(restarts):
        if r == 0:
            p0 = base.clone()
        else:
            noise = torch.tensor(rng.standard_normal(2), dtype=torch.float64)
            p0 = base * (1 + 0.25 * noise) + 0.1 * noise
        param = torch.nn.Parameter(p0)

        opt = torch.optim.LBFGS(
            [param], lr=0.8, max_iter=iters, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            a, b = param[0], param[1]
            q = a + b * ys
            err = torch.abs(1.0 - ys * (q**p_val))
            loss = smooth_max(err, tau=tau) + pos_penalty * torch.mean(
                torch.relu(q_floor - q) ** 2
            )
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            a, b = param[0], param[1]
            q = a + b * ys
            err = torch.abs(1.0 - ys * (q**p_val))
            val = err.max() + pos_penalty * torch.mean(torch.relu(q_floor - q) ** 2)
            if best_val is None or float(val) < float(best_val):
                best_val = val.clone()
                best = param.detach().clone()

    return [float(best[0]), float(best[1])]


def fit_quadratic(
    lo,
    hi,
    init=None,
    q_floor=1e-3,
    pos_penalty=5e5,
    tau=80.0,
    iters=250,
    restarts=6,
    seed=0,
    p_val=2,
):
    ys = build_grid(lo, hi)
    rng = np.random.default_rng(seed)

    if init is None:
        y_np = ys.cpu().numpy()
        V = np.stack([np.ones_like(y_np), y_np, y_np * y_np], axis=1)
        target = y_np ** (-1.0 / p_val)
        init = np.linalg.lstsq(V, target, rcond=None)[0]

    base = torch.tensor(init, dtype=torch.float64)
    best, best_val = None, None

    for r in range(restarts):
        if r == 0:
            p0 = base.clone()
        else:
            noise = torch.tensor(rng.standard_normal(3), dtype=torch.float64)
            p0 = base * (1 + 0.25 * noise) + 0.1 * noise
        param = torch.nn.Parameter(p0)

        opt = torch.optim.LBFGS(
            [param], lr=0.8, max_iter=iters, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            a, b, c = param[0], param[1], param[2]
            q = a + b * ys + c * ys * ys
            err = torch.abs(1.0 - ys * (q**p_val))
            loss = smooth_max(err, tau=tau) + pos_penalty * torch.mean(
                torch.relu(q_floor - q) ** 2
            )
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            a, b, c = param[0], param[1], param[2]
            q = a + b * ys + c * ys * ys
            err = torch.abs(1.0 - ys * (q**p_val))
            val = err.max() + pos_penalty * torch.mean(torch.relu(q_floor - q) ** 2)
            if best_val is None or float(val) < float(best_val):
                best_val = val.clone()
                best = param.detach().clone()

    return [float(best[0]), float(best[1]), float(best[2])]


# ---------------------------------------------------------------------------
# Certified machinery
# ---------------------------------------------------------------------------


def certify_positivity_quadratic(a, b, c, lo, hi, q_min=1e-6):
    """Exact positivity certification for q(y) = a + b*y + c*y^2 on [lo, hi].

    Returns True iff min(q(y)) > q_min on the interval.
    """
    q_lo = a + b * lo + c * lo * lo
    q_hi = a + b * hi + c * hi * hi
    min_val = min(q_lo, q_hi)
    # check vertex if parabola opens up and vertex is in [lo, hi]
    if c > 0:
        y_star = -b / (2.0 * c)
        if lo <= y_star <= hi:
            q_star = a + b * y_star + c * y_star * y_star
            min_val = min(min_val, q_star)
    return min_val > q_min


def inverse_newton_coeffs(p_val):
    """Return (a, b, c) for the inverse-Newton step q(y ) = (p+1-y)/p.

    In standard basis q(y) = a + b*y + c*y^2:
        a = (p+1)/p, b = -1/p, c = 0.
    """
    return (float(p_val + 1) / p_val, -1.0 / p_val, 0.0)


def affine_coeffs_from_b(b_slope: float) -> Tuple[float, float, float]:
    """Affine fixed-point family q_b(y)=1+b(y-1), returned in standard basis."""
    b_f = float(b_slope)
    return (1.0 - b_f, b_f, 0.0)


def affine_qmin(b_slope: float, lo: float, hi: float) -> float:
    """Exact min of q_b(y)=1+b(y-1) on [lo,hi]."""
    b_f = float(b_slope)
    lo_f = float(lo)
    hi_f = float(hi)
    q_lo = 1.0 + b_f * (lo_f - 1.0)
    q_hi = 1.0 + b_f * (hi_f - 1.0)
    return float(min(q_lo, q_hi))


def affine_b_feasible_bounds(
    lo: float,
    hi: float,
    *,
    q_floor: float = 1e-6,
    b_cap: float = 8.0,
) -> Tuple[float, float]:
    """Bounds for b such that q_b(y)=1+b(y-1) >= q_floor on [lo,hi]."""
    lo_f = float(lo)
    hi_f = float(hi)
    qf = float(q_floor)
    if qf <= 0.0:
        raise ValueError(f"q_floor must be > 0, got {q_floor}")

    b_lo = -float("inf")
    b_hi = float("inf")
    for y in (lo_f, hi_f):
        dy = float(y - 1.0)
        if abs(dy) <= 1e-15:
            continue
        thr = (qf - 1.0) / dy
        if dy > 0.0:
            b_lo = max(b_lo, float(thr))
        else:
            b_hi = min(b_hi, float(thr))

    # Typical normalization uses hi ~= 1 and may leave one side unbounded.
    # Use a conservative finite cap for robust 1D search.
    cap_terms = [max(float(b_cap), 1.0)]
    if abs(lo_f - 1.0) > 1e-9:
        cap_terms.append(4.0 / abs(lo_f - 1.0))
    if abs(hi_f - 1.0) > 1e-9:
        cap_terms.append(4.0 / abs(hi_f - 1.0))
    cap = min(max(cap_terms), 1e4)

    if not math.isfinite(b_lo):
        b_lo = -cap
    if not math.isfinite(b_hi):
        b_hi = cap

    return float(b_lo), float(b_hi)


def interval_update_affine_exact(
    b_slope: float,
    lo: float,
    hi: float,
    *,
    p_val: int = 2,
) -> Tuple[float, float]:
    """Exact interval update for φ(y)=y*(1+b(y-1))^p via closed-form critical point."""
    b_f = float(b_slope)
    lo_f = float(lo)
    hi_f = float(hi)
    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")

    try:
        cand = [lo_f, hi_f]
        if abs(b_f) > 1e-15:
            y_star = (b_f - 1.0) / (b_f * (1.0 + float(p_i)))
            if lo_f <= y_star <= hi_f:
                cand.append(float(y_star))

        vals = []
        for y in cand:
            q = 1.0 + b_f * (float(y) - 1.0)
            vals.append(float(y) * float(q**p_i))

        lo_new = float(min(vals))
        hi_new = float(max(vals))
        if not (math.isfinite(lo_new) and math.isfinite(hi_new)):
            raise FloatingPointError("non-finite affine interval extrema")
        return lo_new, hi_new
    except Exception:
        abc = affine_coeffs_from_b(b_f)
        return interval_update_quadratic_exact(abc, lo_f, hi_f, p_val=p_i)


def _bpow_gemm_cost(exp: int) -> int:
    """Estimated GEMM count for _bpow(..., p=exp) in coupled kernels."""
    e = int(exp)
    if e <= 0:
        raise ValueError(f"exp must be >= 1, got {exp}")
    if e == 1:
        return 0
    if e == 2:
        return 1
    if e == 4:
        return 2
    # Generic binary exponentiation:
    # squarings = bit_length - 1, multiplies-into-result = popcount - 1.
    return (e.bit_length() - 1) + (int(e).bit_count() - 1)


def coupled_apply_step_gemm_cost(
    p_val: int,
    *,
    affine_step: bool,
    include_y_update: bool = True,
) -> int:
    """Estimate GEMM count of one coupled apply step.

    The estimate matches the current specialized kernels:
      - build B: 0 GEMM when affine (c=0 fast path), else 1 GEMM.
      - apply to RHS: 1 GEMM.
      - Y update:
          p=1 -> 1 GEMM
          p=2 -> 2 GEMMs
          p=3 -> 3 GEMMs
          even p>2 -> bpow(p/2) + 2 GEMMs
          odd p>3 -> bpow((p-1)/2) + 3 GEMMs
    """
    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")

    gemms = 0 if affine_step else 1  # build B
    gemms += 1  # apply: Z <- B Z

    if not include_y_update:
        return gemms

    if p_i == 1:
        gemms += 1
    elif p_i == 2:
        gemms += 2
    elif p_i == 3:
        gemms += 3
    elif p_i % 2 == 0:
        gemms += _bpow_gemm_cost(p_i // 2) + 2
    else:
        gemms += _bpow_gemm_cost((p_i - 1) // 2) + 3

    return gemms


def coupled_apply_step_cost_units(
    p_val: int,
    *,
    affine_step: bool,
    include_y_update: bool = True,
    rhs_to_n_ratio: float = 1.0,
    terminal_rhs_direct: bool = False,
) -> float:
    """Approximate per-step cost in normalized n^3 units.

    This extends `coupled_apply_step_gemm_cost` with shape awareness:
      - full n x n GEMM has unit cost 1
      - rhs apply GEMM (n x n) @ (n x k) has cost (k/n)
    and terminal RHS-direct apply modeling for quadratic steps (k<n), where
    the kernel uses two rhs GEMMs and skips dense B materialization.
    """
    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")

    rhs_ratio = float(rhs_to_n_ratio)
    if (not math.isfinite(rhs_ratio)) or rhs_ratio <= 0.0:
        raise ValueError(
            f"rhs_to_n_ratio must be finite and > 0, got {rhs_to_n_ratio}"
        )

    is_terminal_rhs_direct = bool(terminal_rhs_direct) and (not include_y_update)
    if is_terminal_rhs_direct and (not affine_step):
        # Quadratic terminal RHS-direct path:
        # out <- (aI + bY + cY^2)M via two rhs GEMMs; no dense B build.
        build_units = 0.0
        apply_units = 2.0 * rhs_ratio
    else:
        # Standard path: optional dense B build plus one rhs apply.
        build_units = 0.0 if bool(affine_step) else 1.0
        apply_units = rhs_ratio

    y_units = 0.0
    if include_y_update:
        if p_i == 1:
            y_units = 1.0
        elif p_i == 2:
            y_units = 2.0
        elif p_i == 3:
            y_units = 3.0
        elif p_i % 2 == 0:
            y_units = float(_bpow_gemm_cost(p_i // 2) + 2)
        else:
            y_units = float(_bpow_gemm_cost((p_i - 1) // 2) + 3)

    return float(build_units + apply_units + y_units)


def interval_error_to_identity(lo: float, hi: float) -> float:
    """Scalar interval error proxy used for schedule planning."""
    lo_f = float(lo)
    hi_f = float(hi)
    return max(abs(1.0 - lo_f), abs(hi_f - 1.0))


def interval_log_width(lo: float, hi: float) -> float:
    lo_f = max(float(lo), 1e-15)
    hi_f = max(float(hi), lo_f * 1.0001)
    return float(math.log(hi_f) - math.log(lo_f))


def local_quadratic_coeffs_from_alpha(alpha: float, p_val: int) -> Tuple[float, float, float]:
    """Local-basis family q(y)=1-(1/p)(y-1)+alpha*(y-1)^2 in standard basis."""
    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")
    inv_p = 1.0 / float(p_i)
    a = 1.0 + inv_p + float(alpha)
    b = -inv_p - 2.0 * float(alpha)
    c = float(alpha)
    return (a, b, c)


def local_quadratic_qmin(alpha: float, p_val: int, lo: float, hi: float) -> float:
    """Exact min of q_alpha(y) over [lo, hi], for q_alpha in local basis."""
    a, b, c = local_quadratic_coeffs_from_alpha(alpha, p_val)
    lo_f = float(lo)
    hi_f = float(hi)
    q_lo = a + b * lo_f + c * lo_f * lo_f
    q_hi = a + b * hi_f + c * hi_f * hi_f
    q_min = min(q_lo, q_hi)
    if c > 0.0:
        y_star = -b / (2.0 * c)
        if lo_f <= y_star <= hi_f:
            q_min = min(q_min, a + b * y_star + c * y_star * y_star)
    return float(q_min)


def _minimax_alpha_bounds(p_val: int) -> Tuple[float, float]:
    p_i = int(p_val)
    center = float(p_i + 1) / (2.0 * float(p_i) * float(p_i))
    lo = min(-0.75, center - 1.5)
    hi = max(2.5, center + 1.5)
    return (float(lo), float(hi))


def solve_local_alpha_minimax(
    *,
    p_val: int,
    lo: float,
    hi: float,
    q_floor: float = 1e-6,
    coarse_points: int = 65,
    refine_iters: int = 18,
    cache_quant: float = 1e-4,
) -> Tuple[float, Dict[str, float]]:
    """Approximate 1D minimax solve over local-basis alpha.

    Objective: minimize max_{y in [lo,hi]} |1 - y q_alpha(y)^p|.
    Uses a robust coarse scan plus local golden-section refinement.
    """
    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")
    if coarse_points < 5:
        raise ValueError(f"coarse_points must be >= 5, got {coarse_points}")
    q = float(cache_quant)
    if q <= 0.0:
        raise ValueError(f"cache_quant must be > 0, got {cache_quant}")

    lo_f = max(float(lo), 1e-12)
    hi_f = max(float(hi), lo_f * 1.0001)
    lo_q = round(lo_f / q) * q
    hi_q = round(hi_f / q) * q
    hi_q = max(hi_q, lo_q * 1.0001)

    alpha, objective, eval_count, fallback_ns = _solve_local_alpha_minimax_cached(
        p_i,
        float(lo_q),
        float(hi_q),
        float(q_floor),
        int(coarse_points),
        int(refine_iters),
    )
    return float(alpha), {
        "objective": float(objective),
        "eval_count": float(eval_count),
        "fallback_ns": float(fallback_ns),
    }


def _solve_local_alpha_minimax_impl(
    p_i: int,
    lo_f: float,
    hi_f: float,
    q_floor_f: float,
    coarse_points: int,
    refine_iters: int,
) -> Tuple[float, float, float, float]:
    alpha_lo, alpha_hi = _minimax_alpha_bounds(p_i)

    eval_count = 0

    def _objective(alpha: float) -> float:
        nonlocal eval_count
        eval_count += 1
        if local_quadratic_qmin(alpha, p_i, lo_f, hi_f) <= q_floor_f:
            return float("inf")
        abc = local_quadratic_coeffs_from_alpha(alpha, p_i)
        lo2, hi2 = interval_update_quadratic_exact(abc, lo_f, hi_f, p_val=p_i)
        return float(interval_error_to_identity(lo2, hi2))

    grid = np.linspace(alpha_lo, alpha_hi, int(coarse_points), dtype=np.float64)
    vals = np.array([_objective(float(a)) for a in grid], dtype=np.float64)
    feasible = np.isfinite(vals)
    if not np.any(feasible):
        return 0.0, float("inf"), float(eval_count), 1.0

    idx = int(np.nanargmin(vals))
    best_alpha = float(grid[idx])
    best_val = float(vals[idx])

    left_idx = max(0, idx - 1)
    right_idx = min(len(grid) - 1, idx + 1)
    left = float(grid[left_idx])
    right = float(grid[right_idx])
    if right <= left:
        left = max(alpha_lo, best_alpha - 0.05)
        right = min(alpha_hi, best_alpha + 0.05)

    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - inv_phi * (right - left)
    x2 = left + inv_phi * (right - left)
    f1 = _objective(x1)
    f2 = _objective(x2)
    for _ in range(int(refine_iters)):
        if f1 <= f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - inv_phi * (right - left)
            f1 = _objective(x1)
        else:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + inv_phi * (right - left)
            f2 = _objective(x2)

    for a, v in ((x1, f1), (x2, f2)):
        if math.isfinite(v) and v < best_val:
            best_alpha = float(a)
            best_val = float(v)

    return best_alpha, float(best_val), float(eval_count), 0.0


@lru_cache(maxsize=4096)
def _solve_local_alpha_minimax_cached(
    p_i: int,
    lo_f: float,
    hi_f: float,
    q_floor_f: float,
    coarse_points: int,
    refine_iters: int,
) -> Tuple[float, float, float, float]:
    return _solve_local_alpha_minimax_impl(
        int(p_i),
        float(lo_f),
        float(hi_f),
        float(q_floor_f),
        int(coarse_points),
        int(refine_iters),
    )


def solve_local_affine_b_optimal(
    *,
    p_val: int,
    lo: float,
    hi: float,
    q_floor: float = 1e-6,
    coarse_points: int = 65,
    refine_iters: int = 18,
    cache_quant: float = 1e-4,
    b_cap: float = 8.0,
) -> Tuple[float, Dict[str, float]]:
    """Approximate 1D minimax solve over affine fixed-point family q_b(y)=1+b(y-1)."""
    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")
    if coarse_points < 5:
        raise ValueError(f"coarse_points must be >= 5, got {coarse_points}")
    q = float(cache_quant)
    if q <= 0.0:
        raise ValueError(f"cache_quant must be > 0, got {cache_quant}")
    if float(b_cap) <= 0.0:
        raise ValueError(f"b_cap must be > 0, got {b_cap}")

    lo_f = max(float(lo), 1e-12)
    hi_f = max(float(hi), lo_f * 1.0001)
    lo_q = round(lo_f / q) * q
    hi_q = round(hi_f / q) * q
    hi_q = max(hi_q, lo_q * 1.0001)

    b_star, objective, eval_count, fallback_ns = _solve_local_affine_b_optimal_cached(
        p_i,
        float(lo_q),
        float(hi_q),
        float(q_floor),
        int(coarse_points),
        int(refine_iters),
        float(b_cap),
    )
    return float(b_star), {
        "objective": float(objective),
        "eval_count": float(eval_count),
        "fallback_ns": float(fallback_ns),
    }


def _solve_local_affine_b_optimal_impl(
    p_i: int,
    lo_f: float,
    hi_f: float,
    q_floor_f: float,
    coarse_points: int,
    refine_iters: int,
    b_cap_f: float,
) -> Tuple[float, float, float, float]:
    b_ns = -1.0 / float(p_i)
    b_lo, b_hi = affine_b_feasible_bounds(
        lo_f, hi_f, q_floor=float(q_floor_f), b_cap=float(b_cap_f)
    )

    eval_count = 0

    def _objective(b_slope: float) -> float:
        nonlocal eval_count
        eval_count += 1
        if affine_qmin(b_slope, lo_f, hi_f) <= q_floor_f:
            return float("inf")
        lo2, hi2 = interval_update_affine_exact(b_slope, lo_f, hi_f, p_val=p_i)
        return float(interval_error_to_identity(lo2, hi2))

    if not (math.isfinite(b_lo) and math.isfinite(b_hi) and b_lo < b_hi):
        ns_obj = _objective(b_ns)
        return float(b_ns), float(ns_obj), float(eval_count), 1.0

    grid = np.linspace(float(b_lo), float(b_hi), int(coarse_points), dtype=np.float64)
    vals = np.array([_objective(float(b)) for b in grid], dtype=np.float64)
    feasible = np.isfinite(vals)
    if not np.any(feasible):
        ns_obj = _objective(b_ns)
        return float(b_ns), float(ns_obj), float(eval_count), 1.0

    idx = int(np.nanargmin(vals))
    best_b = float(grid[idx])
    best_val = float(vals[idx])

    left_idx = max(0, idx - 1)
    right_idx = min(len(grid) - 1, idx + 1)
    left = float(grid[left_idx])
    right = float(grid[right_idx])
    if right <= left:
        span = max(float(b_hi - b_lo), 1e-6)
        left = max(float(b_lo), best_b - 0.1 * span)
        right = min(float(b_hi), best_b + 0.1 * span)

    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - inv_phi * (right - left)
    x2 = left + inv_phi * (right - left)
    f1 = _objective(x1)
    f2 = _objective(x2)
    for _ in range(int(refine_iters)):
        if f1 <= f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - inv_phi * (right - left)
            f1 = _objective(x1)
        else:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + inv_phi * (right - left)
            f2 = _objective(x2)

    for b_slope, val in ((x1, f1), (x2, f2), (b_ns, _objective(b_ns))):
        if math.isfinite(val) and val < best_val:
            best_b = float(b_slope)
            best_val = float(val)

    return float(best_b), float(best_val), float(eval_count), 0.0


@lru_cache(maxsize=4096)
def _solve_local_affine_b_optimal_cached(
    p_i: int,
    lo_f: float,
    hi_f: float,
    q_floor_f: float,
    coarse_points: int,
    refine_iters: int,
    b_cap_f: float,
) -> Tuple[float, float, float, float]:
    return _solve_local_affine_b_optimal_impl(
        int(p_i),
        float(lo_f),
        float(hi_f),
        float(q_floor_f),
        int(coarse_points),
        int(refine_iters),
        float(b_cap_f),
    )


def _quad_pos_ok(a: float, b: float, c: float, lo: float, hi: float, q_floor: float) -> bool:
    return certify_positivity_quadratic(a, b, c, float(lo), float(hi), q_min=float(q_floor))

def _aff_pos_ok(a: float, b: float, lo: float, hi: float, q_floor: float) -> bool:
    q_lo = a + b * float(lo)
    q_hi = a + b * float(hi)
    return min(q_lo, q_hi) > float(q_floor)


def plan_coupled_local_minimax_schedule(
    base_coeffs: Sequence[Tuple[float, float, float]],
    *,
    p_val: int,
    lo_init: float,
    hi_init: float = 1.0,
    min_rel_improve: float = 0.0,
    min_ns_logwidth_rel_improve: float = 0.0,
    terminal_last_step: bool = True,
    rhs_to_n_ratio: float = 1.0,
    terminal_rhs_direct: bool = False,
    q_floor: float = 1e-6,
) -> Tuple[List[Tuple[float, float, float]], Dict[str, float]]:
    """Greedy cost-aware planner with a local-basis minimax candidate.

    Per step candidates:
      - baseline quadratic coefficient triple
      - inverse-Newton affine step (alpha=0)
      - local-basis minimax-alpha step

    The minimax candidate is accepted only if it improves mapped interval log-width
    over Newton by at least `min_ns_logwidth_rel_improve`.
    """
    if len(base_coeffs) == 0:
        raise ValueError("base_coeffs must contain at least one coefficient triple")
    if float(min_rel_improve) < 0.0:
        raise ValueError(f"min_rel_improve must be >= 0, got {min_rel_improve}")
    if float(min_ns_logwidth_rel_improve) < 0.0:
        raise ValueError(
            "min_ns_logwidth_rel_improve must be >= 0, "
            f"got {min_ns_logwidth_rel_improve}"
        )

    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")
    rhs_ratio = float(rhs_to_n_ratio)
    if (not math.isfinite(rhs_ratio)) or rhs_ratio <= 0.0:
        raise ValueError(
            f"rhs_to_n_ratio must be finite and > 0, got {rhs_to_n_ratio}"
        )

    lo = max(float(lo_init), 1e-12)
    hi = max(float(hi_init), lo * 1.0001)
    coeffs = [(float(a), float(b), float(c)) for (a, b, c) in base_coeffs]
    ns_abc = inverse_newton_coeffs(p_i)

    planned: List[Tuple[float, float, float]] = []
    counts: Dict[str, int] = {"base": 0, "newton": 0, "minimax": 0}

    eps = 1e-15
    improve_thr = float(min_rel_improve)
    ns_gate = float(min_ns_logwidth_rel_improve)
    minimax_eval_sum = 0.0

    for t, base_abc in enumerate(coeffs):
        is_terminal_step = bool(terminal_last_step) and (t == len(coeffs) - 1)
        include_y_update = not is_terminal_step
        use_rhs_direct = bool(terminal_rhs_direct) and is_terminal_step
        err_cur = max(interval_error_to_identity(lo, hi), eps)

        def _evaluate(abc: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
            a, b, c = abc
            if not _quad_pos_ok(a, b, c, lo, hi, q_floor=float(q_floor)):
                return float("inf"), lo, hi, float("inf")
            lo2, hi2 = interval_update_quadratic_exact(abc, lo, hi, p_val=p_i)
            err2 = max(interval_error_to_identity(lo2, hi2), eps)
            affine = abs(float(c)) <= 1e-15
            cost_units = coupled_apply_step_cost_units(
                p_i,
                affine_step=affine,
                include_y_update=include_y_update,
                rhs_to_n_ratio=rhs_ratio,
                terminal_rhs_direct=use_rhs_direct,
            )
            score = (err2 / err_cur) ** (1.0 / max(float(cost_units), eps))
            w2 = interval_log_width(lo2, hi2)
            return float(score), float(lo2), float(hi2), float(w2)

        score_base, lo_base, hi_base, w_base = _evaluate(base_abc)
        score_ns, lo_ns, hi_ns, w_ns = _evaluate(ns_abc)

        alpha_star, alpha_meta = solve_local_alpha_minimax(
            p_val=p_i, lo=lo, hi=hi, q_floor=float(q_floor)
        )
        minimax_eval_sum += float(alpha_meta["eval_count"])
        mm_abc = local_quadratic_coeffs_from_alpha(alpha_star, p_i)
        score_mm, lo_mm, hi_mm, w_mm = _evaluate(mm_abc)

        # Gate minimax vs Newton log-width unless Newton itself is infeasible.
        mm_allowed = math.isfinite(score_mm)
        if mm_allowed and math.isfinite(w_ns):
            base_den = max(abs(w_ns), eps)
            rel = (w_ns - w_mm) / base_den
            mm_allowed = rel > ns_gate
        if not mm_allowed:
            score_mm = float("inf")

        cand = [
            ("base", score_base, lo_base, hi_base, base_abc),
            ("newton", score_ns, lo_ns, hi_ns, ns_abc),
            ("minimax", score_mm, lo_mm, hi_mm, mm_abc),
        ]
        finite = [x for x in cand if math.isfinite(x[1])]
        if len(finite) == 0:
            choice = ("newton", score_ns, lo_ns, hi_ns, ns_abc)
        else:
            finite.sort(key=lambda x: x[1])
            choice = finite[0]
            # improvement gate: if best doesn't beat base enough, keep base.
            if math.isfinite(score_base):
                rel_imp = (score_base - choice[1]) / max(score_base, eps)
                if rel_imp <= improve_thr:
                    choice = ("base", score_base, lo_base, hi_base, base_abc)

        tag, _, lo_next, hi_next, abc = choice
        planned.append(abc)
        counts[tag] += 1
        lo = max(float(lo_next), 1e-15)
        hi = max(float(hi_next), lo * 1.0001)

    meta: Dict[str, float] = {
        "base_steps": float(counts["base"]),
        "newton_steps": float(counts["newton"]),
        "minimax_steps": float(counts["minimax"]),
        "pred_lo_final": float(lo),
        "pred_hi_final": float(hi),
        "pred_err_final": float(interval_error_to_identity(lo, hi)),
        "alpha_eval_avg_per_step": float(minimax_eval_sum / float(len(coeffs))),
    }
    return planned, meta


def plan_coupled_quadratic_newton_schedule(
    base_coeffs: Sequence[Tuple[float, float, float]],
    *,
    p_val: int,
    lo_init: float,
    hi_init: float = 1.0,
    min_rel_improve: float = 0.0,
    terminal_last_step: bool = True,
    rhs_to_n_ratio: float = 1.0,
    terminal_rhs_direct: bool = False,
    odd_p_q_floor: float = 1e-6,
) -> Tuple[List[Tuple[float, float, float]], Dict[str, float]]:
    """Greedy cost-aware schedule planner for coupled PE apply.

    At each step, picks between:
      - the provided quadratic coefficient triple, and
      - inverse-Newton affine coefficients (c=0),
    by minimizing predicted interval-error-per-GEMM on the current scalar interval.

    The interval model uses `interval_update_quadratic_exact(...)` (exact for p=2,
    conservative grid fallback for other p values).
    """
    if len(base_coeffs) == 0:
        raise ValueError("base_coeffs must contain at least one coefficient triple")
    if float(min_rel_improve) < 0.0:
        raise ValueError(
            f"min_rel_improve must be >= 0, got {min_rel_improve}"
        )
    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")
    rhs_ratio = float(rhs_to_n_ratio)
    if (not math.isfinite(rhs_ratio)) or rhs_ratio <= 0.0:
        raise ValueError(
            f"rhs_to_n_ratio must be finite and > 0, got {rhs_to_n_ratio}"
        )

    q_floor = float(odd_p_q_floor)  # FIX: use for all p
    lo = max(float(lo_init), 1e-12)
    hi = max(float(hi_init), lo * 1.0001)
    coeffs = [
        (float(a), float(b), float(c)) for (a, b, c) in list(base_coeffs)
    ]
    ns_abc = inverse_newton_coeffs(p_i)

    planned: List[Tuple[float, float, float]] = []
    newton_steps = 0
    base_steps = 0

    eps = 1e-15
    improve_thr = float(min_rel_improve)

    for t, base_abc in enumerate(coeffs):
        is_terminal_step = bool(terminal_last_step) and (t == len(coeffs) - 1)
        include_y_update = not is_terminal_step
        use_rhs_direct = bool(terminal_rhs_direct) and is_terminal_step
        err_cur = max(interval_error_to_identity(lo, hi), eps)

        def _evaluate(abc: Tuple[float, float, float]) -> Tuple[float, float, float]:
            a, b, c = abc
            if not _quad_pos_ok(a, b, c, lo, hi, q_floor=q_floor):
                return float("inf"), lo, hi
            lo2, hi2 = interval_update_quadratic_exact(abc, lo, hi, p_val=p_i)
            err2 = max(interval_error_to_identity(lo2, hi2), eps)
            affine = abs(float(c)) <= 1e-15
            cost_units = coupled_apply_step_cost_units(
                p_i,
                affine_step=affine,
                include_y_update=include_y_update,
                rhs_to_n_ratio=rhs_ratio,
                terminal_rhs_direct=use_rhs_direct,
            )
            # Smaller is better: predicted contraction normalized by compute units.
            score = (err2 / err_cur) ** (1.0 / max(float(cost_units), eps))
            return float(score), float(lo2), float(hi2)

        score_base, lo_base, hi_base = _evaluate(base_abc)
        score_ns, lo_ns, hi_ns = _evaluate(ns_abc)

        use_newton = False
        if math.isfinite(score_ns):
            if not math.isfinite(score_base):
                use_newton = True
            else:
                rel_improve = (score_base - score_ns) / max(score_base, eps)
                use_newton = rel_improve > improve_thr

        if use_newton:
            planned.append(ns_abc)
            lo, hi = lo_ns, hi_ns
            newton_steps += 1
        else:
            planned.append(base_abc)
            lo, hi = lo_base, hi_base
            base_steps += 1

        lo = max(float(lo), 1e-15)
        hi = max(float(hi), lo * 1.0001)

    meta: Dict[str, float] = {
        "base_steps": float(base_steps),
        "newton_steps": float(newton_steps),
        "pred_lo_final": float(lo),
        "pred_hi_final": float(hi),
        "pred_err_final": float(interval_error_to_identity(lo, hi)),
    }
    return planned, meta


def plan_coupled_quadratic_affine_opt_schedule(
    base_coeffs: Sequence[Tuple[float, float, float]],
    *,
    p_val: int,
    lo_init: float,
    hi_init: float = 1.0,
    min_rel_improve: float = 0.0,
    terminal_last_step: bool = True,
    rhs_to_n_ratio: float = 1.0,
    terminal_rhs_direct: bool = False,
    q_floor: float = 1e-6,
) -> Tuple[List[Tuple[float, float, float]], Dict[str, float]]:
    """Greedy cost-aware planner with interval-optimal affine candidate.

    At each step, evaluates:
      - baseline quadratic coefficients,
      - inverse-Newton affine coefficients,
      - interval-optimal affine q_b(y)=1+b(y-1),
    and picks the best predicted contraction-per-GEMM candidate.
    """
    if len(base_coeffs) == 0:
        raise ValueError("base_coeffs must contain at least one coefficient triple")
    if float(min_rel_improve) < 0.0:
        raise ValueError(f"min_rel_improve must be >= 0, got {min_rel_improve}")

    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")
    rhs_ratio = float(rhs_to_n_ratio)
    if (not math.isfinite(rhs_ratio)) or rhs_ratio <= 0.0:
        raise ValueError(
            f"rhs_to_n_ratio must be finite and > 0, got {rhs_to_n_ratio}"
        )

    lo = max(float(lo_init), 1e-12)
    hi = max(float(hi_init), lo * 1.0001)
    coeffs = [(float(a), float(b), float(c)) for (a, b, c) in list(base_coeffs)]
    ns_abc = inverse_newton_coeffs(p_i)

    planned: List[Tuple[float, float, float]] = []
    counts: Dict[str, int] = {"base": 0, "newton": 0, "affine_opt": 0}
    affine_eval_sum = 0.0

    eps = 1e-15
    improve_thr = float(min_rel_improve)

    for t, base_abc in enumerate(coeffs):
        is_terminal_step = bool(terminal_last_step) and (t == len(coeffs) - 1)
        include_y_update = not is_terminal_step
        use_rhs_direct = bool(terminal_rhs_direct) and is_terminal_step
        err_cur = max(interval_error_to_identity(lo, hi), eps)

        def _score_from_interval(
            lo2: float, hi2: float, *, affine_step: bool
        ) -> float:
            err2 = max(interval_error_to_identity(lo2, hi2), eps)
            cost_units = coupled_apply_step_cost_units(
                p_i,
                affine_step=affine_step,
                include_y_update=include_y_update,
                rhs_to_n_ratio=rhs_ratio,
                terminal_rhs_direct=use_rhs_direct,
            )
            return float((err2 / err_cur) ** (1.0 / max(float(cost_units), eps)))

        def _evaluate_quad(
            abc: Tuple[float, float, float]
        ) -> Tuple[float, float, float]:
            a, b, c = abc
            if not _quad_pos_ok(a, b, c, lo, hi, q_floor=float(q_floor)):
                return float("inf"), lo, hi
            lo2, hi2 = interval_update_quadratic_exact(abc, lo, hi, p_val=p_i)
            score = _score_from_interval(
                float(lo2), float(hi2), affine_step=(abs(float(c)) <= 1e-15)
            )
            return score, float(lo2), float(hi2)

        def _evaluate_affine_opt() -> Tuple[float, float, float, Tuple[float, float, float]]:
            b_star, b_meta = solve_local_affine_b_optimal(
                p_val=p_i,
                lo=lo,
                hi=hi,
                q_floor=float(q_floor),
            )
            nonlocal affine_eval_sum
            affine_eval_sum += float(b_meta.get("eval_count", 0.0))
            if affine_qmin(b_star, lo, hi) <= float(q_floor):
                return float("inf"), lo, hi, affine_coeffs_from_b(b_star)
            lo2, hi2 = interval_update_affine_exact(b_star, lo, hi, p_val=p_i)
            score = _score_from_interval(float(lo2), float(hi2), affine_step=True)
            return score, float(lo2), float(hi2), affine_coeffs_from_b(b_star)

        score_base, lo_base, hi_base = _evaluate_quad(base_abc)
        score_ns, lo_ns, hi_ns = _evaluate_quad(ns_abc)
        score_aff, lo_aff, hi_aff, abc_aff = _evaluate_affine_opt()

        cand = [
            ("base", score_base, lo_base, hi_base, base_abc),
            ("newton", score_ns, lo_ns, hi_ns, ns_abc),
            ("affine_opt", score_aff, lo_aff, hi_aff, abc_aff),
        ]

        finite = [x for x in cand if math.isfinite(x[1])]
        if len(finite) == 0:
            choice = ("newton", score_ns, lo_ns, hi_ns, ns_abc)
        else:
            finite.sort(key=lambda x: x[1])
            choice = finite[0]
            if math.isfinite(score_base):
                rel_imp = (score_base - choice[1]) / max(score_base, eps)
                if rel_imp <= improve_thr:
                    choice = ("base", score_base, lo_base, hi_base, base_abc)

        tag, _, lo_next, hi_next, abc = choice
        planned.append(abc)
        counts[tag] += 1
        lo = max(float(lo_next), 1e-15)
        hi = max(float(hi_next), lo * 1.0001)

    meta: Dict[str, float] = {
        "base_steps": float(counts["base"]),
        "newton_steps": float(counts["newton"]),
        "affine_opt_steps": float(counts["affine_opt"]),
        "pred_lo_final": float(lo),
        "pred_hi_final": float(hi),
        "pred_err_final": float(interval_error_to_identity(lo, hi)),
        "affine_eval_avg_per_step": float(affine_eval_sum / float(len(coeffs))),
    }
    return planned, meta


def truncate_coupled_schedule_by_interval_error(
    coeffs: Sequence[Tuple[float, float, float]],
    *,
    p_val: int,
    lo_init: float,
    hi_init: float = 1.0,
    target_err: Optional[float] = None,
    min_steps: int = 1,
    q_floor: float = 1e-6,
) -> Tuple[List[Tuple[float, float, float]], Dict[str, float]]:
    """Return the shortest coefficient prefix meeting a predicted interval target.

    Uses the same scalar interval model as the online schedule planners.
    When target_err is None, returns the full schedule unchanged.
    """
    if len(coeffs) == 0:
        raise ValueError("coeffs must contain at least one coefficient triple")
    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")
    min_steps_i = int(min_steps)
    if min_steps_i < 1:
        raise ValueError(f"min_steps must be >= 1, got {min_steps}")
    if target_err is not None and float(target_err) <= 0.0:
        raise ValueError(
            f"target_err must be > 0 when provided, got {target_err}"
        )

    lo = max(float(lo_init), 1e-12)
    hi = max(float(hi_init), lo * 1.0001)
    coeffs_f = [(float(a), float(b), float(c)) for (a, b, c) in coeffs]
    target = float(target_err) if target_err is not None else None
    target_met = False
    steps_used = len(coeffs_f)

    for idx, abc in enumerate(coeffs_f, start=1):
        a, b, c = abc

        # FIX: positivity must hold for all p
        if not _quad_pos_ok(a, b, c, lo, hi, q_floor=float(q_floor)):
            break

        lo, hi = interval_update_quadratic_exact(abc, lo, hi, p_val=p_i)
        lo = max(float(lo), 1e-15)
        hi = max(float(hi), lo * 1.0001)
        err = float(interval_error_to_identity(lo, hi))

        if (
            target is not None
            and idx >= min_steps_i
            and err <= target
        ):
            target_met = True
            steps_used = idx
            break

    trimmed = coeffs_f[:steps_used]
    meta: Dict[str, float] = {
        "steps_used": float(steps_used),
        "target_met": 1.0 if target_met else 0.0,
        "pred_lo_final": float(lo),
        "pred_hi_final": float(hi),
        "pred_err_final": float(interval_error_to_identity(lo, hi)),
    }
    return trimmed, meta


def _phi_and_dphi_coeffs_p2(a, b, c):
    """Return polynomial coefficients for φ(y) = y·q(y)^2 and φ'(y) when p=2.

    q(y) = a + b*y + c*y^2
    φ(y) = y·(a + b*y + c*y^2)^2  — degree-5 polynomial.
    φ'(y) = d/dy φ(y) — degree-4 polynomial.

    Returns (phi_coeffs, dphi_coeffs) as arrays in *ascending* power order,
    i.e. phi_coeffs[k] is the coefficient of y^k.
    """
    # q(y) = a + b*y + c*y^2
    # q^2 = a^2 + 2ab*y + (b^2+2ac)*y^2 + 2bc*y^3 + c^2*y^4
    # phi = y * q^2 = a^2*y + 2ab*y^2 + (b^2+2ac)*y^3 + 2bc*y^4 + c^2*y^5
    phi = np.array(
        [
            0.0,
            a * a,
            2.0 * a * b,
            b * b + 2.0 * a * c,
            2.0 * b * c,
            c * c,
        ]
    )
    # d/dy: multiply coefficient k by k
    dphi = np.array([phi[k] * k for k in range(1, 6)])
    return phi, dphi


def interval_update_quadratic_exact(abc, lo, hi, p_val=2):
    """Certified interval update via critical-point extrema.

    Finds roots of φ'(y) for φ(y)=y*(a+b*y+c*y^2)^p and evaluates φ at endpoints
    + critical points. Falls back to dense grid sampling when numerical root
    solving is ill-conditioned.

    Returns (lo_new, hi_new).
    """
    a, b, c = abc
    lo, hi = float(lo), float(hi)
    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")

    try:
        # q(y)=a+b*y+c*y^2 in ascending-power basis.
        q = np.array([float(a), float(b), float(c)], dtype=np.float64)
        q_pow = np.polynomial.polynomial.polypow(q, p_i)  # ascending powers
        phi = np.concatenate([np.array([0.0], dtype=np.float64), q_pow])  # y*q(y)^p
        dphi = np.array([phi[k] * k for k in range(1, len(phi))], dtype=np.float64)
        roots = np.roots(dphi[::-1]) if dphi.size > 1 else np.array([], dtype=np.float64)

        candidates = [lo, hi]
        for r in roots:
            if np.isfinite(r.real) and abs(float(np.imag(r))) <= 1e-10:
                yr = float(np.real(r))
                if lo <= yr <= hi:
                    candidates.append(yr)

        vals = [float(np.polyval(phi[::-1], y)) for y in candidates]
        lo_new = float(min(vals))
        hi_new = float(max(vals))
        if not (math.isfinite(lo_new) and math.isfinite(hi_new)):
            raise FloatingPointError("non-finite interval extrema")
        return lo_new, hi_new
    except Exception:
        # fallback: dense grid + conservative padding
        return _interval_update_grid(abc, lo, hi, p_val=p_i)


def _interval_update_grid(abc, lo, hi, n=32768, p_val=2):
    """Grid-based interval update with conservative padding (legacy / generic p)."""
    a, b, c = abc
    ys = build_grid(lo, hi, n=n).cpu().numpy()
    q = a + b * ys + c * ys * ys
    phi = ys * (q**p_val)
    lo_new, hi_new = float(phi.min()), float(phi.max())

    if p_val % 2 == 1 and lo_new <= 0:
        lo_new = 1e-15  # force positivity if odd p gives negative phi (should be blocked by pos_ok)

    # conservative padding for non-exact case
    span = max(hi_new - lo_new, 1e-12)
    # Be more conservative for odd p to strictly ensure we don't overestimate contraction
    padding = 0.005 if p_val % 2 == 1 else 0.001
    return max(1e-15, lo_new - padding * span), hi_new + padding * span


def interval_update_affine(ab, lo, hi, n=16384, p_val=2):
    a, b = ab
    ys = build_grid(lo, hi, n=n).cpu().numpy()
    q = a + b * ys
    phi = ys * (q**p_val)
    return float(phi.min()), float(phi.max())


def interval_update_quadratic(abc, lo, hi, n=16384, p_val=2):
    """Legacy grid-based interval update (kept for backward compatibility)."""
    a, b, c = abc
    ys = build_grid(lo, hi, n=n).cpu().numpy()
    q = a + b * ys + c * ys * ys
    phi = ys * (q**p_val)
    return float(phi.min()), float(phi.max())


# ---------------------------------------------------------------------------
# Local-basis quadratic fit: q(y) = 1 - (1/p)(y-1) + α(y-1)^2
# ---------------------------------------------------------------------------


def fit_quadratic_local(
    lo,
    hi,
    tau=80.0,
    iters=300,
    restarts=8,
    seed=0,
    p_val=2,
):
    """Fit q(y) with q(1)=1 and q'(1)=-1/p baked in.

    Parameterization: q(y) = 1 - (1/p)(y-1) + α*(y-1)^2.
    Only α is free.  Returns (a, b, c) in standard basis q(y)=a+b*y+c*y^2.
    """
    ys = build_grid(lo, hi)
    rng = np.random.default_rng(seed)
    inv_p = 1.0 / p_val

    # To strictly guarantee q(y) > q_min on [lo, hi]:
    # q(y) = 1 - (1/p)(y-1) + alpha(y-1)^2 > q_min
    # => alpha > (q_min - (1 - (1/p)(y-1))) / (y-1)^2  for y != 1
    # We only need to check the endpoints lo and hi because the bounding curve
    # is monotonic on (0, 1) and (1, infinity).
    q_min = 1e-4
    alpha_lb_lo = -1e9
    if abs(lo - 1.0) > 1e-9:
        alpha_lb_lo = (q_min - (1.0 - inv_p * (lo - 1.0))) / ((lo - 1.0) ** 2)

    alpha_lb_hi = -1e9
    if abs(hi - 1.0) > 1e-9:
        alpha_lb_hi = (q_min - (1.0 - inv_p * (hi - 1.0))) / ((hi - 1.0) ** 2)

    alpha_min = max(alpha_lb_lo, alpha_lb_hi)

    best_alpha, best_val = None, None

    for r in range(restarts):
        if r == 0:
            a0 = (p_val + 1.0) / (2.0 * p_val * p_val)  # from φ''(1)=0 condition
        else:
            a0 = (p_val + 1.0) / (2.0 * p_val * p_val) + 0.3 * rng.standard_normal()

        a0 = max(a0, alpha_min + 1e-5)
        param = torch.nn.Parameter(torch.tensor([a0], dtype=torch.float64))

        opt = torch.optim.LBFGS(
            [param], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            # Project alpha to feasible region strictly
            alpha = torch.clamp(param[0], min=alpha_min + 1e-6)
            dy = ys - 1.0
            q = 1.0 - inv_p * dy + alpha * dy * dy
            err = torch.abs(1.0 - ys * (q**p_val))
            loss = smooth_max(err, tau=tau)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            alpha = float(torch.clamp(param[0], min=alpha_min + 1e-6))
            dy = ys - 1.0
            q = 1.0 - inv_p * dy + alpha * dy * dy
            err = torch.abs(1.0 - ys * (q**p_val))
            val = float(err.max())
            if best_val is None or val < best_val:
                best_val = val
                best_alpha = alpha

    # Convert local basis to standard basis: q(y) = a + b*y + c*y^2
    # q(y) = 1 - (1/p)(y-1) + α(y-1)^2
    #       = (1 + 1/p + α) + (-1/p - 2α)*y + α*y^2
    a_std = 1.0 + inv_p + best_alpha
    b_std = -inv_p - 2.0 * best_alpha
    c_std = best_alpha
    return [a_std, b_std, c_std]


def make_schedule(
    kind="affine",
    T=3,
    l0=0.05,
    u0=1.0,
    l_cushion=0.05,
    safety=1.0,
    seed=0,
    p_val=2,
    certified=True,
    q_floor: float = 1e-6,
):
    """
    FIXED version:
    - affine kind now uses the fixed-point family q_b(y)=1+b(y-1) via solve_local_affine_b_optimal
    - quad kind now uses the local-basis quadratic q_alpha(y)=1-(1/p)(y-1)+alpha(y-1)^2 via solve_local_alpha_minimax
    - positivity is enforced for all p
    - true interval is always propagated, never clipped
    """
    lo_true = float(l0)
    hi_true = max(float(u0), lo_true * 1.0001)
    sched = []

    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")

    for t in range(int(T)):
        lo_fit = max(lo_true, float(l_cushion))
        hi_fit = hi_true

        if kind == "affine":
            # FIX: solve in the fixed-point affine family, not free (a,b)
            b_star, _meta = solve_local_affine_b_optimal(
                p_val=p_i,
                lo=lo_fit,
                hi=hi_fit,
                q_floor=float(q_floor),
                seed=seed + t if "seed" in solve_local_affine_b_optimal.__code__.co_varnames else 0,  # harmless
            )
            a, b, c = affine_coeffs_from_b(b_star)

            # safety scaling: shrink b around the fixed point (keeps q(1)=1)
            if float(safety) > 1.0:
                b = b / float(safety)
                a = 1.0 - b
                c = 0.0

            # positivity check (exact)
            if affine_qmin(b, lo_true, hi_true) <= float(q_floor):
                # fall back to inverse Newton (also fixed point at 1)
                a, b, c = inverse_newton_coeffs(p_i)

            sched.append([a, b, lo_true, hi_true])
            lo2, hi2 = interval_update_affine_exact(b, lo_true, hi_true, p_val=p_i)

        else:
            # FIX: use local-basis minimax alpha schedule (fixed point + correct slope)
            alpha_star, _meta = solve_local_alpha_minimax(
                p_val=p_i,
                lo=lo_fit,
                hi=hi_fit,
                q_floor=float(q_floor),
            )
            a, b, c = local_quadratic_coeffs_from_alpha(alpha_star, p_i)

            # safety scaling: shrink (y-1) effect while keeping q(1)=1 and q'(1)=-1/p
            # We do this by scaling the quadratic "curvature" and linear deviation around 1.
            # In standard basis this is not a simple b,c divide unless you re-derive around (y-1).
            # Easiest safe option: only shrink alpha.
            if float(safety) > 1.0:
                alpha_safe = float(alpha_star) / float(safety)
                a, b, c = local_quadratic_coeffs_from_alpha(alpha_safe, p_i)

            # positivity certification (exact)
            if not certify_positivity_quadratic(a, b, c, lo_true, hi_true, q_min=float(q_floor)):
                # fall back to inverse Newton (affine, fixed point)
                a, b, c = inverse_newton_coeffs(p_i)

            sched.append([a, b, c, lo_true, hi_true])

            if certified:
                lo2, hi2 = interval_update_quadratic_exact([a, b, c], lo_true, hi_true, p_val=p_i)
            else:
                lo2, hi2 = interval_update_quadratic([a, b, c], lo_true, hi_true, p_val=p_i)

        lo_true = max(1e-15, float(lo2))
        hi_true = max(lo_true * 1.0001, float(hi2))

    return sched


if __name__ == "__main__":
    l0 = 0.05
    print("Affine schedule (PE-NS), l0=0.05")
    aff = make_schedule("affine", T=3, l0=l0, l_cushion=l0, seed=0)
    for i, row in enumerate(aff, 1):
        a, b, lo, hi = row
        print(f"t={i} a={a:.10f} b={b:.10f}  interval=[{lo:.6f},{hi:.6f}]")
    print("\nQuadratic schedule (PE2/PE4), l0=0.05")
    quad = make_schedule("quad", T=4, l0=l0, l_cushion=l0, seed=0)
    for i, row in enumerate(quad, 1):
        a, b, c, lo, hi = row
        print(f"t={i} a={a:.10f} b={b:.10f} c={c:.10f} interval=[{lo:.6f},{hi:.6f}]")
