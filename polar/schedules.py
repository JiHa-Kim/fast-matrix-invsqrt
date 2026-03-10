from __future__ import annotations

import dataclasses
from typing import List

from polar.polynomial.express import polar_express_step, scalar_map_bounds
from polar.polynomial.minimax import poly_inv_sqrt_coeffs_from_ell
from polar.rational.dwh import dwh_ell_next
from polar.rational.zolo import zolo_coeffs_from_ell, zolo_ell_next


@dataclasses.dataclass(frozen=True)
class StepSpec:
    kind: str
    ell_in: float
    ell_out: float
    pred_kappa_after: float
    r: int = 0
    degree: int = 0
    u_in: float = 1.0
    u_out: float = 1.0
    pe_degree: int = 0
    basis: str = ""
    pe_anchored: bool = False
    pe_coeffs: tuple[float, ...] = ()
    pe_shifted_coeffs: tuple[float, ...] = ()
    pe_interval_lo: float = 0.0
    pe_interval_hi: float = 0.0
    pe_shift_center: float = 0.0


def _zolo_step(ell: float, r: int, zolo_coeff_dps: int) -> StepSpec:
    coeffs = zolo_coeffs_from_ell(r, ell, dps=zolo_coeff_dps)
    ell_out = zolo_ell_next(ell, coeffs)
    return StepSpec(
        kind="ZOLO",
        ell_in=float(ell),
        ell_out=float(ell_out),
        pred_kappa_after=float(1.0 / max(ell_out, 1e-300)),
        r=int(r),
    )


def _dwh_step(ell: float) -> StepSpec:
    ell_out = dwh_ell_next(ell)
    return StepSpec(
        kind="DWH",
        ell_in=float(ell),
        ell_out=float(ell_out),
        pred_kappa_after=float(1.0 / max(ell_out, 1e-300)),
        r=1,
    )


def _dwh_stable_solve_step(ell: float) -> StepSpec:
    ell_out = dwh_ell_next(ell)
    return StepSpec(
        kind="DWH_STABLE_SOLVE",
        ell_in=float(ell),
        ell_out=float(ell_out),
        pred_kappa_after=float(1.0 / max(ell_out, 1e-300)),
        r=1,
    )


def _dwh_tuned_fp32_step(ell: float) -> StepSpec:
    from polar.rational.dwh_tuned_fp32 import get_tuned_dwh_coeffs_fp32
    a, b, c = get_tuned_dwh_coeffs_fp32(ell)
    ell_out = ell * (a + b * ell * ell) / (1.0 + c * ell * ell)
    return StepSpec(
        kind="DWH_TUNED_FP32",
        ell_in=float(ell),
        ell_out=float(ell_out),
        pred_kappa_after=float(1.0 / max(ell_out, 1e-300)),
        r=1,
    )


def _poly_step(ell: float, degree: int) -> StepSpec:
    coeffs = poly_inv_sqrt_coeffs_from_ell(degree, ell)
    sigma_min = max(coeffs.pred_sigma_min, 1e-300)
    sigma_max = max(coeffs.pred_sigma_max, sigma_min)
    ell_out = sigma_min / sigma_max
    return StepSpec(
        kind="POLY",
        ell_in=float(ell),
        ell_out=float(ell_out),
        pred_kappa_after=float(sigma_max / sigma_min),
        degree=int(degree),
    )


def _pe_step_from_interval(
    sigma_lo: float,
    sigma_hi: float,
    basis: str,
    pe_degree: int,
    next_degree: int | None = None,
) -> StepSpec:
    def _stress_interval(lo: float, hi: float, degree: int) -> tuple[float, float]:
        if degree <= 2:
            return max(1e-8, lo), hi * 1.01
        return max(1e-8, lo / 1.02), hi * 1.05

    if basis == "monomial":
        candidates = [
            dict(robust_pad=1.0, l1_reg=0.0, upper_cap=max(1.25, sigma_hi * 1.05)),
            dict(robust_pad=1.0, l1_reg=1e-4, upper_cap=max(1.20, sigma_hi * 1.03)),
        ]
    elif pe_degree == 2:
        candidates = [
            dict(robust_pad=1.0, l1_reg=0.0, upper_cap=max(1.25, sigma_hi * 1.05)),
            dict(robust_pad=1.0, l1_reg=1e-4, upper_cap=max(1.20, sigma_hi * 1.03)),
        ]
    else:
        candidates = [
            dict(robust_pad=1.0, l1_reg=0.0, upper_cap=max(1.20, sigma_hi * 1.03)),
            dict(robust_pad=1.05, l1_reg=1e-4, upper_cap=max(1.15, sigma_hi * 1.02)),
            dict(robust_pad=1.10, l1_reg=5e-4, upper_cap=max(1.12, sigma_hi * 1.01)),
        ]

    best = None
    for cfg in candidates:
        coeffs_try = polar_express_step(
            sigma_lo,
            sigma_hi,
            degree_q=pe_degree,
            basis=basis,
            anchored=False,
            **cfg,
        )
        stress_lo, stress_hi = _stress_interval(sigma_lo, sigma_hi, pe_degree)
        stress_min, stress_max = scalar_map_bounds(coeffs_try, stress_lo, stress_hi)
        if stress_min <= 0.0 or not (stress_max > 0.0):
            continue
        # Favor tighter upper-end control; a bloated certified u_out is what
        # tends to poison the quadratic suffix in bf16.
        score = (stress_min / max(stress_max, 1e-300)) / max(stress_max, 1e-300)
        if next_degree is not None:
            next_step = _pe_step_from_interval(stress_min, stress_max, basis, next_degree, next_degree=None)
            # Optimize the handoff first, then prefer the tighter immediate interval.
            score = (
                (next_step.ell_out / max(next_step.u_out, 1e-300)) / max(next_step.u_out, 1e-300)
                + 0.05 * score
            )
        if best is None or score > best[0]:
            best = (score, coeffs_try, stress_min, stress_max)
    if best is None:
        raise RuntimeError(
            f"no safeguarded PE candidate for sigma interval [{sigma_lo:.3e}, {sigma_hi:.3e}] "
            f"(basis={basis}, qdeg={pe_degree})"
        )
    coeffs = best[1]
    sigma_min = max(best[2], 1e-300)
    sigma_max = max(best[3], sigma_min)
    return StepSpec(
        kind="PE",
        ell_in=float(sigma_lo),
        ell_out=float(sigma_min),
        pred_kappa_after=float(sigma_max / sigma_min),
        u_in=float(sigma_hi),
        u_out=float(sigma_max),
        pe_degree=int(pe_degree),
        basis=basis,
        pe_anchored=bool(coeffs.anchored),
        pe_coeffs=tuple(coeffs.coeffs),
        pe_shifted_coeffs=tuple(coeffs.shifted_coeffs),
        pe_interval_lo=float(coeffs.interval_lo),
        pe_interval_hi=float(coeffs.interval_hi),
        pe_shift_center=float(coeffs.shift_center),
    )


def _build_pe_schedule(ell: float, basis: str, degree_pattern: tuple[int, ...], max_steps: int = 12) -> List[StepSpec]:
    sigma_lo = max(float(ell), 1e-4)
    sigma_hi = 1.0
    out: List[StepSpec] = []
    for i in range(max_steps):
        pe_degree = degree_pattern[min(i, len(degree_pattern) - 1)]
        next_degree = None
        if i + 1 < max_steps:
            next_degree = degree_pattern[min(i + 1, len(degree_pattern) - 1)]
            if next_degree == pe_degree:
                next_degree = None
        step = _pe_step_from_interval(sigma_lo, sigma_hi, basis, pe_degree, next_degree=next_degree)
        out.append(step)
        sigma_lo = step.ell_out
        sigma_hi = step.u_out
        if sigma_lo >= 0.99 and sigma_hi <= 1.01:
            break
    return out


def build_schedule(schedule_name: str, ell0: float, zolo_coeff_dps: int) -> List[StepSpec]:
    ell = float(ell0)

    if schedule_name == "zolo22":
        s1 = _zolo_step(ell, 2, zolo_coeff_dps)
        s2 = _zolo_step(s1.ell_out, 2, zolo_coeff_dps)
        return [s1, s2]

    if schedule_name == "zolo23":
        s1 = _zolo_step(ell, 2, zolo_coeff_dps)
        s2 = _zolo_step(s1.ell_out, 3, zolo_coeff_dps)
        return [s1, s2]

    if schedule_name == "zolo32":
        s1 = _zolo_step(ell, 3, zolo_coeff_dps)
        s2 = _zolo_step(s1.ell_out, 2, zolo_coeff_dps)
        return [s1, s2]

    if schedule_name == "dwh3":
        s1 = _dwh_step(ell)
        s2 = _dwh_step(s1.ell_out)
        s3 = _dwh_step(s2.ell_out)
        return [s1, s2, s3]

    if schedule_name == "dwh3_stable_solve":
        s1 = _dwh_stable_solve_step(ell)
        s2 = _dwh_stable_solve_step(s1.ell_out)
        s3 = _dwh_stable_solve_step(s2.ell_out)
        return [s1, s2, s3]

    if schedule_name == "dwh_tuned_fp32":
        curr_ell = ell
        out: List[StepSpec] = []
        for _ in range(16):
            step = _dwh_tuned_fp32_step(curr_ell)
            out.append(step)
            curr_ell = step.ell_out
            if curr_ell >= 0.999:
                break
        return out

    if schedule_name == "poly16x2":
        s1 = _poly_step(ell, 16)
        s2 = _poly_step(s1.ell_out, 16)
        return [s1, s2]

    if schedule_name == "poly24x2":
        s1 = _poly_step(ell, 24)
        s2 = _poly_step(s1.ell_out, 24)
        return [s1, s2]

    if schedule_name == "pe2mono12":
        return _build_pe_schedule(ell, "monomial", (2,))

    if schedule_name == "pe2cheb12":
        return _build_pe_schedule(ell, "chebyshev", (2,))

    if schedule_name == "pe3cheb12":
        return _build_pe_schedule(ell, "chebyshev", (3,))

    if schedule_name == "pe32hyb12":
        return _build_pe_schedule(ell, "chebyshev", (3, 2))

    raise ValueError(f"unknown schedule_name: {schedule_name}")


def auto_schedule_name(target_kappa_O: float) -> str:
    # For the bf16-floor target, the smallest plausible 2-step schedule is r=2 then r=2.
    if target_kappa_O >= 1.0 + 2.0**-7:
        return "zolo22"
    return "zolo32"
