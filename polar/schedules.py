from __future__ import annotations

import dataclasses
from typing import List

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

    raise ValueError(f"unknown schedule_name: {schedule_name}")


def auto_schedule_name(target_kappa_O: float) -> str:
    # For the bf16-floor target, the smallest plausible 2-step schedule is r=2 then r=2.
    if target_kappa_O >= 1.0 + 2.0**-7:
        return "zolo22"
    return "zolo32"
