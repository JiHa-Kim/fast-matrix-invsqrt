from __future__ import annotations

from typing import List

from polar.polynomial.schedules import build_polynomial_schedule
from polar.rational.dwh import dwh_ell_next
from polar.schedule_spec import StepSpec

def _dwh_step(kind: str, ell: float) -> StepSpec:
    ell_out = dwh_ell_next(ell)
    return StepSpec(
        kind=kind,
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


def build_schedule(schedule_name: str, ell0: float) -> List[StepSpec]:

    ell = float(ell0)

    poly_schedule = build_polynomial_schedule(schedule_name, ell)
    if poly_schedule is not None:
        return poly_schedule

    if schedule_name == "dwh3":
        s1 = _dwh_step("DWH", ell)
        s2 = _dwh_step("DWH", s1.ell_out)
        s3 = _dwh_step("DWH", s2.ell_out)
        return [s1, s2, s3]

    if schedule_name == "dwh3_stable_solve":
        s1 = _dwh_step("DWH_STABLE_SOLVE", ell)
        s2 = _dwh_step("DWH_STABLE_SOLVE", s1.ell_out)
        s3 = _dwh_step("DWH_STABLE_SOLVE", s2.ell_out)
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

    raise ValueError(f"unknown schedule_name: {schedule_name}")


def auto_schedule_name(target_kappa_O: float) -> str:
    _ = target_kappa_O
    return "dwh3"
