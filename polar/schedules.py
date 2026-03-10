from __future__ import annotations

import dataclasses
from typing import List

from polar.dwh import dwh_ell_next
from polar.zolo import zolo_coeffs_from_ell, zolo_ell_next


@dataclasses.dataclass(frozen=True)
class StepSpec:
    kind: str
    ell_in: float
    ell_out: float
    pred_kappa_after: float
    r: int = 0


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

    raise ValueError(f"unknown schedule_name: {schedule_name}")


def auto_schedule_name(target_kappa_O: float) -> str:
    # For the bf16-floor target, the smallest plausible 2-step schedule is r=2 then r=2.
    if target_kappa_O >= 1.0 + 2.0**-7:
        return "zolo22"
    return "zolo32"
