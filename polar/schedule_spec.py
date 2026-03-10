from __future__ import annotations

import dataclasses


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
    pe_shift_scale: float = 1.0
    pe_shift_gain: float = 1.0
    paper_coeffs: tuple[float, ...] = ()
