from __future__ import annotations

from typing import List

from polar.polynomial.express import polar_express_step, scalar_map_bounds
from polar.polynomial.minimax import poly_inv_sqrt_coeffs_from_ell
from polar.schedule_spec import StepSpec


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
        candidates = _quadratic_candidate_configs(sigma_hi)
    else:
        candidates = [
            dict(robust_pad=1.0, l1_reg=0.0, upper_cap=max(1.20, sigma_hi * 1.03)),
            dict(robust_pad=1.05, l1_reg=1e-4, upper_cap=max(1.15, sigma_hi * 1.02)),
            dict(robust_pad=1.10, l1_reg=5e-4, upper_cap=max(1.12, sigma_hi * 1.01)),
        ]

    best = None
    for cfg in candidates:
        cfg_use = dict(cfg)
        coeffs_try = polar_express_step(
            sigma_lo,
            sigma_hi,
            degree_q=pe_degree,
            basis=basis,
            anchored=bool(cfg_use.pop("anchored", False)),
            **cfg_use,
        )
        stress_lo, stress_hi = _stress_interval(sigma_lo, sigma_hi, pe_degree)
        stress_min, stress_max = scalar_map_bounds(coeffs_try, stress_lo, stress_hi)
        if stress_min <= 0.0 or not (stress_max > 0.0):
            continue
        score = (stress_min / max(stress_max, 1e-300)) / max(stress_max, 1e-300)
        if next_degree is not None:
            next_step = _pe_step_from_interval(stress_min, stress_max, basis, next_degree, next_degree=None)
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
        pe_shift_scale=float(coeffs.shift_scale),
        pe_shift_gain=float(coeffs.shift_gain),
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


def _quadratic_candidate_configs(sigma_hi: float) -> list[dict[str, float | bool]]:
    return [
        dict(anchored=True, robust_pad=1.0, l1_reg=1e-6, upper_cap=max(1.16, sigma_hi * 1.02)),
        dict(anchored=False, anchor_q1=True, robust_pad=1.0, l1_reg=1e-6, upper_cap=max(1.18, sigma_hi * 1.025)),
        dict(anchored=True, robust_pad=1.01, l1_reg=1e-5, upper_cap=max(1.14, sigma_hi * 1.015)),
        dict(anchored=False, anchor_q1=True, robust_pad=1.01, l1_reg=1e-5, upper_cap=max(1.16, sigma_hi * 1.02)),
        dict(anchored=False, robust_pad=1.0, l1_reg=1e-4, upper_cap=max(1.20, sigma_hi * 1.03)),
    ]


def _pe_quadratic_candidates(sigma_lo: float, sigma_hi: float, basis: str) -> List[StepSpec]:
    stress_lo, stress_hi = max(1e-8, sigma_lo), sigma_hi * 1.01
    out: List[tuple[tuple[float, float, float], StepSpec]] = []
    for cfg in _quadratic_candidate_configs(sigma_hi):
        cfg_use = dict(cfg)
        coeffs_try = polar_express_step(
            sigma_lo,
            sigma_hi,
            degree_q=2,
            basis=basis,
            anchored=bool(cfg_use.pop("anchored", False)),
            **cfg_use,
        )
        stress_min, stress_max = scalar_map_bounds(coeffs_try, stress_lo, stress_hi)
        if stress_min <= 0.0 or not (stress_max > 0.0):
            continue
        step = StepSpec(
            kind="PE",
            ell_in=float(sigma_lo),
            ell_out=float(max(stress_min, 1e-300)),
            pred_kappa_after=float(max(stress_max, stress_min) / max(stress_min, 1e-300)),
            u_in=float(sigma_hi),
            u_out=float(max(stress_max, stress_min)),
            pe_degree=2,
            basis=basis,
            pe_anchored=bool(coeffs_try.anchored),
            pe_coeffs=tuple(coeffs_try.coeffs),
            pe_shifted_coeffs=tuple(coeffs_try.shifted_coeffs),
            pe_interval_lo=float(coeffs_try.interval_lo),
            pe_interval_hi=float(coeffs_try.interval_hi),
            pe_shift_center=float(coeffs_try.shift_center),
            pe_shift_scale=float(coeffs_try.shift_scale),
            pe_shift_gain=float(coeffs_try.shift_gain),
        )
        key = (
            -float(max(step.u_out - 1.0, 0.0)),
            float(step.ell_out / max(step.u_out, 1e-300)),
            -abs(float(step.pe_shift_center) - 1.0),
        )
        out.append((key, step))
    out.sort(key=lambda item: item[0], reverse=True)
    return [step for _, step in out]


def _build_pe_quadratic_schedule_joint(ell: float, basis: str, total_steps: int = 8, beam_width: int = 8) -> List[StepSpec]:
    beams: List[tuple[List[StepSpec], tuple[float, float]]] = [([], (max(float(ell), 1e-4), 1.0))]
    for _ in range(total_steps):
        expanded: List[tuple[tuple[float, float, float], List[StepSpec], tuple[float, float]]] = []
        for path, (sigma_lo, sigma_hi) in beams:
            for cand in _pe_quadratic_candidates(sigma_lo, sigma_hi, basis)[:5]:
                path_next = path + [cand]
                state_next = (cand.ell_out, cand.u_out)
                key = (
                    float(cand.ell_out / max(cand.u_out, 1e-300)),
                    -float(max(cand.u_out - 1.0, 0.0)),
                    -float(cand.pred_kappa_after),
                )
                expanded.append((key, path_next, state_next))
        expanded.sort(key=lambda item: item[0], reverse=True)
        beams = [(path, state) for _, path, state in expanded[:beam_width]]

    target = 1.0 + 2.0**-6
    best_path, _ = max(
        beams,
        key=lambda item: (
            float(item[0][-1].pred_kappa_after <= target),
            -float(item[0][-1].pred_kappa_after),
            -float(max(item[0][-1].u_out - 1.0, 0.0)),
        ),
    )
    return best_path


def _select_best_pe_hybrid(ell: float, basis: str) -> List[StepSpec]:
    target_kappa = 1.0 + 2.0**-6
    candidate_patterns = [
        (3, 2),
        (3, 3, 2),
        (3, 2, 2),
        (3, 3, 2, 2),
        (3, 3, 3, 2),
    ]

    best_schedule: List[StepSpec] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for pattern in candidate_patterns:
        schedule = _build_pe_schedule(ell, basis, pattern)
        final = schedule[-1]
        reaches_target = float(final.pred_kappa_after <= target_kappa)
        work = float(sum(st.pe_degree for st in schedule if st.kind == "PE"))
        key = (reaches_target, -work, -float(len(schedule)), -float(final.pred_kappa_after))
        if best_key is None or key > best_key:
            best_key = key
            best_schedule = schedule
    assert best_schedule is not None
    return best_schedule


def build_polynomial_schedule(schedule_name: str, ell: float) -> List[StepSpec] | None:
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
        joint = _build_pe_quadratic_schedule_joint(ell, "chebyshev", total_steps=8)
        if joint[-1].pred_kappa_after <= 1.0 + 2.0**-6:
            return joint
        return _build_pe_schedule(ell, "chebyshev", (2,))

    if schedule_name == "pe3cheb12":
        return _build_pe_schedule(ell, "chebyshev", (3,))

    if schedule_name == "pe32hyb12":
        return _select_best_pe_hybrid(ell, "chebyshev")

    return None
