from __future__ import annotations

import dataclasses
import math
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linprog
from numpy.polynomial import Chebyshev, Polynomial

from polar.ops import gram_xtx_chunked, symmetrize
from polar.polynomial.minimax import chebyshev_clenshaw_matrix

Tensor = torch.Tensor


@dataclasses.dataclass(frozen=True)
class PolarExpressStep:
    sigma_lo: float
    sigma_hi: float
    degree_q: int
    basis: str
    anchored: bool
    interval_lo: float
    interval_hi: float
    coeffs: Tuple[float, ...]
    shifted_coeffs: Tuple[float, ...]
    shift_center: float
    shift_scale: float
    shift_gain: float
    max_step_err: float
    pred_sigma_min: float
    pred_sigma_max: float


def _scaled_horner_action(Xi: Tensor, T: Tensor, power_coeffs: tuple[float, ...], gain: float) -> Tensor:
    Z = float(power_coeffs[-1]) * Xi
    for ck in reversed(power_coeffs[:-1]):
        Z = (Z @ T) + float(ck) * Xi
    return float(gain) * Z


def _scaled_horner_matrix(T: Tensor, power_coeffs: tuple[float, ...], gain: float, out_dtype: torch.dtype) -> Tensor:
    n = T.shape[0]
    I = torch.eye(n, device=T.device, dtype=out_dtype)
    Z = float(power_coeffs[-1]) * I
    for ck in reversed(power_coeffs[:-1]):
        Z = symmetrize((Z @ T) + float(ck) * I)
    return symmetrize(float(gain) * Z)


def _recenter_power_coeffs(
    power_coeffs: np.ndarray,
    center: float,
    scale: float,
) -> np.ndarray:
    shifted = np.zeros_like(power_coeffs)
    for j, aj in enumerate(power_coeffs):
        for k in range(j + 1):
            shifted[k] += aj * math.comb(j, k) * (center ** (j - k))
    scaled = shifted.copy()
    for k in range(scaled.shape[0]):
        scaled[k] *= float(scale) ** k
    return scaled


def _choose_scaled_parameterization(
    power_coeffs: np.ndarray,
    x_lo: float,
    x_hi: float,
) -> tuple[float, float, np.ndarray]:
    mid = 0.5 * (x_lo + x_hi)
    centers = np.linspace(x_lo, x_hi, 17, dtype=np.float64)
    centers = np.unique(np.concatenate([centers, np.array([mid, 1.0], dtype=np.float64)]))
    candidates = []
    for center in centers:
        radius = max(abs(x_lo - float(center)), abs(x_hi - float(center)), 1e-30)
        coeffs = _recenter_power_coeffs(power_coeffs, float(center), radius)
        candidates.append((float(center), float(radius), coeffs))

    def _score(item: tuple[float, float, np.ndarray]) -> tuple[float, float, float]:
        center, _scale, coeffs = item
        return (
            float(np.max(np.abs(coeffs))),
            float(np.sum(np.abs(coeffs))),
            abs(center - 1.0),
        )

    return min(candidates, key=_score)


def _sigma_grid(sigma_lo: float, sigma_hi: float, num_log: int = 1024, num_lin: int = 1024) -> np.ndarray:
    sigma_lo = float(max(sigma_lo, 1e-12))
    sigma_hi = float(max(sigma_hi, sigma_lo))
    if sigma_hi <= 1.0:
        return np.logspace(np.log10(sigma_lo), np.log10(sigma_hi), num_log, dtype=np.float64)

    lo_part = np.logspace(np.log10(sigma_lo), 0.0, num_log, dtype=np.float64)
    hi_part = np.linspace(1.0, sigma_hi, num_lin, dtype=np.float64)
    return np.unique(np.concatenate([lo_part, hi_part]))


def _basis_vander(xs: np.ndarray, degree_q: int, basis: str, x_lo: float, x_hi: float) -> np.ndarray:
    if basis == "monomial":
        cols = [xs**j for j in range(degree_q + 1)]
        return np.stack(cols, axis=1)
    if basis == "chebyshev":
        mid = 0.5 * (x_lo + x_hi)
        radius = 0.5 * (x_hi - x_lo)
        ts = (xs - mid) / max(radius, 1e-30)
        return np.polynomial.chebyshev.chebvander(ts, degree_q)
    raise ValueError(f"unknown basis: {basis}")


def _eval_q_values(
    xs: np.ndarray,
    degree_q: int,
    basis: str,
    anchored: bool,
    x_lo: float,
    x_hi: float,
    coeffs: np.ndarray,
) -> np.ndarray:
    V = _basis_vander(xs, degree_q if not anchored else degree_q - 1, basis, x_lo, x_hi)
    vals = V @ coeffs
    if anchored:
        return 1.0 + (xs - 1.0) * vals
    return vals


def _basis_point_rows(
    x: float,
    degree_q: int,
    basis: str,
    x_lo: float,
    x_hi: float,
    max_deriv: int,
) -> list[np.ndarray]:
    if basis == "monomial":
        rows: list[np.ndarray] = []
        value = np.array([x**j for j in range(degree_q + 1)], dtype=np.float64)
        rows.append(value)
        if max_deriv >= 1:
            d1 = np.array([0.0] + [j * (x ** (j - 1)) for j in range(1, degree_q + 1)], dtype=np.float64)
            rows.append(d1)
        if max_deriv >= 2:
            d2 = np.array(
                [0.0, 0.0] + [j * (j - 1) * (x ** (j - 2)) for j in range(2, degree_q + 1)],
                dtype=np.float64,
            )
            rows.append(d2)
        return rows
    if basis == "chebyshev":
        rows = []
        for deriv in range(max_deriv + 1):
            row = np.empty((degree_q + 1,), dtype=np.float64)
            for j in range(degree_q + 1):
                poly = Chebyshev.basis(j, domain=[x_lo, x_hi], window=[-1.0, 1.0])
                row[j] = float(poly.deriv(deriv)(x))
            rows.append(row)
        return rows
    raise ValueError(f"unknown basis: {basis}")


def scalar_map_bounds(
    coeffs: PolarExpressStep,
    sigma_lo: float,
    sigma_hi: float,
    num_log: int = 4096,
    num_lin: int = 4096,
) -> tuple[float, float]:
    grid = _sigma_grid(sigma_lo, sigma_hi, num_log=num_log, num_lin=num_lin)
    xs = grid * grid
    sigma_out = grid * _eval_q_values(
        xs,
        coeffs.degree_q,
        coeffs.basis,
        coeffs.anchored,
        coeffs.interval_lo,
        coeffs.interval_hi,
        np.array(coeffs.coeffs, dtype=np.float64),
    )
    return float(np.min(sigma_out)), float(np.max(sigma_out))


def polar_express_step(
    sigma_lo: float,
    sigma_hi: float,
    degree_q: int,
    basis: str,
    cushion_ratio: float = 0.0,
    robust_pad: float = 1.0,
    l1_reg: float = 0.0,
    upper_cap: float | None = None,
    anchor_q1: bool = False,
    anchor_p1_prime: bool = False,
    anchor_p1_second: bool = False,
    anchored: bool = False,
) -> PolarExpressStep:
    sigma_lo = float(max(sigma_lo, 1e-8))
    sigma_hi = float(max(sigma_hi, sigma_lo))
    pad = float(max(robust_pad, 1.0))
    solve_lo = max(sigma_lo, sigma_hi * float(cushion_ratio))
    solve_lo = max(solve_lo / pad, 1e-8)
    solve_hi = sigma_hi * pad

    grid = _sigma_grid(solve_lo, solve_hi)
    xs = grid * grid
    x_lo = float(solve_lo * solve_lo)
    x_hi = float(solve_hi * solve_hi)
    basis_degree = degree_q if not anchored else degree_q - 1
    if basis_degree < 0:
        raise ValueError("anchored polynomial requires degree_q >= 1")
    V = _basis_vander(xs, basis_degree, basis, x_lo, x_hi)
    P = grid[:, None] * ((xs - 1.0)[:, None] * V if anchored else V)
    if anchored:
        base = grid
    else:
        base = np.zeros_like(grid)

    ncoef = basis_degree + 1
    nvars = 2 * ncoef + 1
    c = np.zeros((nvars,), dtype=np.float64)
    c[ncoef] = 1.0
    c[ncoef + 1 :] = float(max(l1_reg, 0.0))

    A_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    A_eq_rows: list[np.ndarray] = []
    b_eq_rows: list[float] = []

    # Minimize worst-case |1 - p(sigma)|.
    for row in P:
        A_rows.append(np.concatenate([row, [-1.0], np.zeros((ncoef,), dtype=np.float64)]))
        b_rows.append(1.0)
        A_rows.append(np.concatenate([-row, [-1.0], np.zeros((ncoef,), dtype=np.float64)]))
        b_rows.append(-1.0)

    # Constrain the map to move singular values toward 1 without divergence.
    cap = float(upper_cap) if upper_cap is not None else float(max(1.25, sigma_hi * 1.05))
    for sigma, row, base_row in zip(grid, P, base):
        if sigma <= 1.0:
            A_rows.append(np.concatenate([-row, [0.0], np.zeros((ncoef,), dtype=np.float64)]))
            b_rows.append(-(sigma - base_row))
        else:
            A_rows.append(np.concatenate([row, [0.0], np.zeros((ncoef,), dtype=np.float64)]))
            b_rows.append(sigma - base_row)
        A_rows.append(np.concatenate([row, [0.0], np.zeros((ncoef,), dtype=np.float64)]))
        b_rows.append(cap - base_row)
        A_rows.append(np.concatenate([-row, [0.0], np.zeros((ncoef,), dtype=np.float64)]))
        b_rows.append(base_row - 1e-6)

    for j in range(ncoef):
        ej = np.zeros((ncoef,), dtype=np.float64)
        ej[j] = 1.0
        uj = np.zeros((ncoef,), dtype=np.float64)
        uj[j] = -1.0
        A_rows.append(np.concatenate([ej, [0.0], uj]))
        b_rows.append(0.0)
        A_rows.append(np.concatenate([-ej, [0.0], uj]))
        b_rows.append(0.0)

    if anchor_q1 or anchor_p1_prime or anchor_p1_second:
        rows_1 = _basis_point_rows(1.0, basis_degree, basis, x_lo, x_hi, max_deriv=2)
        if anchor_q1:
            if anchored:
                pass
            else:
                A_eq_rows.append(np.concatenate([rows_1[0], [0.0], np.zeros((ncoef,), dtype=np.float64)]))
                b_eq_rows.append(1.0)
        if anchor_p1_prime:
            # p(sigma) = sigma q(sigma^2), so p'(1) = q(1) + 2 q'(1).
            if anchored:
                # q'(1) = r(1)
                A_eq_rows.append(np.concatenate([2.0 * rows_1[0], [0.0], np.zeros((ncoef,), dtype=np.float64)]))
                b_eq_rows.append(-1.0)
            else:
                A_eq_rows.append(np.concatenate([2.0 * rows_1[1], [0.0], np.zeros((ncoef,), dtype=np.float64)]))
                b_eq_rows.append(-1.0)
        if anchor_p1_second:
            # p''(1) = 6 q'(1) + 4 q''(1).
            if anchored:
                # q''(1)=2 r'(1)
                A_eq_rows.append(
                    np.concatenate([(6.0 * rows_1[0]) + (8.0 * rows_1[1]), [0.0], np.zeros((ncoef,), dtype=np.float64)])
                )
                b_eq_rows.append(0.0)
            else:
                A_eq_rows.append(
                    np.concatenate([(6.0 * rows_1[1]) + (4.0 * rows_1[2]), [0.0], np.zeros((ncoef,), dtype=np.float64)])
                )
                b_eq_rows.append(0.0)

    A_ub = np.stack(A_rows, axis=0)
    b_ub = np.array(b_rows, dtype=np.float64)
    A_eq = np.stack(A_eq_rows, axis=0) if A_eq_rows else None
    b_eq = np.array(b_eq_rows, dtype=np.float64) if b_eq_rows else None
    bounds = [(None, None)] * ncoef + [(0.0, None)] + [(0.0, None)] * ncoef
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"restricted polynomial solve failed: {res.message}")

    coeffs = np.array(res.x[:ncoef], dtype=np.float64)
    tmp = PolarExpressStep(
        sigma_lo=sigma_lo,
        sigma_hi=sigma_hi,
        degree_q=int(degree_q),
        basis=basis,
        anchored=anchored,
        interval_lo=x_lo,
        interval_hi=x_hi,
        coeffs=tuple(float(v) for v in coeffs),
        shifted_coeffs=(),
        shift_center=0.0,
        shift_scale=1.0,
        shift_gain=1.0,
        max_step_err=0.0,
        pred_sigma_min=0.0,
        pred_sigma_max=0.0,
    )
    sigma_min, sigma_max = scalar_map_bounds(tmp, sigma_lo, sigma_hi)
    eval_grid = _sigma_grid(sigma_lo, sigma_hi, num_log=4096, num_lin=4096)
    eval_xs = eval_grid * eval_grid
    q_eval = _eval_q_values(eval_xs, degree_q, basis, anchored, x_lo, x_hi, coeffs)
    sigma_out = eval_grid * q_eval
    max_step_err = float(np.max(np.abs(1.0 - sigma_out)))
    if basis == "chebyshev":
        cheb = Chebyshev(coeffs, domain=[x_lo, x_hi], window=[-1.0, 1.0])
        power = cheb.convert(kind=Polynomial)
        power_coeffs = np.array(power.coef, dtype=np.float64)
    else:
        power_coeffs = np.array(coeffs, dtype=np.float64)
    if anchored:
        anchored_power = np.zeros((power_coeffs.shape[0] + 1,), dtype=np.float64)
        anchored_power[0] = 1.0 - power_coeffs[0]
        anchored_power[1:] += power_coeffs
        anchored_power[:-1] -= power_coeffs
        power_coeffs = anchored_power
    shift_center = 0.5 * (x_lo + x_hi)
    shift_scale = 1.0
    shifted = _recenter_power_coeffs(power_coeffs, shift_center, shift_scale)
    if not anchored and degree_q <= 3:
        shift_center, shift_scale, shifted = _choose_scaled_parameterization(power_coeffs, x_lo, x_hi)
    shift_gain = float(max(np.max(np.abs(shifted)), 1e-30))
    shifted = shifted / shift_gain
    return PolarExpressStep(
        sigma_lo=sigma_lo,
        sigma_hi=sigma_hi,
        degree_q=int(degree_q),
        basis=basis,
        anchored=anchored,
        interval_lo=x_lo,
        interval_hi=x_hi,
        coeffs=tuple(float(v) for v in coeffs),
        shifted_coeffs=tuple(float(v) for v in shifted),
        shift_center=float(shift_center),
        shift_scale=float(shift_scale),
        shift_gain=float(shift_gain),
        max_step_err=max_step_err,
        pred_sigma_min=float(sigma_min),
        pred_sigma_max=float(sigma_max),
    )


@torch.no_grad()
def polar_express_step_matrix_only(
    S: Tensor,
    coeffs: PolarExpressStep,
    matmul_dtype: torch.dtype,
) -> tuple[Tensor, float]:
    S_work = symmetrize(S.to(dtype=matmul_dtype))
    if coeffs.basis == "monomial":
        n = S.shape[0]
        I = torch.eye(n, device=S.device, dtype=matmul_dtype)
        V = symmetrize((S_work - float(coeffs.shift_center) * I) / max(float(coeffs.shift_scale), 1e-30))
        Q = _scaled_horner_matrix(V, coeffs.shifted_coeffs, coeffs.shift_gain, matmul_dtype)
    elif coeffs.basis == "chebyshev":
        n = S.shape[0]
        I = torch.eye(n, device=S.device, dtype=matmul_dtype)
        if coeffs.anchored:
            R = chebyshev_clenshaw_matrix(
                S_work,
                coeffs.coeffs,
                interval_lo=coeffs.interval_lo,
                interval_hi=coeffs.interval_hi,
                out_dtype=matmul_dtype,
            )
            Q = symmetrize(I + (S_work - I) @ R)
        elif coeffs.degree_q <= 3:
            V = symmetrize((S_work - float(coeffs.shift_center) * I) / max(float(coeffs.shift_scale), 1e-30))
            Q = _scaled_horner_matrix(V, coeffs.shifted_coeffs, coeffs.shift_gain, matmul_dtype)
        else:
            Q = chebyshev_clenshaw_matrix(
                S_work,
                coeffs.coeffs,
                interval_lo=coeffs.interval_lo,
                interval_hi=coeffs.interval_hi,
                out_dtype=matmul_dtype,
            )
    else:
        raise ValueError(f"unknown basis: {coeffs.basis}")

    if not torch.isfinite(Q).all():
        raise RuntimeError("non-finite restricted polynomial step")
    return Q, 0.0


@torch.no_grad()
def polar_express_action_chunked(
    X: Tensor,
    S: Tensor,
    coeffs: PolarExpressStep,
    rhs_chunk_rows: int,
    out_dtype: torch.dtype,
) -> tuple[Tensor, float]:
    S_work = symmetrize(S.to(dtype=out_dtype))
    m, n = X.shape
    Y = torch.empty((m, n), device=X.device, dtype=out_dtype)

    if coeffs.basis == "monomial":
        I = torch.eye(n, device=S.device, dtype=out_dtype)
        V = symmetrize((S_work - float(coeffs.shift_center) * I) / max(float(coeffs.shift_scale), 1e-30))
        for i in range(0, m, rhs_chunk_rows):
            Xi = X[i : i + rhs_chunk_rows].to(dtype=out_dtype)
            Y[i : i + rhs_chunk_rows] = _scaled_horner_action(Xi, V, coeffs.shifted_coeffs, coeffs.shift_gain)
    elif coeffs.basis == "chebyshev":
        I = torch.eye(n, device=S.device, dtype=out_dtype)
        T = None
        V = None
        if coeffs.anchored or coeffs.degree_q > 3:
            mid = 0.5 * (coeffs.interval_lo + coeffs.interval_hi)
            radius = 0.5 * (coeffs.interval_hi - coeffs.interval_lo)
            T = symmetrize((S_work - float(mid) * I) / max(float(radius), 1e-30))
        else:
            V = symmetrize((S_work - float(coeffs.shift_center) * I) / max(float(coeffs.shift_scale), 1e-30))
        for i in range(0, m, rhs_chunk_rows):
            Xi = X[i : i + rhs_chunk_rows].to(dtype=out_dtype)
            if coeffs.anchored:
                rhs = Xi @ symmetrize(S_work - I)
                b_kplus1 = torch.zeros_like(rhs)
                b_kplus2 = torch.zeros_like(rhs)
                for ck in reversed(coeffs.coeffs[1:]):
                    b_k = 2.0 * (b_kplus1 @ T) - b_kplus2 + float(ck) * rhs
                    b_kplus2 = b_kplus1
                    b_kplus1 = b_k
                Ri = (b_kplus1 @ T) - b_kplus2 + float(coeffs.coeffs[0]) * rhs
                Y[i : i + rhs_chunk_rows] = Xi + Ri
            elif coeffs.degree_q <= 3:
                Y[i : i + rhs_chunk_rows] = _scaled_horner_action(Xi, V, coeffs.shifted_coeffs, coeffs.shift_gain)
            else:
                b_kplus1 = torch.zeros_like(Xi)
                b_kplus2 = torch.zeros_like(Xi)
                for ck in reversed(coeffs.coeffs[1:]):
                    b_k = 2.0 * (b_kplus1 @ T) - b_kplus2 + float(ck) * Xi
                    b_kplus2 = b_kplus1
                    b_kplus1 = b_k
                Y[i : i + rhs_chunk_rows] = (b_kplus1 @ T) - b_kplus2 + float(coeffs.coeffs[0]) * Xi
    else:
        raise ValueError(f"unknown basis: {coeffs.basis}")

    if not torch.isfinite(Y).all():
        raise RuntimeError("non-finite restricted polynomial action")
    return Y, 0.0


@torch.no_grad()
def polar_express_fro_scale(
    X: Tensor,
    eps: float = 1e-12,
) -> tuple[Tensor, float]:
    fro = torch.linalg.matrix_norm(X.float(), ord="fro").clamp_min(float(eps))
    X_scaled = X / fro.to(dtype=X.dtype)
    return X_scaled, float(fro.item())


@torch.no_grad()
def polar_express_aol_scale(
    X: Tensor,
    gram_chunk_rows: int,
    accum_dtype: torch.dtype,
    eps: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    S = gram_xtx_chunked(X, gram_chunk_rows, accum_dtype)
    s = torch.rsqrt(S.abs().sum(dim=-1).clamp_min(float(eps)))
    X_scaled = X * s.unsqueeze(0).to(dtype=X.dtype)
    return X_scaled, s
