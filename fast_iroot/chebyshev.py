from dataclasses import dataclass
from typing import Optional, Tuple, Callable

import torch
import numpy as np
import functools

from .utils import _matmul_into, _check_square


@dataclass
class ChebyshevApplyWorkspace:
    """Workspace for Chebyshev Clenshaw recurrence.
    Note: T_curr, T_prev, T_next are Clenshaw b-recurrence buffers, not Chebyshev T polynomials.
    """

    T_curr: torch.Tensor
    T_prev: torch.Tensor
    T_next: torch.Tensor
    Z: torch.Tensor
    tmp: torch.Tensor


def _alloc_ws_chebyshev(B: torch.Tensor) -> ChebyshevApplyWorkspace:
    shape = B.shape
    return ChebyshevApplyWorkspace(
        T_curr=B.new_empty(shape),
        T_prev=B.new_empty(shape),
        T_next=B.new_empty(shape),
        Z=B.new_empty(shape),
        tmp=B.new_empty(shape),
    )


def _ws_ok_chebyshev(ws: Optional[ChebyshevApplyWorkspace], B: torch.Tensor) -> bool:
    if ws is None:
        return False

    def _ok(t: torch.Tensor) -> bool:
        return t.device == B.device and t.dtype == B.dtype and t.shape == B.shape

    return (
        _ok(ws.T_curr)
        and _ok(ws.T_prev)
        and _ok(ws.T_next)
        and _ok(ws.Z)
        and _ok(ws.tmp)
    )


@functools.lru_cache(maxsize=128)
def compute_chebyshev_coeffs_cached(
    p_val: int,
    degree: int,
    l_min: float,
    l_max: float = 1.0,
) -> Tuple[float, ...]:
    """
    Cached version of Chebyshev coefficients for f(x) = x^{-1/p}.
    Returns a tuple of floats instead of a numpy array so it is hashable.
    """

    if l_min <= 0:
        raise ValueError("l_min must be > 0 for x^{-1/p}")

    def f_inv_root(x: np.ndarray) -> np.ndarray:
        return np.power(x, -1.0 / p_val)

    coeffs = compute_chebyshev_coeffs(f_inv_root, degree, l_min, l_max)
    return tuple(coeffs.tolist())


def compute_chebyshev_coeffs(
    func: Callable[[np.ndarray], np.ndarray],
    degree: int,
    l_min: float,
    l_max: float = 1.0,
) -> np.ndarray:
    """
    Compute Chebyshev coefficients for func(x) over [l_min, l_max].
    Uses discrete orthogonality matching numpy.polynomial.chebyshev.chebfit.
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    if l_min >= l_max:
        raise ValueError(f"l_min ({l_min}) must be strictly less than l_max ({l_max})")

    # Chebyshev nodes in [-1, 1]
    k = np.arange(degree + 1)
    nodes_std = np.cos((2 * k + 1) * np.pi / (2 * (degree + 1)))

    # Map nodes to [l_min, l_max]
    nodes_mapped = 0.5 * (l_max - l_min) * nodes_std + 0.5 * (l_max + l_min)

    # Evaluate function
    y = func(nodes_mapped)

    # Compute coefficients using discrete orthogonality
    coeffs = np.zeros(degree + 1)
    for j in range(degree + 1):
        T_j = np.cos(j * np.arccos(nodes_std))
        c_j = (2.0 / (degree + 1)) * np.sum(y * T_j)
        coeffs[j] = c_j

    coeffs[0] /= 2.0  # standard Chebyshev definition
    return coeffs


def _map_to_chebyshev_domain(x: np.ndarray, l_min: float, l_max: float) -> np.ndarray:
    return (2.0 * x - (l_max + l_min)) / (l_max - l_min)


@functools.lru_cache(maxsize=128)
def compute_inverse_proot_minimax_coeffs_cached(
    p_val: int,
    degree: int,
    l_min: float,
    l_max: float = 1.0,
    grid_n: int = 2049,
    iters: int = 24,
    damping: float = 0.5,
) -> Tuple[float, ...]:
    """Approximate minimax coefficients for x^{-1/p} on [l_min, l_max].

    Uses a Lawson-style IRLS procedure in the Chebyshev basis.
    """
    if p_val <= 0:
        raise ValueError(f"p_val must be strictly positive, got {p_val}")
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    if l_min <= 0:
        raise ValueError("l_min must be > 0 for x^{-1/p}")
    if l_min >= l_max:
        raise ValueError(f"l_min ({l_min}) must be strictly less than l_max ({l_max})")
    if grid_n < max(257, 4 * (degree + 1)):
        grid_n = max(257, 4 * (degree + 1))
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")
    if not (0.0 <= damping < 1.0):
        raise ValueError(f"damping must be in [0, 1), got {damping}")

    # Use Chebyshev-distributed fit samples for numerical stability.
    t = np.cos(np.pi * np.arange(grid_n) / (grid_n - 1))
    x = 0.5 * (l_max - l_min) * t + 0.5 * (l_max + l_min)
    y = np.power(x, -1.0 / p_val)
    V = np.polynomial.chebyshev.chebvander(t, degree)

    w = np.ones_like(y)
    coeffs = np.zeros(degree + 1, dtype=np.float64)

    for _ in range(iters):
        Aw = V * w[:, None]
        bw = y * w
        coeffs = np.linalg.lstsq(Aw, bw, rcond=None)[0]

        resid = y - (V @ coeffs)
        abs_resid = np.abs(resid)
        # Lawson-style multiplicative update to flatten the error distribution (minimax).
        rmax = max(float(abs_resid.max()), 1e-30)
        floor = max(rmax * 1e-3, 1e-15)
        w_new = w * np.maximum(abs_resid, floor)
        w_new /= np.mean(w_new)
        w = damping * w + (1.0 - damping) * w_new

    return tuple(float(c) for c in coeffs.tolist())


def estimate_inverse_proot_chebyshev_error(
    c_list: Tuple[float, ...],
    p_val: int,
    l_min: float,
    l_max: float = 1.0,
    grid_n: int = 4097,
) -> Tuple[float, float]:
    """Return (max_abs_err, max_rel_err) on a dense grid over [l_min, l_max]."""
    if p_val <= 0:
        raise ValueError(f"p_val must be strictly positive, got {p_val}")
    if l_min <= 0:
        raise ValueError("l_min must be > 0 for x^{-1/p}")
    if l_min >= l_max:
        raise ValueError(f"l_min ({l_min}) must be strictly less than l_max ({l_max})")
    if grid_n < 257:
        raise ValueError(f"grid_n must be >= 257, got {grid_n}")

    x = np.linspace(l_min, l_max, int(grid_n), dtype=np.float64)
    t = _map_to_chebyshev_domain(x, l_min, l_max)
    approx = np.polynomial.chebyshev.chebval(t, np.asarray(c_list, dtype=np.float64))
    truth = np.power(x, -1.0 / p_val)
    abs_err = np.abs(truth - approx)
    rel_err = abs_err / np.maximum(np.abs(truth), 1e-30)
    return float(np.max(abs_err)), float(np.max(rel_err))


@functools.lru_cache(maxsize=256)
def select_inverse_proot_chebyshev_minimax_auto(
    p_val: int,
    baseline_degree: int,
    l_min: float,
    l_max: float = 1.0,
    candidate_degrees: Tuple[int, ...] = (8, 12, 16, 24, 32),
    error_grid_n: int = 4097,
    max_relerr_mult: float = 1.0,
) -> Tuple[int, Tuple[float, ...], float, float, str]:
    """Pick the fastest minimax-auto degree that is no worse than fixed baseline.

    Returns (degree, coeffs, baseline_rel_err, selected_rel_err, mode_used).
    mode_used is one of {"fixed", "minimax-auto"}.
    """
    if baseline_degree < 0:
        raise ValueError(f"baseline_degree must be >= 0, got {baseline_degree}")
    if max_relerr_mult < 1.0:
        raise ValueError(
            f"max_relerr_mult must be >= 1.0 for safe fallback semantics, got {max_relerr_mult}"
        )

    base_coeffs = compute_chebyshev_coeffs_cached(p_val, baseline_degree, l_min, l_max)
    _, base_rel = estimate_inverse_proot_chebyshev_error(
        base_coeffs, p_val, l_min, l_max, grid_n=error_grid_n
    )

    if len(candidate_degrees) == 0:
        return baseline_degree, base_coeffs, float(base_rel), float(base_rel), "fixed"

    degs = sorted(
        {
            int(d)
            for d in candidate_degrees
            if int(d) >= 0 and int(d) <= int(baseline_degree)
        }
    )
    if len(degs) == 0:
        return baseline_degree, base_coeffs, float(base_rel), float(base_rel), "fixed"

    for d in degs:
        if d >= baseline_degree:
            continue
        try:
            coeffs = compute_inverse_proot_minimax_coeffs_cached(p_val, d, l_min, l_max)
            _, rel = estimate_inverse_proot_chebyshev_error(
                coeffs, p_val, l_min, l_max, grid_n=error_grid_n
            )
        except Exception:
            continue

        # Hard safety gate: only accept if approximation bound is no worse.
        if np.isfinite(rel) and rel <= (base_rel * max_relerr_mult):
            return int(d), coeffs, float(base_rel), float(rel), "minimax-auto"

    return baseline_degree, base_coeffs, float(base_rel), float(base_rel), "fixed"


@torch.no_grad()
def apply_inverse_chebyshev_with_coeffs(
    A: torch.Tensor,
    B: torch.Tensor,
    c_list: Tuple[float, ...],
    degree: int,
    l_min: float,
    l_max: float = 1.0,
    ws: Optional[ChebyshevApplyWorkspace] = None,
) -> Tuple[torch.Tensor, ChebyshevApplyWorkspace]:
    """
    Computes func(A) @ B using Clenshaw recurrence for Chebyshev polynomials, given predefined coefficients.
    Assumes eigenvalues of A are in [l_min, l_max].
    """
    _check_square(A)
    n = A.shape[-1]
    if B.shape[-2] != n:
        raise ValueError(f"B must have shape [..., {n}, :], got {B.shape}")
    if B.device != A.device or B.dtype != A.dtype:
        raise ValueError("A and B must have matching dtype and device")

    if l_min >= l_max:
        raise ValueError(f"l_min ({l_min}) must be strictly less than l_max ({l_max})")
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    if len(c_list) != degree + 1:
        raise ValueError(
            f"c_list must have length degree + 1 ({degree + 1}), got {len(c_list)}"
        )

    if not _ws_ok_chebyshev(ws, B):
        ws = _alloc_ws_chebyshev(B)
    assert ws is not None

    # Mappings
    scale = 2.0 / (l_max - l_min)
    shift = -(l_max + l_min) / (l_max - l_min)

    # Clenshaw Recurrence
    ws.T_next.zero_()  # U_{k+2}
    ws.T_curr.zero_()  # U_{k+1}

    for k in range(degree, 0, -1):
        # T_prev = (2 * scale) * A @ T_curr + (2 * shift) * T_curr
        # Compute A @ T_curr into T_prev since T_prev is free buffer
        _matmul_into(A, ws.T_curr, ws.T_prev)
        ws.T_prev.mul_(2 * scale)
        ws.T_prev.add_(ws.T_curr, alpha=2 * shift)

        # T_prev = T_prev - T_next + c_k * B
        ws.T_prev.sub_(ws.T_next)
        ws.T_prev.add_(B, alpha=c_list[k])

        # Shift buffers
        ws.T_next, ws.T_curr, ws.T_prev = ws.T_curr, ws.T_prev, ws.T_next

    # Final step k=0: result = c_0 * B + scale * (A @ T_curr) + shift * T_curr - T_next
    _matmul_into(A, ws.T_curr, ws.Z)
    ws.Z.mul_(scale)
    ws.Z.add_(ws.T_curr, alpha=shift)

    ws.Z.sub_(ws.T_next)
    ws.Z.add_(B, alpha=c_list[0])

    return ws.Z, ws


@torch.no_grad()
def apply_inverse_chebyshev(
    A: torch.Tensor,
    B: torch.Tensor,
    func: Callable[[np.ndarray], np.ndarray],
    degree: int,
    l_min: float,
    l_max: float = 1.0,
    ws: Optional[ChebyshevApplyWorkspace] = None,
) -> Tuple[torch.Tensor, ChebyshevApplyWorkspace]:
    """
    Computes func(A) @ B using Clenshaw recurrence for Chebyshev polynomials.
    """
    # Compute coefficients
    coeffs = compute_chebyshev_coeffs(func, degree, l_min, l_max)
    c_list = tuple(coeffs.tolist())
    return apply_inverse_chebyshev_with_coeffs(A, B, c_list, degree, l_min, l_max, ws)


def apply_inverse_proot_chebyshev(
    A: torch.Tensor,
    B: torch.Tensor,
    p_val: int,
    degree: int,
    l_min: float,
    l_max: float = 1.0,
    ws: Optional[ChebyshevApplyWorkspace] = None,
) -> Tuple[torch.Tensor, ChebyshevApplyWorkspace]:
    """
    Computes A^{-1/p} B using Chebyshev approximation.
    """
    if p_val <= 0:
        raise ValueError(f"p_val must be strictly positive, got {p_val}")
    if l_min <= 0:
        raise ValueError("l_min must be > 0 for x^{-1/p}")

    c_list = compute_chebyshev_coeffs_cached(p_val, degree, l_min, l_max)
    return apply_inverse_chebyshev_with_coeffs(A, B, c_list, degree, l_min, l_max, ws)
