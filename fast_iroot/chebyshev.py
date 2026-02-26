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
