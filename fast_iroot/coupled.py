from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .utils import _matmul_into, _symmetrize_inplace, _bpow_times_y
from .coeffs import _quad_coeffs


@dataclass
class IrootWorkspaceCoupled:
    X: torch.Tensor
    Xbuf: torch.Tensor
    Y: torch.Tensor
    Ybuf: torch.Tensor
    Y2: torch.Tensor
    B: torch.Tensor
    B2: torch.Tensor
    eye_mat: torch.Tensor


@dataclass
class InverseSolveWorkspaceCoupled:
    Z: torch.Tensor
    Zbuf: torch.Tensor
    Y: torch.Tensor
    Ybuf: torch.Tensor
    Y2: torch.Tensor
    B: torch.Tensor
    B2: torch.Tensor


IsqrtWorkspaceCoupled = IrootWorkspaceCoupled


def _validate_p_val(p_val: int) -> None:
    if not isinstance(p_val, int) or p_val <= 0:
        raise ValueError("p_val must be a positive integer")


def _alloc_ws_coupled(A: torch.Tensor) -> IrootWorkspaceCoupled:
    shape = A.shape
    n = shape[-1]
    # IMPORTANT: do NOT .contiguous() an expanded identity; that materializes a full batch of I.
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    return IrootWorkspaceCoupled(
        X=A.new_empty(shape),
        Xbuf=A.new_empty(shape),
        Y=A.new_empty(shape),
        Ybuf=A.new_empty(shape),
        Y2=A.new_empty(shape),
        B=A.new_empty(shape),
        B2=A.new_empty(shape),
        eye_mat=eye,
    )


def _ws_ok_coupled(ws: Optional[IrootWorkspaceCoupled], A: torch.Tensor) -> bool:
    if ws is None:
        return False

    def _ok(t: torch.Tensor) -> bool:
        return t.device == A.device and t.dtype == A.dtype and t.shape == A.shape

    return (
        _ok(ws.X)
        and _ok(ws.Xbuf)
        and _ok(ws.Y)
        and _ok(ws.Ybuf)
        and _ok(ws.Y2)
        and _ok(ws.B)
        and _ok(ws.B2)
        and _ok(ws.eye_mat)
    )


def _alloc_ws_inverse_solve(
    A: torch.Tensor, M: torch.Tensor
) -> InverseSolveWorkspaceCoupled:
    shape_A = A.shape
    shape_M = M.shape
    return InverseSolveWorkspaceCoupled(
        Z=M.new_empty(shape_M),
        Zbuf=M.new_empty(shape_M),
        Y=A.new_empty(shape_A),
        Ybuf=A.new_empty(shape_A),
        Y2=A.new_empty(shape_A),
        B=A.new_empty(shape_A),
        B2=A.new_empty(shape_A),
    )


def _ws_ok_inverse_solve(
    ws: Optional[InverseSolveWorkspaceCoupled], A: torch.Tensor, M: torch.Tensor
) -> bool:
    if ws is None:
        return False

    def _ok_a(t: torch.Tensor) -> bool:
        return t.device == A.device and t.dtype == A.dtype and t.shape == A.shape

    def _ok_m(t: torch.Tensor) -> bool:
        return t.device == M.device and t.dtype == M.dtype and t.shape == M.shape

    return (
        _ok_m(ws.Z)
        and _ok_m(ws.Zbuf)
        and _ok_a(ws.Y)
        and _ok_a(ws.Ybuf)
        and _ok_a(ws.Y2)
        and _ok_a(ws.B)
        and _ok_a(ws.B2)
    )


@torch.no_grad()
def inverse_sqrt_pe_quadratic(
    A_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[IsqrtWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspaceCoupled]:
    """Coupled quadratic PE iteration for p=2 (inverse square root)."""
    if not _ws_ok_coupled(ws, A_norm):
        ws = _alloc_ws_coupled(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)
    coeffs = _quad_coeffs(abc_t)

    T = len(coeffs)
    for t, (a, b, c) in enumerate(coeffs):
        _matmul_into(ws.Y, ws.Y, ws.Y2)
        ws.B.copy_(ws.Y2).mul_(c)
        ws.B.add_(ws.Y, alpha=b)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(a)

        _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        if terminal_last_step and (t == T - 1):
            break

        _matmul_into(ws.Y, ws.B, ws.B2)
        _matmul_into(ws.B, ws.B2, ws.Ybuf)
        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf, ws.B)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws


@torch.no_grad()
def inverse_proot_pe_quadratic_coupled(
    A_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    p_val: int = 2,
    ws: Optional[IrootWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
) -> Tuple[torch.Tensor, IrootWorkspaceCoupled]:
    """Coupled quadratic PE iteration for general p (inverse p-th root)."""
    _validate_p_val(p_val)
    if not _ws_ok_coupled(ws, A_norm):
        ws = _alloc_ws_coupled(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)
    coeffs = _quad_coeffs(abc_t)

    T = len(coeffs)
    for t, (a, b, c) in enumerate(coeffs):
        _matmul_into(ws.Y, ws.Y, ws.Y2)
        ws.B.copy_(ws.Y2).mul_(c)
        ws.B.add_(ws.Y, alpha=b)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(a)

        _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        if terminal_last_step and (t == T - 1):
            break

        if p_val == 1:
            _matmul_into(ws.B, ws.Y, ws.Ybuf)
        elif p_val == 2:
            _matmul_into(ws.Y, ws.B, ws.B2)
            _matmul_into(ws.B, ws.B2, ws.Ybuf)
        elif p_val == 4:
            _matmul_into(ws.B, ws.B, ws.B2)
            _matmul_into(ws.B2, ws.Y, ws.Xbuf)
            _matmul_into(ws.B2, ws.Xbuf, ws.Ybuf)
        else:
            _bpow_times_y(ws.B, ws.Y, p_val, out=ws.Ybuf, tmp1=ws.B2, tmp2=ws.Xbuf)

        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf, ws.B)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws


@torch.no_grad()
def inverse_solve_pe_quadratic_coupled(
    A_norm: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[InverseSolveWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """Coupled quadratic PE iteration for computing A^{-1} M (inverse solving).

    This functionally equivalent to solving X * A_norm = I to get X = A_norm^{-1}
    and returning X * M_norm, but doing it directly via Z_{k+1} = Z_k B_k
    without materializing X.
    """
    if not _ws_ok_inverse_solve(ws, A_norm, M_norm):
        ws = _alloc_ws_inverse_solve(A_norm, M_norm)
    assert ws is not None

    ws.Z.copy_(M_norm)
    ws.Y.copy_(A_norm)
    coeffs = _quad_coeffs(abc_t)

    T = len(coeffs)
    for t, (a, b, c) in enumerate(coeffs):
        _matmul_into(ws.Y, ws.Y, ws.Y2)
        ws.B.copy_(ws.Y2).mul_(c)
        ws.B.add_(ws.Y, alpha=b)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(a)

        _matmul_into(ws.B, ws.Z, ws.Zbuf)
        ws.Z, ws.Zbuf = ws.Zbuf, ws.Z

        if terminal_last_step and (t == T - 1):
            break

        _matmul_into(ws.B, ws.Y, ws.Ybuf)

        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf, ws.B)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.Z, ws
