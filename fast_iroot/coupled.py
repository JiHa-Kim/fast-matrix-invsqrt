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


IsqrtWorkspaceCoupled = IrootWorkspaceCoupled


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
    return (
        ws.X.device == A.device
        and ws.X.dtype == A.dtype
        and ws.X.shape == A.shape
        and ws.eye_mat.device == A.device
        and ws.eye_mat.dtype == A.dtype
        and ws.eye_mat.shape == A.shape
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

        if p_val == 2:
            _matmul_into(ws.Y, ws.B, ws.B2)
            _matmul_into(ws.B, ws.B2, ws.Ybuf)
        else:
            _bpow_times_y(ws.B, ws.Y, p_val, out=ws.Ybuf, tmp1=ws.B2, tmp2=ws.Xbuf)

        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf, ws.B)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws
