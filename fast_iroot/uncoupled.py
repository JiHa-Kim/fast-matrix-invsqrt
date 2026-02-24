from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .utils import _matmul_into, _symmetrize_inplace
from .coeffs import _affine_coeffs, _quad_coeffs


@dataclass
class IrootWorkspaceUncoupled:
    X: torch.Tensor
    Xbuf: torch.Tensor
    T1: torch.Tensor
    T2: torch.Tensor
    eye_mat: torch.Tensor


def _alloc_ws_uncoupled(A: torch.Tensor) -> IrootWorkspaceUncoupled:
    shape = A.shape
    n = shape[-1]
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    return IrootWorkspaceUncoupled(
        X=A.new_empty(shape),
        Xbuf=A.new_empty(shape),
        T1=A.new_empty(shape),
        T2=A.new_empty(shape),
        eye_mat=eye,
    )


def _ws_ok_uncoupled(ws: Optional[IrootWorkspaceUncoupled], A: torch.Tensor) -> bool:
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
def inverse_proot_pe_affine_uncoupled(
    A_norm: torch.Tensor,
    ab_t: Sequence[Tuple[float, float]] | torch.Tensor,
    ws: Optional[IrootWorkspaceUncoupled] = None,
    symmetrize_X: bool = True,
) -> Tuple[torch.Tensor, IrootWorkspaceUncoupled]:
    if not _ws_ok_uncoupled(ws, A_norm):
        ws = _alloc_ws_uncoupled(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    coeffs = _affine_coeffs(ab_t)

    for a, b in coeffs:
        _matmul_into(A_norm, ws.X, ws.T1)
        _matmul_into(ws.X, ws.T1, ws.T2)

        ws.Xbuf.copy_(ws.T2).mul_(b)
        ws.Xbuf.add_(ws.X, alpha=a)

        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        if symmetrize_X:
            _symmetrize_inplace(ws.X, ws.Xbuf)

    return ws.X, ws


@torch.no_grad()
def inverse_proot_pe_quadratic_uncoupled(
    A_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[IrootWorkspaceUncoupled] = None,
    symmetrize_X: bool = True,
) -> Tuple[torch.Tensor, IrootWorkspaceUncoupled]:
    if not _ws_ok_uncoupled(ws, A_norm):
        ws = _alloc_ws_uncoupled(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    coeffs = _quad_coeffs(abc_t)

    for a, b, c in coeffs:
        _matmul_into(A_norm, ws.X, ws.T1)
        _matmul_into(ws.T1, ws.T1, ws.T2)

        ws.T1.mul_(b)
        ws.T2.mul_(c)
        ws.T1.add_(ws.T2)
        ws.T1.diagonal(dim1=-2, dim2=-1).add_(a)  # B in T1

        _matmul_into(ws.X, ws.T1, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        if symmetrize_X:
            _symmetrize_inplace(ws.X, ws.Xbuf)

    return ws.X, ws


@torch.no_grad()
def inverse_proot_ns_uncoupled(
    A_norm: torch.Tensor,
    iters: int,
    ws: Optional[IrootWorkspaceUncoupled] = None,
    symmetrize_X: bool = True,
) -> Tuple[torch.Tensor, IrootWorkspaceUncoupled]:
    ab_t = [(2.0, -1.0)] * iters
    return inverse_proot_pe_affine_uncoupled(
        A_norm=A_norm,
        ab_t=ab_t,
        ws=ws,
        symmetrize_X=symmetrize_X,
    )
