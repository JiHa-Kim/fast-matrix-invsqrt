"""
Archived affine-based inverse p-th root iterations.

These methods are superseded by the quadratic (PE-Quad) variants which deliver
consistently better residuals at equal or lower iteration cost.

This file is kept for reference only — NOT wired into the main package.

Archived functions:
  - inverse_proot_pe_affine_uncoupled  (uncoupled affine PE iteration)
  - inverse_proot_pe_affine_coupled    (coupled affine PE iteration)
  - inverse_proot_ns_uncoupled         (Newton-Schulz via affine, uncoupled)
  - inverse_proot_ns_coupled           (Newton-Schulz via affine, coupled)
  - inverse_sqrt_ns                    (Newton-Schulz for p=2, coupled)
  - inverse_sqrt_pe_affine             (PE affine for p=2, coupled)
  - _affine_coeffs                     (coefficient conversion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Coefficient helpers
# ---------------------------------------------------------------------------


def _affine_coeffs(
    ab_t: Sequence[Tuple[float, float]] | torch.Tensor,
) -> List[Tuple[float, float]]:
    if isinstance(ab_t, torch.Tensor):
        ab_cpu = ab_t.detach().to(device="cpu", dtype=torch.float32)
        return [
            (float(ab_cpu[t, 0]), float(ab_cpu[t, 1]))
            for t in range(int(ab_cpu.shape[0]))
        ]
    return [(float(a), float(b)) for (a, b) in ab_t]


# ---------------------------------------------------------------------------
# Utility stubs (inlined to keep archive self-contained)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _symmetrize_inplace(M: torch.Tensor, tmp: Optional[torch.Tensor] = None) -> None:
    if tmp is None:
        M.copy_(0.5 * (M + M.mT))
        return
    tmp.copy_(M.mT)
    M.add_(tmp).mul_(0.5)


@torch.no_grad()
def _matmul_into(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    torch.matmul(A, B, out=out)
    return out


@torch.no_grad()
def _addmm_into(bias, mat1, mat2, *, beta, alpha, out):
    if mat1.dim() == 2:
        torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha, out=out)
    else:
        torch.baddbmm(bias, mat1, mat2, beta=beta, alpha=alpha, out=out)
    return out


# ---------------------------------------------------------------------------
# Uncoupled workspace (shared with quadratic — reproduced here for reference)
# ---------------------------------------------------------------------------


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


def _ws_ok_uncoupled(ws, A):
    if ws is None:
        return False
    return ws.X.device == A.device and ws.X.dtype == A.dtype and ws.X.shape == A.shape


# ---------------------------------------------------------------------------
# Uncoupled affine PE iteration
# ---------------------------------------------------------------------------


@torch.no_grad()
def inverse_proot_pe_affine_uncoupled(
    A_norm: torch.Tensor,
    ab_t: Sequence[Tuple[float, float]] | torch.Tensor,
    p_val: int = 2,
    ws: Optional[IrootWorkspaceUncoupled] = None,
    symmetrize_X: bool = True,
) -> Tuple[torch.Tensor, IrootWorkspaceUncoupled]:
    if not _ws_ok_uncoupled(ws, A_norm):
        ws = _alloc_ws_uncoupled(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    coeffs = _affine_coeffs(ab_t)

    for a, b in coeffs:
        if p_val == 1:
            _matmul_into(ws.X, A_norm, ws.T1)
            _addmm_into(ws.X, ws.X, ws.T1, beta=a, alpha=b, out=ws.Xbuf)
        elif p_val == 2:
            _matmul_into(ws.X, ws.X, ws.T1)
            _matmul_into(ws.T1, A_norm, ws.T2)
            _addmm_into(ws.X, ws.X, ws.T2, beta=a, alpha=b, out=ws.Xbuf)
        elif p_val == 3:
            _matmul_into(ws.X, ws.X, ws.T1)
            _matmul_into(ws.T1, ws.X, ws.T2)
            _matmul_into(ws.T2, A_norm, ws.T1)
            _addmm_into(ws.X, ws.X, ws.T1, beta=a, alpha=b, out=ws.Xbuf)
        elif p_val == 4:
            _matmul_into(ws.X, ws.X, ws.T1)
            _matmul_into(ws.T1, ws.T1, ws.T2)
            _matmul_into(ws.T2, A_norm, ws.T1)
            _addmm_into(ws.X, ws.X, ws.T1, beta=a, alpha=b, out=ws.Xbuf)
        else:
            ws.T1.copy_(ws.X)
            for _ in range(p_val - 1):
                _matmul_into(ws.T1, ws.X, ws.T2)
                ws.T1.copy_(ws.T2)
            _matmul_into(ws.T1, A_norm, ws.T2)
            _addmm_into(ws.X, ws.X, ws.T2, beta=a, alpha=b, out=ws.Xbuf)

        ws.X, ws.Xbuf = ws.Xbuf, ws.X
        if symmetrize_X:
            _symmetrize_inplace(ws.X, ws.Xbuf)

    return ws.X, ws


@torch.no_grad()
def inverse_proot_ns_uncoupled(
    A_norm: torch.Tensor,
    iters: int,
    p_val: int = 2,
    ws: Optional[IrootWorkspaceUncoupled] = None,
    symmetrize_X: bool = True,
) -> Tuple[torch.Tensor, IrootWorkspaceUncoupled]:
    ab_t = [((p_val + 1) / p_val, -1.0 / p_val)] * iters
    return inverse_proot_pe_affine_uncoupled(
        A_norm=A_norm,
        ab_t=ab_t,
        p_val=p_val,
        ws=ws,
        symmetrize_X=symmetrize_X,
    )


# ---------------------------------------------------------------------------
# Coupled workspace (shared with quadratic — reproduced for reference)
# ---------------------------------------------------------------------------


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


def _alloc_ws_coupled(A):
    shape = A.shape
    n = shape[-1]
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


def _ws_ok_coupled(ws, A):
    if ws is None:
        return False
    return ws.X.device == A.device and ws.X.dtype == A.dtype and ws.X.shape == A.shape


# ---------------------------------------------------------------------------
# Coupled affine PE iterations
# ---------------------------------------------------------------------------


@torch.no_grad()
def inverse_sqrt_ns(
    A_norm: torch.Tensor,
    iters: int,
    ws=None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
):
    if not _ws_ok_coupled(ws, A_norm):
        ws = _alloc_ws_coupled(A_norm)
    assert ws is not None
    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)
    T = int(iters)
    for t in range(T):
        ws.B.copy_(ws.Y).mul_(-0.5)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(1.5)
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
def inverse_sqrt_pe_affine(
    A_norm: torch.Tensor,
    ab_t: Sequence[Tuple[float, float]] | torch.Tensor,
    ws=None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
):
    if not _ws_ok_coupled(ws, A_norm):
        ws = _alloc_ws_coupled(A_norm)
    assert ws is not None
    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)
    coeffs = _affine_coeffs(ab_t)
    T = len(coeffs)
    for t, (a, b) in enumerate(coeffs):
        ws.B.copy_(ws.Y).mul_(b)
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
def inverse_proot_ns_coupled(
    A_norm: torch.Tensor,
    iters: int,
    p_val: int = 2,
    ws=None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
):
    ab_t = [((p_val + 1) / p_val, -1.0 / p_val)] * iters
    return inverse_proot_pe_affine_coupled(
        A_norm=A_norm,
        ab_t=ab_t,
        p_val=p_val,
        ws=ws,
        symmetrize_Y=symmetrize_Y,
        terminal_last_step=terminal_last_step,
    )


@torch.no_grad()
def inverse_proot_pe_affine_coupled(
    A_norm: torch.Tensor,
    ab_t: Sequence[Tuple[float, float]] | torch.Tensor,
    p_val: int = 2,
    ws=None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
):
    if not _ws_ok_coupled(ws, A_norm):
        ws = _alloc_ws_coupled(A_norm)
    assert ws is not None
    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)
    coeffs = _affine_coeffs(ab_t)
    T = len(coeffs)
    for t, (a, b) in enumerate(coeffs):
        ws.B.copy_(ws.Y).mul_(b)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(a)
        _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X
        if terminal_last_step and (t == T - 1):
            break
        if p_val == 2:
            _matmul_into(ws.Y, ws.B, ws.B2)
            _matmul_into(ws.B, ws.B2, ws.Ybuf)
        else:
            ws.Ybuf.copy_(ws.Y)
            for _ in range(p_val):
                _matmul_into(ws.B, ws.Ybuf, ws.B2)
                ws.Ybuf.copy_(ws.B2)
        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf, ws.B)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y
    return ws.X, ws
