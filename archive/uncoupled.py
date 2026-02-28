from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .coeffs import _quad_coeffs_hot
from .utils import (
    _bpow_times_y,
    _addmm_into,
    _matmul_into,
    _symmetrize_inplace,
    _validate_p_val,
    _check_square,
)


@dataclass
class IrootWorkspaceUncoupled:
    X: torch.Tensor
    Xbuf: torch.Tensor
    T1: torch.Tensor
    T2: torch.Tensor


def _alloc_ws_uncoupled(A: torch.Tensor) -> IrootWorkspaceUncoupled:
    shape = A.shape
    return IrootWorkspaceUncoupled(
        X=A.new_empty(shape),
        Xbuf=A.new_empty(shape),
        T1=A.new_empty(shape),
        T2=A.new_empty(shape),
    )


def _ws_ok_uncoupled(ws: Optional[IrootWorkspaceUncoupled], A: torch.Tensor) -> bool:
    if ws is None:
        return False

    def _ok(t: torch.Tensor) -> bool:
        return t.device == A.device and t.dtype == A.dtype and t.shape == A.shape

    return _ok(ws.X) and _ok(ws.Xbuf) and _ok(ws.T1) and _ok(ws.T2)


@torch.no_grad()
def inverse_proot_pe_quadratic_uncoupled(
    A_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    p_val: int = 2,
    ws: Optional[IrootWorkspaceUncoupled] = None,
    symmetrize_X: bool = True,
    assume_spd: bool = True,
) -> Tuple[torch.Tensor, IrootWorkspaceUncoupled]:
    _validate_p_val(p_val)
    _check_square(A_norm)
    assume_spd = bool(assume_spd)
    if not assume_spd and bool(symmetrize_X):
        raise ValueError(
            "symmetrize_X=True requires assume_spd=True for inverse_proot_pe_quadratic_uncoupled"
        )
    if not _ws_ok_uncoupled(ws, A_norm):
        ws = _alloc_ws_uncoupled(A_norm)
    assert ws is not None

    ws.X.zero_()
    ws.X.diagonal(dim1=-2, dim2=-1).fill_(1)
    coeffs = _quad_coeffs_hot(abc_t)

    for a, b, c in coeffs:
        if p_val == 1:
            _matmul_into(ws.X, A_norm, ws.T2)  # T2 = Y = X*A
        elif p_val == 2:
            _matmul_into(ws.X, ws.X, ws.T1)
            _matmul_into(ws.T1, A_norm, ws.T2)
        elif p_val == 4:
            _matmul_into(ws.X, ws.X, ws.T2)  # X^2
            _matmul_into(ws.T2, ws.T2, ws.T1)  # X^4
            _matmul_into(ws.T1, A_norm, ws.T2)
        elif p_val == 3:
            # Specialized odd-p fast path to avoid generic binary-power buffer logic.
            _matmul_into(ws.X, A_norm, ws.T1)  # X * A
            _matmul_into(ws.X, ws.T1, ws.T2)  # X^2 * A
            _matmul_into(ws.X, ws.T2, ws.T1)  # X^3 * A
            ws.T1, ws.T2 = ws.T2, ws.T1
        else:
            _bpow_times_y(ws.X, A_norm, p_val, out=ws.T2, tmp1=ws.T1, tmp2=ws.Xbuf)

        _addmm_into(ws.T2, ws.T2, ws.T2, beta=b, alpha=c, out=ws.T1)
        ws.T1.diagonal(dim1=-2, dim2=-1).add_(a)  # B in T1

        _matmul_into(ws.X, ws.T1, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        if symmetrize_X:
            _symmetrize_inplace(ws.X, ws.Xbuf)

    return ws.X, ws
