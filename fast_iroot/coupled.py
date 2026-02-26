from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .utils import (
    _matmul_into,
    _addmm_into,
    _symmetrize_inplace,
    _bpow,
    _validate_p_val,
    _check_square,
)
from .coeffs import _quad_coeffs

_AFFINE_C_EPS = 1e-15


@dataclass
class IrootWorkspaceCoupled:
    X: torch.Tensor
    Xbuf: torch.Tensor
    Y: torch.Tensor
    Ybuf: torch.Tensor
    B: torch.Tensor
    B2: torch.Tensor


@dataclass
class InverseSolveWorkspaceCoupled:
    Z: torch.Tensor
    Zbuf: torch.Tensor
    Y: torch.Tensor
    Ybuf: torch.Tensor
    B: torch.Tensor
    B2: torch.Tensor
    tmp: torch.Tensor


IsqrtWorkspaceCoupled = IrootWorkspaceCoupled


@torch.no_grad()
def _build_step_polynomial(
    Y: torch.Tensor, *, a: float, b: float, c: float, out: torch.Tensor
) -> torch.Tensor:
    """Build B = a I + b Y + c Y^2; skip GEMM when c == 0 (affine step)."""
    if abs(float(c)) <= _AFFINE_C_EPS:
        if float(b) == 0.0:
            out.zero_()
        elif float(b) == 1.0:
            out.copy_(Y)
        else:
            out.copy_(Y)
            out.mul_(float(b))
    else:
        _addmm_into(Y, Y, Y, beta=float(b), alpha=float(c), out=out)
    out.diagonal(dim1=-2, dim2=-1).add_(float(a))
    return out


@torch.no_grad()
def _apply_affine_left(
    Y: torch.Tensor, M: torch.Tensor, *, a: float, b: float, out: torch.Tensor
) -> torch.Tensor:
    """out <- (a I + b Y) M, without materializing B."""
    _addmm_into(M, Y, M, beta=float(a), alpha=float(b), out=out)
    return out


@torch.no_grad()
def _apply_affine_right(
    X: torch.Tensor, Y: torch.Tensor, *, a: float, b: float, out: torch.Tensor
) -> torch.Tensor:
    """out <- X (a I + b Y), without materializing B."""
    _addmm_into(X, X, Y, beta=float(a), alpha=float(b), out=out)
    return out


@torch.no_grad()
def _update_y_affine_p1(
    Y: torch.Tensor, *, a: float, b: float, out: torch.Tensor
) -> torch.Tensor:
    """out <- (a I + b Y) Y = a Y + b Y^2."""
    _addmm_into(Y, Y, Y, beta=float(a), alpha=float(b), out=out)
    return out


@torch.no_grad()
def _update_y_affine_p2(
    Y: torch.Tensor,
    *,
    a: float,
    b: float,
    tmp_y2: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """out <- (a I + b Y) Y (a I + b Y) = a^2 Y + 2ab Y^2 + b^2 Y^3."""
    _matmul_into(Y, Y, tmp_y2)        # Y^2
    _matmul_into(tmp_y2, Y, out)      # Y^3
    bb = float(b) * float(b)
    if bb != 1.0:
        out.mul_(bb)
    out.add_(tmp_y2, alpha=2.0 * float(a) * float(b))
    out.add_(Y, alpha=float(a) * float(a))
    return out


def _alloc_ws_coupled(A: torch.Tensor) -> IrootWorkspaceCoupled:
    shape = A.shape
    return IrootWorkspaceCoupled(
        X=A.new_empty(shape),
        Xbuf=A.new_empty(shape),
        Y=A.new_empty(shape),
        Ybuf=A.new_empty(shape),
        B=A.new_empty(shape),
        B2=A.new_empty(shape),
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
        and _ok(ws.B)
        and _ok(ws.B2)
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
        B=A.new_empty(shape_A),
        B2=A.new_empty(shape_A),
        tmp=A.new_empty(shape_A),
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
        and _ok_a(ws.B)
        and _ok_a(ws.B2)
        and _ok_a(ws.tmp)
    )


@torch.no_grad()
def inverse_sqrt_pe_quadratic(
    A_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[IsqrtWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    symmetrize_every: int = 1,
    terminal_last_step: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspaceCoupled]:
    """Coupled quadratic PE iteration for p=2 (inverse square root)."""
    _check_square(A_norm)
    sym_every = int(symmetrize_every)
    if sym_every < 1:
        raise ValueError(f"symmetrize_every must be >= 1, got {symmetrize_every}")
    if not _ws_ok_coupled(ws, A_norm):
        ws = _alloc_ws_coupled(A_norm)
    assert ws is not None

    ws.X.zero_()
    ws.X.diagonal(dim1=-2, dim2=-1).fill_(1)
    ws.Y.copy_(A_norm)
    coeffs = _quad_coeffs(abc_t)

    T = len(coeffs)
    for t, (a, b, c) in enumerate(coeffs):
        _build_step_polynomial(ws.Y, a=a, b=b, c=c, out=ws.B)

        _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        if terminal_last_step and (t == T - 1):
            break

        _matmul_into(ws.Y, ws.B, ws.B2)
        _matmul_into(ws.B, ws.B2, ws.Ybuf)
        if symmetrize_Y and ((t + 1) % sym_every == 0):
            # ws.B2 is used as scratch here; contents destroyed.
            _symmetrize_inplace(ws.Ybuf, ws.B2)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws


@torch.no_grad()
def inverse_proot_pe_quadratic_coupled(
    A_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    p_val: int = 2,
    ws: Optional[IrootWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    symmetrize_every: int = 1,
    terminal_last_step: bool = True,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
) -> Tuple[torch.Tensor, IrootWorkspaceCoupled]:
    """Coupled quadratic PE iteration for general p (inverse p-th root)."""
    _validate_p_val(p_val)
    _check_square(A_norm)
    sym_every = int(symmetrize_every)
    if sym_every < 1:
        raise ValueError(f"symmetrize_every must be >= 1, got {symmetrize_every}")
    if online_stop_tol is not None and float(online_stop_tol) <= 0.0:
        raise ValueError(
            f"online_stop_tol must be > 0 when provided, got {online_stop_tol}"
        )
    online_min = int(online_min_steps)
    if online_min < 1:
        raise ValueError(f"online_min_steps must be >= 1, got {online_min_steps}")

    # p=2 has an optimized coupled implementation (avoids an extra B->B2 copy).
    if p_val == 2:
        X, ws2 = inverse_sqrt_pe_quadratic(
            A_norm,
            abc_t=abc_t,
            ws=ws,
            symmetrize_Y=symmetrize_Y,
            symmetrize_every=sym_every,
            terminal_last_step=terminal_last_step,
        )
        return X, ws2

    if not _ws_ok_coupled(ws, A_norm):
        ws = _alloc_ws_coupled(A_norm)
    assert ws is not None

    ws.X.zero_()
    ws.X.diagonal(dim1=-2, dim2=-1).fill_(1)
    ws.Y.copy_(A_norm)
    coeffs = _quad_coeffs(abc_t)

    T = len(coeffs)
    for t, (a, b, c) in enumerate(coeffs):
        affine_step = abs(float(c)) <= _AFFINE_C_EPS
        if affine_step and p_val == 1:
            _apply_affine_right(ws.X, ws.Y, a=a, b=b, out=ws.Xbuf)
        else:
            _build_step_polynomial(ws.Y, a=a, b=b, c=c, out=ws.B)
            _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        if terminal_last_step and (t == T - 1):
            break

        if affine_step and p_val == 1:
            _update_y_affine_p1(ws.Y, a=a, b=b, out=ws.Ybuf)
        elif p_val == 1:
            _matmul_into(ws.B, ws.Y, ws.Ybuf)
        elif p_val == 3:
            # Specialized odd-p fast path avoids _bpow(..., p_half=1) copy overhead.
            # This realizes Y <- B^2 Y B (commuting-model equivalent to B^3 Y);
            # optional symmetrization controls finite-precision drift.
            _matmul_into(ws.B, ws.Y, ws.Xbuf)
            _matmul_into(ws.B, ws.Xbuf, ws.Ybuf)
            _matmul_into(ws.Ybuf, ws.B, ws.B2)
            ws.Ybuf, ws.B2 = ws.B2, ws.Ybuf
        elif p_val % 2 == 0:
            p_half = p_val // 2
            _bpow(ws.B, p_half, out=ws.B2, tmp1=ws.Xbuf, tmp2=ws.Ybuf)
            _matmul_into(ws.B2, ws.Y, ws.Xbuf)
            _matmul_into(ws.Xbuf, ws.B2, ws.Ybuf)
        else:
            p_half = (p_val - 1) // 2
            _bpow(ws.B, p_half, out=ws.B2, tmp1=ws.Xbuf, tmp2=ws.Ybuf)
            _matmul_into(ws.B, ws.Y, ws.Xbuf)
            _matmul_into(ws.B2, ws.Xbuf, ws.Ybuf)
            # Odd-p update: Y <- B^h (B Y) B^h = B^(h+1) Y B^h.
            # Under the commuting PE model this is equivalent to B^p Y.
            # Avoid a full-matrix copy_ by writing the final result into ws.B (B is dead after this point).
            _matmul_into(ws.Ybuf, ws.B2, ws.B)
            ws.Ybuf, ws.B = ws.B, ws.Ybuf

        if symmetrize_Y and ((t + 1) % sym_every == 0):
            # ws.B2 is used as scratch here; contents destroyed.
            _symmetrize_inplace(ws.Ybuf, ws.B2)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

        # Low-overhead online early-stop based on Y diagonal closeness to identity.
        if (
            online_stop_tol is not None
            and (t + 1) >= online_min
            and (t + 1) < T
        ):
            diag = ws.Y.diagonal(dim1=-2, dim2=-1)
            diag_err = torch.max(torch.abs(diag - 1.0))
            if float(diag_err) <= float(online_stop_tol):
                break

    return ws.X, ws


@torch.no_grad()
def inverse_solve_pe_quadratic_coupled(
    A_norm: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    p_val: int = 2,
    ws: Optional[InverseSolveWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    symmetrize_every: int = 1,
    terminal_last_step: bool = True,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """Coupled quadratic PE iteration for computing an inverse-like solve on M.

    This function continuously applies the generated coupled polynomial preconditioners
    as Z_{k+1} = B_k Z_k. Note that because B_k are dynamically generated left-to-right
    and applied iteratively, the final output corresponds to Z_T = B_{T-1}...B_1 B_0 M_norm.
    """
    _validate_p_val(p_val)
    _check_square(A_norm)
    sym_every = int(symmetrize_every)
    if sym_every < 1:
        raise ValueError(f"symmetrize_every must be >= 1, got {symmetrize_every}")
    if online_stop_tol is not None and float(online_stop_tol) <= 0.0:
        raise ValueError(
            f"online_stop_tol must be > 0 when provided, got {online_stop_tol}"
        )
    online_min = int(online_min_steps)
    if online_min < 1:
        raise ValueError(f"online_min_steps must be >= 1, got {online_min_steps}")
    if M_norm.shape[-2] != A_norm.shape[-1]:
        raise ValueError(
            f"M_norm must have shape[..., {A_norm.shape[-1]}, :], got {M_norm.shape}"
        )
    if M_norm.device != A_norm.device:
        raise ValueError("A_norm and M_norm must be on the same device")
    if M_norm.dtype != A_norm.dtype:
        raise ValueError("A_norm and M_norm must have the same dtype")
    if not _ws_ok_inverse_solve(ws, A_norm, M_norm):
        ws = _alloc_ws_inverse_solve(A_norm, M_norm)
    assert ws is not None

    ws.Z.copy_(M_norm)
    ws.Y.copy_(A_norm)
    coeffs = _quad_coeffs(abc_t)

    T = len(coeffs)
    for t, (a, b, c) in enumerate(coeffs):
        affine_step = abs(float(c)) <= _AFFINE_C_EPS
        if affine_step and (p_val == 1 or p_val == 2):
            _apply_affine_left(ws.Y, ws.Z, a=a, b=b, out=ws.Zbuf)
        else:
            _build_step_polynomial(ws.Y, a=a, b=b, c=c, out=ws.B)
            _matmul_into(ws.B, ws.Z, ws.Zbuf)
        ws.Z, ws.Zbuf = ws.Zbuf, ws.Z

        if terminal_last_step and (t == T - 1):
            break

        if affine_step and p_val == 1:
            _update_y_affine_p1(ws.Y, a=a, b=b, out=ws.Ybuf)
        elif affine_step and p_val == 2:
            _update_y_affine_p2(ws.Y, a=a, b=b, tmp_y2=ws.tmp, out=ws.Ybuf)
        elif p_val == 1:
            _matmul_into(ws.B, ws.Y, ws.Ybuf)
        elif p_val == 2:
            # Avoid _bpow(p_half=1) which would copy B into B2; use the symmetric update directly.
            _matmul_into(ws.Y, ws.B, ws.B2)
            _matmul_into(ws.B, ws.B2, ws.Ybuf)
        elif p_val == 3:
            # Specialized odd-p fast path avoids _bpow(..., p_half=1) copy overhead.
            # This realizes Y <- B^2 Y B (commuting-model equivalent to B^3 Y);
            # optional symmetrization controls finite-precision drift.
            _matmul_into(ws.B, ws.Y, ws.tmp)
            _matmul_into(ws.B, ws.tmp, ws.Ybuf)
            _matmul_into(ws.Ybuf, ws.B, ws.B2)
            ws.Ybuf, ws.B2 = ws.B2, ws.Ybuf
        elif p_val % 2 == 0:
            p_half = p_val // 2
            _bpow(ws.B, p_half, out=ws.B2, tmp1=ws.tmp, tmp2=ws.Ybuf)
            _matmul_into(ws.B2, ws.Y, ws.tmp)
            _matmul_into(ws.tmp, ws.B2, ws.Ybuf)
        else:
            p_half = (p_val - 1) // 2
            _bpow(ws.B, p_half, out=ws.B2, tmp1=ws.tmp, tmp2=ws.Ybuf)
            _matmul_into(ws.B, ws.Y, ws.tmp)
            _matmul_into(ws.B2, ws.tmp, ws.Ybuf)
            # Odd-p update: Y <- B^h (B Y) B^h = B^(h+1) Y B^h.
            # Under the commuting PE model this is equivalent to B^p Y.
            # Avoid a full-matrix copy_ by writing the final result into ws.B (B is dead after this point).
            _matmul_into(ws.Ybuf, ws.B2, ws.B)
            ws.Ybuf, ws.B = ws.B, ws.Ybuf

        if symmetrize_Y and ((t + 1) % sym_every == 0):
            # ws.B2 is used as scratch here; contents destroyed.
            _symmetrize_inplace(ws.Ybuf, ws.B2)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

        # Low-overhead online early-stop based on Y diagonal closeness to identity.
        if (
            online_stop_tol is not None
            and (t + 1) >= online_min
            and (t + 1) < T
        ):
            diag = ws.Y.diagonal(dim1=-2, dim2=-1)
            diag_err = torch.max(torch.abs(diag - 1.0))
            if float(diag_err) <= float(online_stop_tol):
                break

    return ws.Z, ws
