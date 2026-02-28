import math
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
from .coeffs import _quad_coeffs_hot

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
    Ztmp: torch.Tensor
    Y: torch.Tensor
    Ybuf: torch.Tensor
    B: torch.Tensor
    B2: torch.Tensor
    tmp: torch.Tensor


IsqrtWorkspaceCoupled = IrootWorkspaceCoupled


def _tail_poly_coeffs_from_residual_binomial(
    p_val: int,
    order: int,
) -> Tuple[float, float, float]:
    """Return q(Y)=a+bY+cY^2 from a local binomial tail in E=(I-Y).

    Uses:
      (I-E)^(-1/p) ≈ 1 + a1 E + a2 E^2, where
      a1 = 1/p and a2 = (1/p)(1/p + 1)/2.
    """
    p_i = int(p_val)
    if p_i <= 0:
        raise ValueError(f"p_val must be >= 1, got {p_val}")
    ord_i = int(order)
    if ord_i not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")

    inv_p = 1.0 / float(p_i)
    if ord_i == 1:
        return (1.0 + inv_p, -inv_p, 0.0)

    a2 = 0.5 * inv_p * (1.0 + inv_p)
    a = 1.0 + inv_p + a2
    b = -(inv_p + 2.0 * a2)
    c = a2
    return (float(a), float(b), float(c))


@torch.no_grad()
def _online_stop_error(
    Y: torch.Tensor,
    *,
    metric: str,
    scratch: torch.Tensor,
) -> float:
    """Compute a scalar Y-to-I proximity metric for online early stopping."""
    m = str(metric)
    if m == "diag":
        diag = Y.diagonal(dim1=-2, dim2=-1)
        return float(torch.max(torch.abs(diag - 1.0)).item())
    if m == "fro":
        scratch.copy_(Y)
        scratch.diagonal(dim1=-2, dim2=-1).sub_(1.0)
        fro = torch.linalg.matrix_norm(scratch, ord="fro")
        scaled = fro / math.sqrt(float(Y.shape[-1]))
        return float(torch.max(scaled).item())
    raise ValueError(
        f"online_stop_metric must be 'diag' or 'fro', got '{metric}'"
    )


@torch.no_grad()
def _renorm_coupled_state_(
    Y: torch.Tensor,
    W: torch.Tensor,
    *,
    p_val: int,
    eps: float,
) -> None:
    """Re-center Y around identity by trace scaling and keep W consistent.

    Uses mu = mean(diag(Y)) per batch item and applies:
      Y <- Y / |mu|
      W <- W / |mu|^(1/p)
    """
    diag_mean = Y.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    mu_abs = torch.abs(diag_mean)
    mu_abs = torch.where(torch.isfinite(mu_abs), mu_abs, torch.zeros_like(mu_abs))
    mu_abs = mu_abs.clamp_min(float(eps))

    scale_y = mu_abs.reciprocal().unsqueeze(-1).unsqueeze(-1)
    scale_w = torch.pow(mu_abs, -1.0 / float(p_val)).unsqueeze(-1).unsqueeze(-1)
    Y.mul_(scale_y)
    W.mul_(scale_w)


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
        Ztmp=M.new_empty(shape_M),
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
        and _ok_m(ws.Ztmp)
        and _ok_a(ws.Y)
        and _ok_a(ws.Ybuf)
        and _ok_a(ws.B)
        and _ok_a(ws.B2)
        and _ok_a(ws.tmp)
    )


@torch.no_grad()
def _apply_quadratic_left_rhs_terminal(
    Y: torch.Tensor,
    M: torch.Tensor,
    *,
    a: float,
    b: float,
    c: float,
    tmp_rhs: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """out <- (a I + b Y + c Y^2) M, without materializing B.

    Used on terminal steps where Y is not updated; this avoids a dense n×n
    polynomial build and is particularly faster for skinny RHS blocks (k << n).
    """
    _matmul_into(Y, M, tmp_rhs)  # Y M
    if abs(float(c)) <= _AFFINE_C_EPS:
        out.copy_(tmp_rhs)
        if float(b) != 1.0:
            out.mul_(float(b))
        out.add_(M, alpha=float(a))
        return out

    _matmul_into(Y, tmp_rhs, out)  # Y^2 M
    if float(c) != 1.0:
        out.mul_(float(c))
    if float(b) != 0.0:
        out.add_(tmp_rhs, alpha=float(b))
    out.add_(M, alpha=float(a))
    return out


@torch.no_grad()
def inverse_sqrt_pe_quadratic(
    A_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[IsqrtWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    symmetrize_every: int = 1,
    terminal_last_step: bool = True,
    assume_spd: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspaceCoupled]:
    """Coupled quadratic PE iteration for p=2 (inverse square root)."""
    _check_square(A_norm)
    assume_spd = bool(assume_spd)
    if not assume_spd:
        raise ValueError(
            "inverse_sqrt_pe_quadratic is SPD-only; use inverse_proot_pe_quadratic_coupled "
            "with p_val=2 and assume_spd=False for general matrices"
        )
    sym_every = int(symmetrize_every)
    if sym_every < 1:
        raise ValueError(f"symmetrize_every must be >= 1, got {symmetrize_every}")
    if not _ws_ok_coupled(ws, A_norm):
        ws = _alloc_ws_coupled(A_norm)
    assert ws is not None

    ws.X.zero_()
    ws.X.diagonal(dim1=-2, dim2=-1).fill_(1)
    ws.Y.copy_(A_norm)
    coeffs = _quad_coeffs_hot(abc_t)

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
    terminal_tail_steps: int = 1,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
    online_stop_metric: str = "diag",
    online_stop_check_every: int = 1,
    post_correction_steps: int = 0,
    post_correction_order: int = 2,
    assume_spd: bool = True,
    renorm_every: int = 0,
    renorm_eps: float = 1e-12,
) -> Tuple[torch.Tensor, IrootWorkspaceCoupled]:
    """Coupled quadratic PE iteration for general p (inverse p-th root).

    assume_spd=True keeps SPD-optimized symmetric updates for p>=2.
    assume_spd=False uses the general update Y <- B^p Y (no symmetry assumptions).
    For p=1, SPD assumptions are automatically disabled.
    """
    _validate_p_val(p_val)
    _check_square(A_norm)
    assume_spd = bool(assume_spd)
    if p_val == 1:
        # p=1 handles general inverses; do not force symmetry assumptions.
        assume_spd = False
        symmetrize_Y = False
    if not assume_spd and bool(symmetrize_Y):
        raise ValueError(
            "symmetrize_Y=True requires assume_spd=True for inverse_proot_pe_quadratic_coupled"
        )
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
    online_metric = str(online_stop_metric).strip().lower()
    if online_metric not in ("diag", "fro"):
        raise ValueError(
            "online_stop_metric must be 'diag' or 'fro', "
            f"got '{online_stop_metric}'"
        )
    stop_check_every = int(online_stop_check_every)
    if stop_check_every < 1:
        raise ValueError(
            f"online_stop_check_every must be >= 1, got {online_stop_check_every}"
        )
    post_steps = int(post_correction_steps)
    if post_steps < 0:
        raise ValueError(
            f"post_correction_steps must be >= 0, got {post_correction_steps}"
        )
    post_order = int(post_correction_order)
    if post_order not in (1, 2):
        raise ValueError(
            f"post_correction_order must be 1 or 2, got {post_correction_order}"
        )
    tail_steps = int(terminal_tail_steps)
    if tail_steps < 0:
        raise ValueError(
            f"terminal_tail_steps must be >= 0, got {terminal_tail_steps}"
        )
    renorm_period = int(renorm_every)
    if renorm_period < 0:
        raise ValueError(f"renorm_every must be >= 0, got {renorm_every}")
    renorm_eps_f = float(renorm_eps)
    if renorm_eps_f <= 0.0:
        raise ValueError(f"renorm_eps must be > 0, got {renorm_eps}")
    if post_steps > 0:
        if not assume_spd:
            raise ValueError(
                "post-correction tail requires assume_spd=True for stability"
            )
        if int(p_val) not in (2, 4):
            raise ValueError(
                "post-correction tail currently supports p_val in {2,4}, "
                f"got {p_val}"
            )
        tail_abc = _tail_poly_coeffs_from_residual_binomial(p_val, post_order)
    else:
        tail_abc = (0.0, 0.0, 0.0)
    # Preserve existing post-tail semantics: when post-correction is enabled,
    # keep Y updates through the main loop.
    if post_steps > 0:
        tail_steps = 0

    # SPD-only p=2 fast path (symmetric sandwich update).
    if (
        p_val == 2
        and assume_spd
        and ((not terminal_last_step) or tail_steps == 1)
        and online_stop_tol is None
        and online_metric == "diag"
        and stop_check_every == 1
        and post_steps == 0
        and renorm_period == 0
    ):
        X, ws2 = inverse_sqrt_pe_quadratic(
            A_norm,
            abc_t=abc_t,
            ws=ws,
            symmetrize_Y=symmetrize_Y,
            symmetrize_every=sym_every,
            terminal_last_step=terminal_last_step,
            assume_spd=assume_spd,
        )
        return X, ws2

    if not _ws_ok_coupled(ws, A_norm):
        ws = _alloc_ws_coupled(A_norm)
    assert ws is not None

    ws.X.zero_()
    ws.X.diagonal(dim1=-2, dim2=-1).fill_(1)
    ws.Y.copy_(A_norm)
    coeffs = _quad_coeffs_hot(abc_t)

    T = len(coeffs)
    if not terminal_last_step:
        tail_steps = 0
    if tail_steps > T:
        tail_steps = T
    tail_start = T - tail_steps if tail_steps > 0 else T
    for t, (a, b, c) in enumerate(coeffs):
        affine_step = abs(float(c)) <= _AFFINE_C_EPS
        if affine_step and p_val == 1:
            _apply_affine_right(ws.X, ws.Y, a=a, b=b, out=ws.Xbuf)
        else:
            _build_step_polynomial(ws.Y, a=a, b=b, c=c, out=ws.B)
            _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        is_tail_frozen = bool(t >= tail_start)
        if is_tail_frozen:
            continue

        if affine_step and p_val == 1:
            _update_y_affine_p1(ws.Y, a=a, b=b, out=ws.Ybuf)
        elif p_val == 1:
            _matmul_into(ws.B, ws.Y, ws.Ybuf)
        elif not assume_spd:
            _bpow(ws.B, p_val, out=ws.B2, tmp1=ws.Xbuf, tmp2=ws.Ybuf)
            _matmul_into(ws.B2, ws.Y, ws.Ybuf)
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
        if renorm_period > 0 and ((t + 1) % renorm_period == 0):
            _renorm_coupled_state_(
                ws.Y,
                ws.X,
                p_val=p_val,
                eps=renorm_eps_f,
            )

        # Low-overhead online early-stop based on Y diagonal closeness to identity.
        if (
            online_stop_tol is not None
            and (t + 1) >= online_min
            and (t + 1) < T
            and ((t + 1) % stop_check_every == 0)
        ):
            stop_err = _online_stop_error(
                ws.Y, metric=online_metric, scratch=ws.B2
            )
            if stop_err <= float(online_stop_tol):
                break

    if post_steps > 0:
        a_tail, b_tail, c_tail = tail_abc
        for _ in range(post_steps):
            affine_tail = abs(float(c_tail)) <= _AFFINE_C_EPS
            if affine_tail:
                _apply_affine_right(ws.X, ws.Y, a=a_tail, b=b_tail, out=ws.Xbuf)
            else:
                _build_step_polynomial(ws.Y, a=a_tail, b=b_tail, c=c_tail, out=ws.B)
                _matmul_into(ws.X, ws.B, ws.Xbuf)
            ws.X, ws.Xbuf = ws.Xbuf, ws.X

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
    terminal_tail_steps: int = 1,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
    online_stop_metric: str = "diag",
    online_stop_check_every: int = 1,
    post_correction_steps: int = 0,
    post_correction_order: int = 2,
    assume_spd: bool = True,
    nonspd_adaptive: bool = False,
    nonspd_adaptive_resid_tol: float = 1.0,
    nonspd_adaptive_growth_tol: float = 1.02,
    nonspd_adaptive_check_every: int = 1,
    nonspd_safe_fallback_tol: Optional[float] = None,
    nonspd_safe_early_y_tol: Optional[float] = None,
    nonspd_safe_early_metric: str = "fro",
    renorm_every: int = 0,
    renorm_eps: float = 1e-12,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """Coupled quadratic PE iteration for computing an inverse-like solve on M.

    This function continuously applies the generated coupled polynomial preconditioners
    as Z_{k+1} = B_k Z_k. Note that because B_k are dynamically generated left-to-right
    and applied iteratively, the final output corresponds to Z_T = B_{T-1}...B_1 B_0 M_norm.
    Set assume_spd=False to disable SPD-only symmetric Y updates.
    For p=1, SPD assumptions are automatically disabled.
    """
    _validate_p_val(p_val)
    _check_square(A_norm)
    assume_spd = bool(assume_spd)
    if p_val == 1:
        # p=1 handles general inverses; do not force symmetry assumptions.
        assume_spd = False
        symmetrize_Y = False
    if not assume_spd and bool(symmetrize_Y):
        raise ValueError(
            "symmetrize_Y=True requires assume_spd=True for inverse_solve_pe_quadratic_coupled"
        )
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
    online_metric = str(online_stop_metric).strip().lower()
    if online_metric not in ("diag", "fro"):
        raise ValueError(
            "online_stop_metric must be 'diag' or 'fro', "
            f"got '{online_stop_metric}'"
        )
    stop_check_every = int(online_stop_check_every)
    if stop_check_every < 1:
        raise ValueError(
            f"online_stop_check_every must be >= 1, got {online_stop_check_every}"
        )
    post_steps = int(post_correction_steps)
    if post_steps < 0:
        raise ValueError(
            f"post_correction_steps must be >= 0, got {post_correction_steps}"
        )
    post_order = int(post_correction_order)
    if post_order not in (1, 2):
        raise ValueError(
            f"post_correction_order must be 1 or 2, got {post_correction_order}"
        )
    tail_steps = int(terminal_tail_steps)
    if tail_steps < 0:
        raise ValueError(
            f"terminal_tail_steps must be >= 0, got {terminal_tail_steps}"
        )
    renorm_period = int(renorm_every)
    if renorm_period < 0:
        raise ValueError(f"renorm_every must be >= 0, got {renorm_every}")
    renorm_eps_f = float(renorm_eps)
    if renorm_eps_f <= 0.0:
        raise ValueError(f"renorm_eps must be > 0, got {renorm_eps}")
    if post_steps > 0:
        if not assume_spd:
            raise ValueError(
                "post-correction tail requires assume_spd=True for stability"
            )
        if int(p_val) not in (2, 4):
            raise ValueError(
                "post-correction tail currently supports p_val in {2,4}, "
                f"got {p_val}"
            )
        tail_abc = _tail_poly_coeffs_from_residual_binomial(p_val, post_order)
    else:
        tail_abc = (0.0, 0.0, 0.0)
    # Preserve existing post-tail semantics: when post-correction is enabled,
    # keep Y updates through the main loop.
    if post_steps > 0:
        tail_steps = 0
    if float(nonspd_adaptive_resid_tol) <= 0.0:
        raise ValueError(
            "nonspd_adaptive_resid_tol must be > 0, "
            f"got {nonspd_adaptive_resid_tol}"
        )
    if float(nonspd_adaptive_growth_tol) < 1.0:
        raise ValueError(
            "nonspd_adaptive_growth_tol must be >= 1.0, "
            f"got {nonspd_adaptive_growth_tol}"
        )
    if int(nonspd_adaptive_check_every) < 1:
        raise ValueError(
            "nonspd_adaptive_check_every must be >= 1, "
            f"got {nonspd_adaptive_check_every}"
        )
    if (
        nonspd_safe_fallback_tol is not None
        and float(nonspd_safe_fallback_tol) <= 0.0
    ):
        raise ValueError(
            "nonspd_safe_fallback_tol must be > 0 when provided, "
            f"got {nonspd_safe_fallback_tol}"
        )
    if (
        nonspd_safe_early_y_tol is not None
        and float(nonspd_safe_early_y_tol) <= 0.0
    ):
        raise ValueError(
            "nonspd_safe_early_y_tol must be > 0 when provided, "
            f"got {nonspd_safe_early_y_tol}"
        )
    nonspd_early_metric = str(nonspd_safe_early_metric).strip().lower()
    if nonspd_early_metric not in ("diag", "fro"):
        raise ValueError(
            "nonspd_safe_early_metric must be 'diag' or 'fro', "
            f"got '{nonspd_safe_early_metric}'"
        )
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
    coeffs = _quad_coeffs_hot(abc_t)
    adaptive_active = bool(nonspd_adaptive) and (p_val == 1) and (not assume_spd)
    safe_early_active = (
        (p_val == 1)
        and (not assume_spd)
        and (nonspd_safe_fallback_tol is not None)
        and (nonspd_safe_early_y_tol is not None)
    )
    needs_eye = bool(adaptive_active) or (
        bool(safe_early_active) and nonspd_early_metric == "fro"
    )
    if needs_eye:
        n = A_norm.shape[-1]
        eye = torch.eye(n, device=A_norm.device, dtype=A_norm.dtype)
    else:
        eye = None
    if adaptive_active:
        prev_proxy: Optional[float] = None
        check_every = int(nonspd_adaptive_check_every)
        resid_tol = float(nonspd_adaptive_resid_tol)
        growth_tol = float(nonspd_adaptive_growth_tol)
    else:
        prev_proxy = None
        check_every = 1
        resid_tol = 0.0
        growth_tol = 1.0

    def _fallback_solve_inplace() -> None:
        # Keep fallback close to torch.linalg.solve performance on CUDA.
        # Use fp32 solve for low-precision tensors; otherwise solve in input dtype.
        if A_norm.dtype in (torch.float16, torch.bfloat16):
            z_fb = torch.linalg.solve(A_norm.float(), M_norm.float()).to(
                dtype=M_norm.dtype
            )
        else:
            z_fb = torch.linalg.solve(A_norm, M_norm)
        ws.Z.copy_(z_fb)

    def _p1_y_proxy() -> float:
        assert eye is not None
        return float(
            (
                torch.linalg.matrix_norm(ws.Y - eye, ord="fro")
                / math.sqrt(float(ws.Y.shape[-1]))
            )
            .mean()
            .item()
        )

    def _p1_diag_proxy() -> float:
        diag = ws.Y.diagonal(dim1=-2, dim2=-1)
        return float(torch.max(torch.abs(diag - 1.0)).item())

    def _p1_z_proxy() -> float:
        num = torch.linalg.matrix_norm(A_norm @ ws.Z - M_norm, ord="fro")
        den = torch.linalg.matrix_norm(M_norm, ord="fro").clamp_min(1e-12)
        return float((num / den).mean().item())

    def _apply_p1_step(a_step: float, b_step: float, c_step: float, *, terminal: bool) -> None:
        affine_step = abs(float(c_step)) <= _AFFINE_C_EPS
        use_rhs_direct_terminal = (
            (not affine_step)
            and terminal
            and (ws.Z.shape[-1] < ws.Y.shape[-1])
        )
        if affine_step:
            _apply_affine_left(ws.Y, ws.Z, a=a_step, b=b_step, out=ws.Zbuf)
        elif use_rhs_direct_terminal:
            _apply_quadratic_left_rhs_terminal(
                ws.Y,
                ws.Z,
                a=a_step,
                b=b_step,
                c=c_step,
                tmp_rhs=ws.Ztmp,
                out=ws.Zbuf,
            )
        else:
            _build_step_polynomial(ws.Y, a=a_step, b=b_step, c=c_step, out=ws.B)
            _matmul_into(ws.B, ws.Z, ws.Zbuf)
        ws.Z, ws.Zbuf = ws.Zbuf, ws.Z
        if terminal:
            return
        if affine_step:
            _update_y_affine_p1(ws.Y, a=a_step, b=b_step, out=ws.Ybuf)
        else:
            _matmul_into(ws.B, ws.Y, ws.Ybuf)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    T = len(coeffs)
    if not terminal_last_step:
        tail_steps = 0
    if tail_steps > T:
        tail_steps = T
    tail_start = T - tail_steps if tail_steps > 0 else T
    for t, (a_base, b_base, c_base) in enumerate(coeffs):
        a = float(a_base)
        b = float(b_base)
        c = float(c_base)
        is_tail_frozen = bool(t >= tail_start)

        if p_val == 1:
            should_check = (
                adaptive_active
                and (t % check_every == 0)
                and (not is_tail_frozen)
            )

            if should_check:
                y_proxy_cur = _p1_y_proxy()
                unstable = False
                if prev_proxy is not None:
                    unstable = y_proxy_cur > prev_proxy * growth_tol
                    if (not unstable) and (y_proxy_cur > resid_tol) and (
                        y_proxy_cur > prev_proxy
                    ):
                        unstable = True

                if not unstable:
                    _apply_p1_step(a, b, c, terminal=False)
                    prev_proxy = _p1_y_proxy()
                else:
                    old_y = ws.Y.clone()
                    old_z = ws.Z.clone()

                    _apply_p1_step(a, b, c, terminal=False)
                    z_proxy_base = _p1_z_proxy()
                    y_proxy_base = _p1_y_proxy()
                    z_base = ws.Z.clone()
                    y_base = ws.Y.clone()

                    ws.Y.copy_(old_y)
                    ws.Z.copy_(old_z)
                    _apply_p1_step(2.0, -1.0, 0.0, terminal=False)
                    z_proxy_newton = _p1_z_proxy()
                    y_proxy_newton = _p1_y_proxy()

                    choose_newton = z_proxy_newton < z_proxy_base
                    if not choose_newton:
                        ws.Z.copy_(z_base)
                        ws.Y.copy_(y_base)
                        prev_proxy = y_proxy_base
                    else:
                        prev_proxy = y_proxy_newton
            else:
                _apply_p1_step(a, b, c, terminal=is_tail_frozen)
                if adaptive_active and (t % check_every == 0) and (not is_tail_frozen):
                    prev_proxy = _p1_y_proxy()

            if (not is_tail_frozen) and renorm_period > 0 and ((t + 1) % renorm_period == 0):
                _renorm_coupled_state_(
                    ws.Y,
                    ws.Z,
                    p_val=p_val,
                    eps=renorm_eps_f,
                )

            if safe_early_active and t == 0:
                # Cheap early divergence proxy for non-SPD p=1:
                # use Frobenius by default; diag is available for back-compat.
                if nonspd_early_metric == "fro":
                    y_proxy = _p1_y_proxy()
                else:
                    y_proxy = _p1_diag_proxy()
                if y_proxy > float(nonspd_safe_early_y_tol):
                    _fallback_solve_inplace()
                    return ws.Z, ws

            # Low-overhead online early-stop based on Y diagonal closeness to identity.
            if (
                online_stop_tol is not None
                and (t + 1) >= online_min
                and (t + 1) < T
                and (not is_tail_frozen)
                and ((t + 1) % stop_check_every == 0)
            ):
                stop_err = _online_stop_error(
                    ws.Y, metric=online_metric, scratch=ws.B2
                )
                if stop_err <= float(online_stop_tol):
                    break
            continue

        affine_step = abs(float(c)) <= _AFFINE_C_EPS
        use_rhs_direct_terminal = (
            (not affine_step)
            and is_tail_frozen
            and (ws.Z.shape[-1] < ws.Y.shape[-1])
        )
        if affine_step and (p_val == 1 or (p_val == 2 and assume_spd)):
            _apply_affine_left(ws.Y, ws.Z, a=a, b=b, out=ws.Zbuf)
        elif use_rhs_direct_terminal:
            _apply_quadratic_left_rhs_terminal(
                ws.Y,
                ws.Z,
                a=a,
                b=b,
                c=c,
                tmp_rhs=ws.Ztmp,
                out=ws.Zbuf,
            )
        else:
            _build_step_polynomial(ws.Y, a=a, b=b, c=c, out=ws.B)
            _matmul_into(ws.B, ws.Z, ws.Zbuf)
        ws.Z, ws.Zbuf = ws.Zbuf, ws.Z

        # If terminal RHS-direct apply was used but we still need Y (e.g., post-tail),
        # materialize the current-step polynomial once for a correct final Y update.
        if (t == T - 1) and (post_steps > 0) and use_rhs_direct_terminal:
            _build_step_polynomial(ws.Y, a=a, b=b, c=c, out=ws.B)

        if is_tail_frozen:
            continue

        if affine_step and p_val == 1:
            _update_y_affine_p1(ws.Y, a=a, b=b, out=ws.Ybuf)
        elif affine_step and p_val == 2 and assume_spd:
            _update_y_affine_p2(ws.Y, a=a, b=b, tmp_y2=ws.tmp, out=ws.Ybuf)
        elif p_val == 1:
            _matmul_into(ws.B, ws.Y, ws.Ybuf)
        elif p_val == 2 and assume_spd:
            # Avoid _bpow(p_half=1) which would copy B into B2; use the symmetric update directly.
            _matmul_into(ws.Y, ws.B, ws.B2)
            _matmul_into(ws.B, ws.B2, ws.Ybuf)
        elif not assume_spd:
            _bpow(ws.B, p_val, out=ws.B2, tmp1=ws.tmp, tmp2=ws.Ybuf)
            _matmul_into(ws.B2, ws.Y, ws.Ybuf)
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
        if renorm_period > 0 and ((t + 1) % renorm_period == 0):
            _renorm_coupled_state_(
                ws.Y,
                ws.Z,
                p_val=p_val,
                eps=renorm_eps_f,
            )

        # Low-overhead online early-stop based on Y diagonal closeness to identity.
        if (
            online_stop_tol is not None
            and (t + 1) >= online_min
            and (t + 1) < T
            and ((t + 1) % stop_check_every == 0)
        ):
            stop_err = _online_stop_error(
                ws.Y, metric=online_metric, scratch=ws.B2
            )
            if stop_err <= float(online_stop_tol):
                break

    if post_steps > 0:
        a_tail, b_tail, c_tail = tail_abc
        for _ in range(post_steps):
            affine_tail = abs(float(c_tail)) <= _AFFINE_C_EPS
            if affine_tail:
                _apply_affine_left(ws.Y, ws.Z, a=a_tail, b=b_tail, out=ws.Zbuf)
            else:
                _apply_quadratic_left_rhs_terminal(
                    ws.Y,
                    ws.Z,
                    a=a_tail,
                    b=b_tail,
                    c=c_tail,
                    tmp_rhs=ws.Ztmp,
                    out=ws.Zbuf,
                )
            ws.Z, ws.Zbuf = ws.Zbuf, ws.Z

    if (p_val == 1) and (not assume_spd) and (nonspd_safe_fallback_tol is not None):
        num = torch.linalg.matrix_norm(A_norm @ ws.Z - M_norm, ord="fro")
        den = torch.linalg.matrix_norm(M_norm, ord="fro").clamp_min(1e-12)
        rel = float((num / den).max().item())
        if rel > float(nonspd_safe_fallback_tol):
            _fallback_solve_inplace()

    return ws.Z, ws
