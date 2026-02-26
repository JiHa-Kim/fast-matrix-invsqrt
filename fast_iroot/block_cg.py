"""
block_cg.py — Block Conjugate Gradient solver for SPD systems.

Implements GEMM-heavy approximate linear solve Z ≈ A⁻¹B using
preconditioned Block CG. The dominant cost per iteration is one
A @ P matmul (n×n × n×k GEMM), making it GPU-efficient for k > 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .utils import _matmul_into, _check_square


@dataclass
class BlockCGWorkspace:
    Z: torch.Tensor  # solution (n×k)
    R: torch.Tensor  # residual (n×k)
    P: torch.Tensor  # search direction (n×k)
    AP: torch.Tensor  # A @ P buffer (n×k)
    S: torch.Tensor  # preconditioned residual buffer (n×k)


def _alloc_cg_ws(B: torch.Tensor) -> BlockCGWorkspace:
    shape = B.shape
    return BlockCGWorkspace(
        Z=B.new_empty(shape),
        R=B.new_empty(shape),
        P=B.new_empty(shape),
        AP=B.new_empty(shape),
        S=B.new_empty(shape),
    )


def _ws_ok_cg(ws: Optional[BlockCGWorkspace], B: torch.Tensor) -> bool:
    if ws is None:
        return False

    def _ok(t: torch.Tensor) -> bool:
        return t.device == B.device and t.dtype == B.dtype and t.shape == B.shape

    return _ok(ws.Z) and _ok(ws.R) and _ok(ws.P) and _ok(ws.AP) and _ok(ws.S)


def _apply_diag_precond(
    R: torch.Tensor, diag_inv: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Apply diagonal preconditioner: out = diag(diag_inv) @ R."""
    # diag_inv has shape (..., n); R has shape (..., n, k)
    out.copy_(R)
    out.mul_(diag_inv.unsqueeze(-1))
    return out


@torch.no_grad()
def block_cg_solve(
    A_norm: torch.Tensor,
    B: torch.Tensor,
    max_iter: int = 10,
    tol: float = 1e-3,
    diag_precond: Optional[torch.Tensor] = None,
    ws: Optional[BlockCGWorkspace] = None,
) -> Tuple[torch.Tensor, BlockCGWorkspace, int]:
    """Preconditioned Block Conjugate Gradient for SPD A_norm.

    Solves A_norm @ Z = B via Block CG. The dominant cost per iteration
    is one (n×n)×(n×k) GEMM.

    Args:
        A_norm: SPD matrix (n×n), eigenvalues in ~[l_min, 1].
        B: right-hand side matrix (n×k).
        max_iter: maximum CG iterations.
        tol: relative residual tolerance for early stopping.
        diag_precond: optional diagonal preconditioner, shape (..., n).
            If None, uses identity (unpreconditioned CG).
        ws: reusable workspace.

    Returns:
        (Z, ws, iters) where Z ≈ A_norm⁻¹ B and iters is iterations used.
    """
    _check_square(A_norm)
    n = A_norm.shape[-1]
    if B.shape[-2] != n:
        raise ValueError(f"B must have shape [..., {n}, :], got {B.shape}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0.0:
        raise ValueError(f"tol must be > 0, got {tol}")

    if not _ws_ok_cg(ws, B):
        ws = _alloc_cg_ws(B)
    assert ws is not None

    has_precond = diag_precond is not None

    # Z₀ = 0 (or preconditioned initial guess)
    if has_precond:
        # Z₀ = M⁻¹ B as initial guess
        _apply_diag_precond(B, diag_precond, ws.Z)
        # R₀ = B - A Z₀
        _matmul_into(A_norm, ws.Z, ws.R)
        ws.R.mul_(-1.0)
        ws.R.add_(B)
        # P₀ = M⁻¹ R₀
        _apply_diag_precond(ws.R, diag_precond, ws.P)
    else:
        # Z₀ = 0, R₀ = B, P₀ = B
        ws.Z.zero_()
        ws.R.copy_(B)
        ws.P.copy_(B)

    b_norm = float(torch.linalg.matrix_norm(B, ord="fro").max().clamp_min(1e-30).item())

    # Track r^T M⁻¹ r for the CG beta update
    # For block CG we use the trace-based scalar variant for simplicity
    if has_precond:
        _apply_diag_precond(ws.R, diag_precond, ws.S)
        rtz = float(torch.sum(ws.R * ws.S).item())
    else:
        rtz = float(torch.sum(ws.R * ws.R).item())

    iters_used = 0

    for i in range(max_iter):
        # AP = A @ P — the dominant GEMM
        _matmul_into(A_norm, ws.P, ws.AP)

        # α = (r^T z) / (p^T A p) — scalar for simplicity
        ptap = float(torch.sum(ws.P * ws.AP).item())
        if abs(ptap) < 1e-30:
            break
        alpha_cg = rtz / ptap

        # Z += α P
        ws.Z.add_(ws.P, alpha=alpha_cg)

        # R -= α AP
        ws.R.sub_(ws.AP, alpha=alpha_cg)

        iters_used = i + 1

        # Check convergence
        r_norm = float(torch.linalg.matrix_norm(ws.R, ord="fro").max().item())
        if r_norm / b_norm <= tol:
            break

        # Compute new z = M⁻¹ R
        if has_precond:
            _apply_diag_precond(ws.R, diag_precond, ws.S)
            rtz_new = float(torch.sum(ws.R * ws.S).item())
        else:
            rtz_new = float(torch.sum(ws.R * ws.R).item())

        # β = rtz_new / rtz
        if abs(rtz) < 1e-30:
            break
        beta_cg = rtz_new / rtz
        rtz = rtz_new

        # P = z + β P  (where z = S if preconditioned, else z = R)
        ws.P.mul_(beta_cg)
        if has_precond:
            ws.P.add_(ws.S)
        else:
            ws.P.add_(ws.R)

    return ws.Z, ws, iters_used


@torch.no_grad()
def block_cg_solve_with_precond(
    A_norm: torch.Tensor,
    B: torch.Tensor,
    M_inv: torch.Tensor,
    max_iter: int = 10,
    tol: float = 1e-3,
    ws: Optional[BlockCGWorkspace] = None,
) -> Tuple[torch.Tensor, BlockCGWorkspace, int]:
    """Block CG with a dense matrix preconditioner M⁻¹.

    Same as block_cg_solve but uses a full (n×n) matrix preconditioner
    instead of a diagonal. Each preconditioner application is an additional GEMM.

    Args:
        A_norm: SPD matrix (n×n).
        B: right-hand side matrix (n×k).
        M_inv: approximate inverse preconditioner (n×n).
        max_iter: maximum CG iterations.
        tol: relative residual tolerance for early stopping.
        ws: reusable workspace.

    Returns:
        (Z, ws, iters) where Z ≈ A_norm⁻¹ B.
    """
    _check_square(A_norm)
    _check_square(M_inv)
    n = A_norm.shape[-1]
    if B.shape[-2] != n:
        raise ValueError(f"B must have shape [..., {n}, :], got {B.shape}")
    if M_inv.shape[-1] != n:
        raise ValueError(f"M_inv must have shape [..., {n}, {n}], got {M_inv.shape}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    if not _ws_ok_cg(ws, B):
        ws = _alloc_cg_ws(B)
    assert ws is not None

    # Z₀ = M⁻¹ B
    _matmul_into(M_inv, B, ws.Z)

    # R₀ = B - A Z₀
    _matmul_into(A_norm, ws.Z, ws.R)
    ws.R.mul_(-1.0)
    ws.R.add_(B)

    # P₀ = M⁻¹ R₀
    _matmul_into(M_inv, ws.R, ws.P)

    b_norm = float(torch.linalg.matrix_norm(B, ord="fro").max().clamp_min(1e-30).item())

    # r^T z_0
    rtz = float(torch.sum(ws.R * ws.P).item())
    iters_used = 0

    for i in range(max_iter):
        # AP = A @ P
        _matmul_into(A_norm, ws.P, ws.AP)

        ptap = float(torch.sum(ws.P * ws.AP).item())
        if abs(ptap) < 1e-30:
            break
        alpha_cg = rtz / ptap

        ws.Z.add_(ws.P, alpha=alpha_cg)
        ws.R.sub_(ws.AP, alpha=alpha_cg)
        iters_used = i + 1

        r_norm = float(torch.linalg.matrix_norm(ws.R, ord="fro").max().item())
        if r_norm / b_norm <= tol:
            break

        # S = M⁻¹ R (preconditioner apply — 1 extra GEMM)
        _matmul_into(M_inv, ws.R, ws.S)

        rtz_new = float(torch.sum(ws.R * ws.S).item())
        if abs(rtz) < 1e-30:
            break
        beta_cg = rtz_new / rtz
        rtz = rtz_new

        # P = S + β P
        ws.P.mul_(beta_cg)
        ws.P.add_(ws.S)

    return ws.Z, ws, iters_used
