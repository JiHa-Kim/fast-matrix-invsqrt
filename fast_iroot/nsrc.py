"""
nsrc.py — Neumann-Series Residual Correction solvers.

Implements GEMM-heavy approximate linear solve Z ≈ A⁻¹B using:
  1. Scalar-preconditioned residual iteration (NSRC)
  2. Matrix-preconditioned residual iteration (using a frozen M⁻¹)
  3. Hybrid PE + NSRC (use PE iteration to build M⁻¹, then refine)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .utils import _matmul_into, _check_square


@dataclass
class NSRCWorkspace:
    Z: torch.Tensor  # current solution (n×k)
    R: torch.Tensor  # residual buffer (n×k)
    D: torch.Tensor  # correction buffer (n×k)


def _alloc_nsrc_ws(B: torch.Tensor) -> NSRCWorkspace:
    shape = B.shape
    return NSRCWorkspace(
        Z=B.new_empty(shape),
        R=B.new_empty(shape),
        D=B.new_empty(shape),
    )


def _ws_ok_nsrc(ws: Optional[NSRCWorkspace], B: torch.Tensor) -> bool:
    if ws is None:
        return False

    def _ok(t: torch.Tensor) -> bool:
        return t.device == B.device and t.dtype == B.dtype and t.shape == B.shape

    return _ok(ws.Z) and _ok(ws.R) and _ok(ws.D)


@torch.no_grad()
def nsrc_solve(
    A_norm: torch.Tensor,
    B: torch.Tensor,
    alpha: float,
    max_iter: int = 10,
    tol: Optional[float] = None,
    ws: Optional[NSRCWorkspace] = None,
) -> Tuple[torch.Tensor, NSRCWorkspace]:
    """Scalar-preconditioned Neumann-series residual correction.

    Iterates: Z₀ = α·B, then Z_{k+1} = Z_k + α·(B - A·Z_k).

    Args:
        A_norm: preconditioned SPD matrix (n×n), eigenvalues in ~[l_min, 1].
        B: right-hand side matrix (n×k).
        alpha: scalar preconditioner, typically 2/(λ_min + λ_max).
        max_iter: maximum number of refinement iterations.
        tol: optional early-stop tolerance on relative residual norm.
        ws: reusable workspace.

    Returns:
        (Z, ws) where Z ≈ A_norm⁻¹ B.
    """
    _check_square(A_norm)
    n = A_norm.shape[-1]
    if B.shape[-2] != n:
        raise ValueError(f"B must have shape [..., {n}, :], got {B.shape}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    a = float(alpha)
    if a <= 0.0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    if not _ws_ok_nsrc(ws, B):
        ws = _alloc_nsrc_ws(B)
    assert ws is not None

    # Z₀ = α·B
    ws.Z.copy_(B)
    ws.Z.mul_(a)

    b_norm: Optional[float] = None

    for i in range(max_iter):
        # R = B - A·Z
        _matmul_into(A_norm, ws.Z, ws.R)
        ws.R.mul_(-1.0)
        ws.R.add_(B)

        # Early stop check
        if tol is not None:
            r_norm = float(torch.linalg.matrix_norm(ws.R, ord="fro").max().item())
            if b_norm is None:
                b_norm = float(
                    torch.linalg.matrix_norm(B, ord="fro").max().clamp_min(1e-30).item()
                )
            if r_norm / b_norm <= float(tol):
                break

        # Z += α·R
        ws.Z.add_(ws.R, alpha=a)

    return ws.Z, ws


@torch.no_grad()
def nsrc_solve_preconditioned(
    A_norm: torch.Tensor,
    B: torch.Tensor,
    M_inv: torch.Tensor,
    max_iter: int = 5,
    tol: Optional[float] = None,
    ws: Optional[NSRCWorkspace] = None,
) -> Tuple[torch.Tensor, NSRCWorkspace]:
    """Matrix-preconditioned residual correction.

    Iterates: Z₀ = M⁻¹·B, then Z_{k+1} = Z_k + M⁻¹·(B - A·Z_k).
    M_inv is a frozen approximate inverse of A_norm.

    Args:
        A_norm: preconditioned matrix (n×n).
        B: right-hand side matrix (n×k).
        M_inv: approximate inverse of A_norm (n×n). Must be precomputed.
        max_iter: maximum number of refinement iterations.
        tol: optional early-stop tolerance on relative residual norm.
        ws: reusable workspace.

    Returns:
        (Z, ws) where Z ≈ A_norm⁻¹ B.
    """
    _check_square(A_norm)
    _check_square(M_inv)
    n = A_norm.shape[-1]
    if B.shape[-2] != n:
        raise ValueError(f"B must have shape [..., {n}, :], got {B.shape}")
    if M_inv.shape[-1] != n or M_inv.shape[-2] != n:
        raise ValueError(f"M_inv must have shape [..., {n}, {n}], got {M_inv.shape}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    if not _ws_ok_nsrc(ws, B):
        ws = _alloc_nsrc_ws(B)
    assert ws is not None

    # Z₀ = M⁻¹·B
    _matmul_into(M_inv, B, ws.Z)

    b_norm: Optional[float] = None

    for i in range(max_iter):
        # R = B - A·Z
        _matmul_into(A_norm, ws.Z, ws.R)
        ws.R.mul_(-1.0)
        ws.R.add_(B)

        # Early stop check
        if tol is not None:
            r_norm = float(torch.linalg.matrix_norm(ws.R, ord="fro").max().item())
            if b_norm is None:
                b_norm = float(
                    torch.linalg.matrix_norm(B, ord="fro").max().clamp_min(1e-30).item()
                )
            if r_norm / b_norm <= float(tol):
                break

        # D = M⁻¹·R
        _matmul_into(M_inv, ws.R, ws.D)

        # Z += D
        ws.Z.add_(ws.D)

    return ws.Z, ws


@torch.no_grad()
def hybrid_pe_nsrc_solve(
    A_norm: torch.Tensor,
    B: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    pe_steps: int = 2,
    ref_steps: int = 3,
    tol: Optional[float] = None,
    ws: Optional[NSRCWorkspace] = None,
) -> Tuple[torch.Tensor, NSRCWorkspace]:
    """Hybrid PE + NSRC solve.

    Phase 1: Build rough M⁻¹ using `pe_steps` steps of PE iteration.
    Phase 2: Refine with `ref_steps` steps of matrix-preconditioned NSRC.

    Args:
        A_norm: preconditioned SPD matrix (n×n).
        B: right-hand side matrix (n×k).
        abc_t: PE coefficient schedule (≥ pe_steps triples).
        pe_steps: number of PE iteration steps to build preconditioner.
        ref_steps: number of NSRC refinement steps.
        tol: optional early-stop tolerance on relative residual norm.
        ws: reusable workspace (for NSRC phase only).

    Returns:
        (Z, ws) where Z ≈ A_norm⁻¹ B.
    """
    from .coupled import inverse_proot_pe_quadratic_coupled
    from .coeffs import _quad_coeffs

    _check_square(A_norm)
    n = A_norm.shape[-1]
    if B.shape[-2] != n:
        raise ValueError(f"B must have shape [..., {n}, :], got {B.shape}")
    if pe_steps < 1:
        raise ValueError(f"pe_steps must be >= 1, got {pe_steps}")
    if ref_steps < 0:
        raise ValueError(f"ref_steps must be >= 0, got {ref_steps}")

    # Parse and truncate coefficients to pe_steps
    coeffs = _quad_coeffs(abc_t)
    if len(coeffs) < pe_steps:
        raise ValueError(
            f"abc_t has {len(coeffs)} triples but pe_steps={pe_steps} requested"
        )
    pe_coeffs = coeffs[:pe_steps]

    # Phase 1: Build M⁻¹ via PE iteration (X → A⁻¹)
    M_inv, _pe_ws = inverse_proot_pe_quadratic_coupled(
        A_norm,
        abc_t=pe_coeffs,
        p_val=1,
        symmetrize_Y=False,
        terminal_last_step=False,
        assume_spd=False,
    )

    if ref_steps == 0:
        # No refinement; just apply M⁻¹
        if not _ws_ok_nsrc(ws, B):
            ws = _alloc_nsrc_ws(B)
        assert ws is not None
        _matmul_into(M_inv, B, ws.Z)
        return ws.Z, ws

    # Phase 2: Refine with NSRC
    return nsrc_solve_preconditioned(
        A_norm,
        B,
        M_inv,
        max_iter=ref_steps,
        tol=tol,
        ws=ws,
    )
