"""
lu_ir.py — Mixed-precision LU + Iterative Refinement solver.

Uses torch.linalg.lu_factor/lu_solve (cuSOLVER getrf/getrs) for the
factorization, then refines via residual correction. This is the direct
analog of cuSOLVER's cusolverDnIRSXgesv, implemented in pure PyTorch.

Key idea: factor once, refine cheaply. The factorization is O(n^3/3),
the refinement loop is O(n^2*k) per step (1 GEMM for residual + 1 TRSM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .utils import _check_square


@dataclass
class LUIRWorkspace:
    Z: torch.Tensor  # current solution (n×k)
    R: torch.Tensor  # residual buffer (n×k)


def _alloc_luir_ws(B: torch.Tensor) -> LUIRWorkspace:
    return LUIRWorkspace(
        Z=B.new_empty(B.shape),
        R=B.new_empty(B.shape),
    )


def _ws_ok(ws: Optional[LUIRWorkspace], B: torch.Tensor) -> bool:
    if ws is None:
        return False
    return (
        ws.Z.device == B.device
        and ws.Z.dtype == B.dtype
        and ws.Z.shape == B.shape
        and ws.R.device == B.device
        and ws.R.dtype == B.dtype
        and ws.R.shape == B.shape
    )


@torch.no_grad()
def lu_ir_solve(
    A: torch.Tensor,
    B: torch.Tensor,
    max_refine: int = 3,
    tol: Optional[float] = None,
    factor_dtype: Optional[torch.dtype] = None,
    ws: Optional[LUIRWorkspace] = None,
) -> Tuple[torch.Tensor, LUIRWorkspace]:
    """Mixed-precision LU + iterative refinement solve.

    1. Factor A (optionally in lower precision) via torch.linalg.lu_factor
    2. Initial solve Z₀ via torch.linalg.lu_solve
    3. Refine: R = B - A@Z, D = lu_solve(LU, R), Z += D

    Args:
        A: square matrix (n×n).
        B: right-hand side (n×k).
        max_refine: number of refinement iterations (0 = pure LU solve).
        tol: optional early-stop on relative residual norm.
        factor_dtype: optional dtype for LU factorization. If set and differs
            from A.dtype, the factorization is done in this lower precision
            (e.g. torch.float32 when A is float64, or even casting to a
            supported type). Default: same as A.dtype.
        ws: reusable workspace.

    Returns:
        (Z, ws) where Z ≈ A⁻¹B.
    """
    _check_square(A)
    n = A.shape[-1]
    if B.shape[-2] != n:
        raise ValueError(f"B must have shape [..., {n}, :], got {B.shape}")

    if not _ws_ok(ws, B):
        ws = _alloc_luir_ws(B)
    assert ws is not None

    # Phase 1: LU factorization (possibly in lower precision)
    if factor_dtype is not None and factor_dtype != A.dtype:
        A_fac = A.to(factor_dtype)
    else:
        A_fac = A
    LU, pivots = torch.linalg.lu_factor(A_fac)

    # Phase 2: Initial solve
    B_fac = B.to(A_fac.dtype) if B.dtype != A_fac.dtype else B
    Z_fac = torch.linalg.lu_solve(LU, pivots, B_fac)

    # Cast back to working precision
    ws.Z.copy_(Z_fac.to(B.dtype) if Z_fac.dtype != B.dtype else Z_fac)

    if max_refine <= 0:
        return ws.Z, ws

    # Phase 3: Iterative refinement
    b_norm: Optional[float] = None

    for i in range(max_refine):
        # R = B - A @ Z  (residual in working precision — the key to stability)
        torch.matmul(A, ws.Z, out=ws.R)
        ws.R.mul_(-1.0)
        ws.R.add_(B)

        # Check convergence
        if tol is not None:
            r_norm = float(torch.linalg.matrix_norm(ws.R, ord="fro").max().item())
            if b_norm is None:
                b_norm = float(
                    torch.linalg.matrix_norm(B, ord="fro").max().clamp_min(1e-30).item()
                )
            if r_norm / b_norm <= float(tol):
                break

        # Correction: D = LU⁻¹ R (solve in factor precision)
        R_fac = ws.R.to(A_fac.dtype) if ws.R.dtype != A_fac.dtype else ws.R
        D = torch.linalg.lu_solve(LU, pivots, R_fac)

        # Z += D (in working precision)
        ws.Z.add_(D.to(B.dtype) if D.dtype != B.dtype else D)

    return ws.Z, ws


@torch.no_grad()
def lu_solve_direct(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Direct LU solve — thin wrapper around torch.linalg.solve for benchmarking.

    This calls the same cuSOLVER path as torch.linalg.solve but via explicit
    lu_factor + lu_solve, allowing future optimization of each step.
    """
    LU, pivots = torch.linalg.lu_factor(A)
    return torch.linalg.lu_solve(LU, pivots, B)
