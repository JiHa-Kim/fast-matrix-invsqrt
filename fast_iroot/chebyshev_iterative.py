"""
chebyshev_iterative.py — Chebyshev semi-iterative method for SPD systems.

Computes Z ≈ A⁻¹B using the Chebyshev three-term recurrence.
Unlike CG, this method requires no dot products (no global GPU synchronization),
making it purely GEMM + pointwise. Requires spectral bounds [λ_min, λ_max].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .utils import _matmul_into, _check_square


@dataclass
class ChebyshevIterWorkspace:
    Z_curr: torch.Tensor  # current iterate (n×k)
    Z_prev: torch.Tensor  # previous iterate (n×k)
    AZ: torch.Tensor  # A @ Z buffer (n×k)


def _alloc_cheb_iter_ws(B: torch.Tensor) -> ChebyshevIterWorkspace:
    shape = B.shape
    return ChebyshevIterWorkspace(
        Z_curr=B.new_empty(shape),
        Z_prev=B.new_empty(shape),
        AZ=B.new_empty(shape),
    )


def _ws_ok_cheb_iter(ws: Optional[ChebyshevIterWorkspace], B: torch.Tensor) -> bool:
    if ws is None:
        return False

    def _ok(t: torch.Tensor) -> bool:
        return t.device == B.device and t.dtype == B.dtype and t.shape == B.shape

    return _ok(ws.Z_curr) and _ok(ws.Z_prev) and _ok(ws.AZ)


@torch.no_grad()
def chebyshev_iterative_solve(
    A_norm: torch.Tensor,
    B: torch.Tensor,
    l_min: float,
    l_max: float = 1.0,
    max_iter: int = 10,
    tol: Optional[float] = None,
    ws: Optional[ChebyshevIterWorkspace] = None,
) -> Tuple[torch.Tensor, ChebyshevIterWorkspace, int]:
    """Chebyshev semi-iterative method for SPD systems.

    Solves A_norm @ Z = B using three-term recurrence with Chebyshev-optimal
    acceleration parameters. No dot products needed (sync-free on GPU).

    Each iteration costs 1 GEMM (n×n × n×k) + pointwise operations.

    Args:
        A_norm: SPD matrix (n×n), eigenvalues in [l_min, l_max].
        B: right-hand side matrix (n×k).
        l_min: lower bound on eigenvalues of A_norm (must be > 0).
        l_max: upper bound on eigenvalues of A_norm.
        max_iter: maximum iterations.
        tol: optional early-stop tolerance on residual norm.
        ws: reusable workspace.

    Returns:
        (Z, ws, iters) where Z ≈ A_norm⁻¹ B.
    """
    _check_square(A_norm)
    n = A_norm.shape[-1]
    if B.shape[-2] != n:
        raise ValueError(f"B must have shape [..., {n}, :], got {B.shape}")
    if l_min <= 0:
        raise ValueError(f"l_min must be > 0, got {l_min}")
    if l_min >= l_max:
        raise ValueError(f"l_min ({l_min}) must be < l_max ({l_max})")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    if not _ws_ok_cheb_iter(ws, B):
        ws = _alloc_cheb_iter_ws(B)
    assert ws is not None

    lmin = float(l_min)
    lmax = float(l_max)

    # Chebyshev parameters
    # Center and half-width of the spectral interval
    c = (lmax + lmin) / 2.0  # center
    d = (lmax - lmin) / 2.0  # half-width
    alpha = 2.0 / (lmax + lmin)  # optimal first-step scaling

    # Step 0: Z₀ = α·B  (Richardson step with optimal scalar)
    ws.Z_curr.copy_(B)
    ws.Z_curr.mul_(alpha)

    b_norm: Optional[float] = None
    iters_used = 1

    if max_iter == 1:
        return ws.Z_curr, ws, iters_used

    # Step 1: Compute residual R₀ = B - A·Z₀, then Z₁ via Chebyshev
    # AZ = A @ Z₀
    _matmul_into(A_norm, ws.Z_curr, ws.AZ)

    # R₀ = B - AZ (in-place into AZ, reusing as buffer)
    # Actually we need Z_prev = Z₀ for the recurrence, so save first
    ws.Z_prev.copy_(ws.Z_curr)

    # For the first Chebyshev step (k=1), ω₁ = 2c/(2c² - d²)... but
    # the standard form uses the recurrence directly.
    #
    # Standard Chebyshev iteration for Ax=b (Golub & Van Loan §11.2.7):
    #   σ₀ = 1/c,  ρ₀ = 1
    #   Z₀ = σ₀ · B
    #   For k ≥ 1:
    #     ρ_k = 1/(2σ₀c - ρ_{k-1})       (σ₀ = 1/c)
    #     Actually: ρ_k = 1/(2c/d² · c - ρ_{k-1})
    #
    # Simpler parametrization via the standard recurrence:
    #   Let θ = d/c (relative half-width)
    #   σ = c (shift)
    #   Z₀ = (1/c)·B
    #   R_k = B - A·Z_k
    #   Z_{k+1} = Z_k + ω_k · ((1/c)·R_k + (Z_k - Z_{k-1}) · something)
    #
    # We use the form from Saad's "Iterative Methods" (Algorithm 12.1):
    #   d_param = (ℓ_max - ℓ_min)/2, c_param = (ℓ_max + ℓ_min)/2
    #   α = 1/c_param
    #   Z₀ = α B
    #   rho_old = 1
    #   for k = 1, 2, ...:
    #     rho_new = 1 / (1 - (d/2c)² · rho_old)    ... no, the standard form:
    #
    # Let me use the cleanest known form. Define:
    #   σ = c = (ℓ_max+ℓ_min)/2
    #   δ = d/σ = (ℓ_max-ℓ_min)/(ℓ_max+ℓ_min)
    #
    # Iteration:
    #   Z₀ = (1/σ)·B
    #   ρ₁ = 2/δ²    ... actually let me use the "omega" form:

    # Clean implementation of Chebyshev iteration for Ax=b:
    # (Reference: Golub & Van Loan, 4th ed., Algorithm 11.2.6)
    #
    # Let α_opt = 2/(ℓ_max+ℓ_min), β = (ℓ_max-ℓ_min)/(ℓ_max+ℓ_min)
    # Z₀ = α_opt·B
    # Z₁ = Z₀ + ω₁·α_opt·(B - A·Z₀) + (ω₁-1)·(Z₀ - 0)
    #     where ω₁ = 2/(2 - β²), and 0 is Z_{-1} = 0
    # Z_{k+1} = Z_k + ω_k·α_opt·(B - A·Z_k) + (ω_k-1)·(Z_k - Z_{k-1})
    #     where ω_k = 1/(1 - β²/4·ω_{k-1})

    beta_ratio = d / c  # = (ℓ_max-ℓ_min)/(ℓ_max+ℓ_min)
    beta_sq = beta_ratio * beta_ratio

    # ω for step 1
    omega = 2.0 / (2.0 - beta_sq)

    # R₀ = B - A·Z₀ (already have AZ = A·Z₀)
    # Z₁ = Z₀ + ω·α·R₀ + (ω-1)·Z₀   (since Z_{-1} = 0)
    #     = ω·Z₀ + ω·α·R₀
    #     = ω·(Z₀ + α·(B - A·Z₀))
    #     = ω·(Z₀ + α·B - α·AZ)

    # ws.Z_curr will become Z₁
    # ws.Z_prev is Z₀
    # Use AZ as scratch for R₀ = B - AZ
    ws.AZ.mul_(-1.0)
    ws.AZ.add_(B)  # AZ now holds R₀

    # Z₁ = Z₀ + ω·α·R₀ + (ω-1)·(Z₀ - 0)
    # Z₁ = ω·Z₀ + ω·α·R₀
    ws.Z_curr.mul_(omega)
    ws.Z_curr.add_(ws.AZ, alpha=omega * alpha)

    iters_used = 2

    if tol is not None:
        _matmul_into(A_norm, ws.Z_curr, ws.AZ)
        ws.AZ.mul_(-1.0)
        ws.AZ.add_(B)
        r_norm = float(torch.linalg.matrix_norm(ws.AZ, ord="fro").max().item())
        if b_norm is None:
            b_norm = float(
                torch.linalg.matrix_norm(B, ord="fro").max().clamp_min(1e-30).item()
            )
        if r_norm / b_norm <= float(tol):
            return ws.Z_curr, ws, iters_used

    # Steps 2..max_iter-1
    for k in range(2, max_iter):
        omega = 1.0 / (1.0 - (beta_sq / 4.0) * omega)

        # R_k = B - A·Z_curr
        _matmul_into(A_norm, ws.Z_curr, ws.AZ)
        ws.AZ.mul_(-1.0)
        ws.AZ.add_(B)  # AZ now = R_k

        # Z_{k+1} = Z_k + ω·α·R_k + (ω-1)·(Z_k - Z_{k-1})
        # Rewrite: Z_{k+1} = ω·Z_k + ω·α·R_k + (1-ω)·Z_{k-1}
        # Use Z_prev as scratch for Z_{k+1}, then swap

        # Z_new = (1-ω)·Z_prev + ω·Z_curr + ω·α·R_k
        ws.Z_prev.mul_(1.0 - omega)
        ws.Z_prev.add_(ws.Z_curr, alpha=omega)
        ws.Z_prev.add_(ws.AZ, alpha=omega * alpha)

        # Swap: Z_prev becomes Z_curr, Z_curr becomes Z_new
        ws.Z_curr, ws.Z_prev = ws.Z_prev, ws.Z_curr

        iters_used = k + 1

        # Check convergence (optional, costs 1 extra GEMM)
        if tol is not None and (k % 3 == 0 or k == max_iter - 1):
            _matmul_into(A_norm, ws.Z_curr, ws.AZ)
            ws.AZ.mul_(-1.0)
            ws.AZ.add_(B)
            r_norm = float(torch.linalg.matrix_norm(ws.AZ, ord="fro").max().item())
            if b_norm is None:
                b_norm = float(
                    torch.linalg.matrix_norm(B, ord="fro").max().clamp_min(1e-30).item()
                )
            if r_norm / b_norm <= float(tol):
                break

    return ws.Z_curr, ws, iters_used
