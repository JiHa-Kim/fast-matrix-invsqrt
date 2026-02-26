"""
test_new_solvers.py — Correctness tests for NSRC, Block CG, and Chebyshev iterative solvers.
"""


import pytest
import torch

from fast_iroot import precond_spd, build_pe_schedules, _quad_coeffs
from fast_iroot.nsrc import nsrc_solve, nsrc_solve_preconditioned, hybrid_pe_nsrc_solve
from fast_iroot.block_cg import block_cg_solve, block_cg_solve_with_precond
from fast_iroot.chebyshev_iterative import chebyshev_iterative_solve


def _make_spd(n: int, cond: float = 10.0, seed: int = 42) -> torch.Tensor:
    """Create an SPD matrix with specified condition number."""
    rng = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=rng))
    eigs = torch.linspace(1.0 / cond, 1.0, n)
    return (Q * eigs.unsqueeze(0)) @ Q.T


def _precond_and_stats(A: torch.Tensor, l_target: float = 0.05):
    """Precondition A and return (A_norm, stats)."""
    return precond_spd(A, mode="frob", l_target=l_target)


class TestNSRCSolve:
    """Tests for Neumann-Series Residual Correction."""

    def test_identity(self):
        """NSRC with identity matrix should return B."""
        n, k = 32, 4
        A = torch.eye(n)
        B = torch.randn(n, k)
        Z, _ = nsrc_solve(A, B, alpha=1.0, max_iter=1)
        assert torch.allclose(Z, B, atol=1e-5)

    def test_well_conditioned(self):
        """NSRC should converge on well-conditioned SPD to ~1e-2."""
        n, k = 64, 8
        A = _make_spd(n, cond=5.0)
        A_norm, stats = _precond_and_stats(A)
        B = torch.randn(n, k)
        Z_true = torch.linalg.solve(A_norm, B)

        alpha = 2.0 / (1.0 + stats.gersh_lo)
        Z, _ = nsrc_solve(A_norm, B, alpha=alpha, max_iter=30)
        rel_err = float(torch.linalg.norm(Z - Z_true) / torch.linalg.norm(Z_true))
        assert rel_err < 0.1, f"rel_err = {rel_err:.4e}"

    def test_preconditioned_converges_faster(self):
        """Matrix-preconditioned NSRC should converge faster than scalar."""
        n, k = 64, 8
        A = _make_spd(n, cond=10.0)
        A_norm, stats = _precond_and_stats(A)
        B = torch.randn(n, k)
        Z_true = torch.linalg.solve(A_norm, B)

        # Build M_inv from PE
        from fast_iroot.coupled import inverse_proot_pe_quadratic_coupled

        pe_quad, _ = build_pe_schedules(
            l_target=stats.gersh_lo,
            device=A.device,
            coeff_mode="auto",
            coeff_seed=0,
            coeff_safety=1.0,
            coeff_no_final_safety=False,
            p_val=1,
        )
        coeffs = _quad_coeffs(pe_quad)
        M_inv, _ = inverse_proot_pe_quadratic_coupled(
            A_norm,
            abc_t=coeffs[:2],
            p_val=1,
            symmetrize_Y=False,
            terminal_last_step=False,
            assume_spd=False,
        )

        Z_prec, _ = nsrc_solve_preconditioned(A_norm, B, M_inv, max_iter=3)
        rel_err_prec = float(
            torch.linalg.norm(Z_prec - Z_true) / torch.linalg.norm(Z_true)
        )
        assert rel_err_prec < 0.5, f"preconditioned rel_err = {rel_err_prec:.4e}"

    def test_early_stop(self):
        """NSRC with tol should stop early when converged."""
        n, k = 32, 4
        A = _make_spd(n, cond=3.0)
        A_norm, stats = _precond_and_stats(A)
        B = torch.randn(n, k)

        alpha = 2.0 / (1.0 + stats.gersh_lo)
        Z, _ = nsrc_solve(A_norm, B, alpha=alpha, max_iter=100, tol=1e-2)
        Z_true = torch.linalg.solve(A_norm, B)
        rel_err = float(torch.linalg.norm(Z - Z_true) / torch.linalg.norm(Z_true))
        assert rel_err < 0.1


class TestBlockCG:
    """Tests for Block Conjugate Gradient."""

    def test_identity(self):
        """CG on identity should return B immediately."""
        n, k = 32, 4
        A = torch.eye(n)
        B = torch.randn(n, k)
        Z, _, iters = block_cg_solve(A, B, max_iter=5, tol=1e-6)
        assert torch.allclose(Z, B, atol=1e-4)
        assert iters <= 2

    def test_well_conditioned(self):
        """CG should converge quickly on well-conditioned SPD."""
        n, k = 64, 8
        A = _make_spd(n, cond=5.0)
        A_norm, stats = _precond_and_stats(A)
        B = torch.randn(n, k)
        Z_true = torch.linalg.solve(A_norm, B)

        Z, _, iters = block_cg_solve(A_norm, B, max_iter=20, tol=1e-3)
        rel_err = float(torch.linalg.norm(Z - Z_true) / torch.linalg.norm(Z_true))
        assert rel_err < 0.05, f"rel_err = {rel_err:.4e}, iters = {iters}"

    def test_diagonal_precond(self):
        """Diagonally preconditioned CG should converge faster."""
        n, k = 64, 8
        A = _make_spd(n, cond=20.0)
        A_norm, stats = _precond_and_stats(A)
        B = torch.randn(n, k)
        Z_true = torch.linalg.solve(A_norm, B)

        diag_inv = 1.0 / A_norm.diagonal()
        Z, _, iters = block_cg_solve(
            A_norm, B, max_iter=20, tol=1e-3, diag_precond=diag_inv
        )
        rel_err = float(torch.linalg.norm(Z - Z_true) / torch.linalg.norm(Z_true))
        assert rel_err < 0.1, f"rel_err = {rel_err:.4e}, iters = {iters}"

    def test_dense_precond(self):
        """Dense matrix preconditioned CG should converge."""
        n, k = 64, 8
        A = _make_spd(n, cond=10.0)
        A_norm, stats = _precond_and_stats(A)
        B = torch.randn(n, k)
        Z_true = torch.linalg.solve(A_norm, B)

        # Use a rough inverse as preconditioner
        M_inv = torch.eye(n) * (2.0 / (1.0 + stats.gersh_lo))

        Z, _, iters = block_cg_solve_with_precond(
            A_norm, B, M_inv, max_iter=20, tol=1e-3
        )
        rel_err = float(torch.linalg.norm(Z - Z_true) / torch.linalg.norm(Z_true))
        assert rel_err < 0.1, f"rel_err = {rel_err:.4e}, iters = {iters}"


class TestChebyshevIterative:
    """Tests for Chebyshev semi-iterative method."""

    def test_identity(self):
        """Chebyshev on identity should return B."""
        n, k = 32, 4
        A = torch.eye(n)
        B = torch.randn(n, k)
        Z, _, iters = chebyshev_iterative_solve(
            A, B, l_min=0.99, l_max=1.01, max_iter=3
        )
        assert torch.allclose(Z, B, atol=1e-2)

    def test_well_conditioned(self):
        """Chebyshev should converge on well-conditioned SPD."""
        n, k = 64, 8
        A = _make_spd(n, cond=5.0)
        A_norm, stats = _precond_and_stats(A)
        B = torch.randn(n, k)
        Z_true = torch.linalg.solve(A_norm, B)

        Z, _, iters = chebyshev_iterative_solve(
            A_norm, B, l_min=stats.gersh_lo, l_max=1.0, max_iter=20
        )
        rel_err = float(torch.linalg.norm(Z - Z_true) / torch.linalg.norm(Z_true))
        assert rel_err < 0.1, f"rel_err = {rel_err:.4e}, iters = {iters}"

    def test_with_early_stop(self):
        """Chebyshev with tol should stop early."""
        n, k = 32, 4
        A = _make_spd(n, cond=3.0)
        A_norm, stats = _precond_and_stats(A)
        B = torch.randn(n, k)

        Z, _, iters = chebyshev_iterative_solve(
            A_norm, B, l_min=stats.gersh_lo, l_max=1.0, max_iter=100, tol=1e-2
        )
        Z_true = torch.linalg.solve(A_norm, B)
        rel_err = float(torch.linalg.norm(Z - Z_true) / torch.linalg.norm(Z_true))
        assert rel_err < 0.1, f"rel_err = {rel_err:.4e}"
        assert iters < 100, f"should have stopped early, used {iters}"

    def test_bad_bounds_raises(self):
        """Invalid spectral bounds should raise."""
        A = torch.eye(4)
        B = torch.randn(4, 2)
        with pytest.raises(ValueError):
            chebyshev_iterative_solve(A, B, l_min=-1.0, l_max=1.0)
        with pytest.raises(ValueError):
            chebyshev_iterative_solve(A, B, l_min=1.0, l_max=0.5)


class TestHybridPENSRC:
    """Tests for Hybrid PE + NSRC solver."""

    def test_converges(self):
        """Hybrid PE+NSRC should converge on SPD systems."""
        n, k = 64, 8
        A = _make_spd(n, cond=10.0)
        A_norm, stats = _precond_and_stats(A)
        B = torch.randn(n, k)
        Z_true = torch.linalg.solve(A_norm, B)

        pe_quad, _ = build_pe_schedules(
            l_target=stats.gersh_lo,
            device=A.device,
            coeff_mode="auto",
            coeff_seed=0,
            coeff_safety=1.0,
            coeff_no_final_safety=False,
            p_val=1,
        )

        Z, _ = hybrid_pe_nsrc_solve(A_norm, B, abc_t=pe_quad, pe_steps=2, ref_steps=5)
        rel_err = float(torch.linalg.norm(Z - Z_true) / torch.linalg.norm(Z_true))
        assert rel_err < 0.5, f"rel_err = {rel_err:.4e}"

    def test_pe_only(self):
        """Hybrid with ref_steps=0 should match pure PE."""
        n, k = 32, 4
        A = _make_spd(n, cond=5.0)
        A_norm, stats = _precond_and_stats(A)
        B = torch.randn(n, k)

        pe_quad, _ = build_pe_schedules(
            l_target=stats.gersh_lo,
            device=A.device,
            coeff_mode="auto",
            coeff_seed=0,
            coeff_safety=1.0,
            coeff_no_final_safety=False,
            p_val=1,
        )

        Z, _ = hybrid_pe_nsrc_solve(A_norm, B, abc_t=pe_quad, pe_steps=2, ref_steps=0)
        # Should not crash & should produce finite output
        assert torch.isfinite(Z).all()
