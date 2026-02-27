import math

import pytest
import torch

from fast_iroot.precond import precond_gram_dual_spd, precond_gram_spd, precond_spd


def _make_spd(n: int = 32, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, n, generator=g)
    A = (X @ X.mT) / float(n)
    A.diagonal().add_(1e-2)
    return A


@pytest.mark.parametrize("mode", ["jacobi", "ruiz"])
def test_precond_spd_new_modes_finite(mode: str):
    A = _make_spd()
    A_norm, stats = precond_spd(A, mode=mode, l_target=0.05, ruiz_iters=2)

    assert A_norm.shape == A.shape
    assert torch.isfinite(A_norm).all()
    assert torch.allclose(A_norm, A_norm.mT, atol=1e-5, rtol=1e-5)
    assert stats.rho_proxy > 0.0
    assert stats.kappa_proxy > 0.0


def test_precond_spd_ruiz_iters_validation():
    A = _make_spd()
    with pytest.raises(ValueError, match="ruiz_iters"):
        precond_spd(A, mode="ruiz", ruiz_iters=0)


def test_precond_spd_can_skip_rho_proxy():
    A = _make_spd()
    A_norm, stats = precond_spd(
        A, mode="jacobi", l_target=0.05, compute_rho_proxy=False
    )
    assert A_norm.shape == A.shape
    assert stats.gersh_lo > 0.0
    assert stats.kappa_proxy > 0.0
    assert math.isnan(stats.rho_proxy)


def test_precond_gram_col_norm_matches_jacobi_plus_none():
    g = torch.Generator().manual_seed(123)
    G = torch.randn(96, 24, generator=g)
    A = G.mT @ G

    A_jac, stats_jac = precond_spd(
        A,
        mode="jacobi",
        l_target=0.05,
        ridge_rel=0.0,
    )
    A_gram, stats_gram = precond_gram_spd(
        G,
        gram_mode="col-norm",
        mode="none",
        l_target=0.05,
        ridge_rel=0.0,
    )

    assert torch.allclose(A_gram, A_jac, atol=1e-5, rtol=2e-4)
    assert stats_gram.gersh_lo == pytest.approx(stats_jac.gersh_lo, rel=1e-4, abs=1e-6)
    assert stats_gram.rho_proxy == pytest.approx(
        stats_jac.rho_proxy, rel=1e-4, abs=1e-6
    )


def test_precond_gram_spd_batched_output_shape():
    g = torch.Generator().manual_seed(7)
    G = torch.randn(2, 80, 20, generator=g)
    A_norm, stats = precond_gram_spd(G, gram_mode="col-norm", mode="aol")

    assert A_norm.shape == (2, 20, 20)
    assert torch.isfinite(A_norm).all()
    assert stats.kappa_proxy > 0.0


def test_precond_gram_spd_validation():
    G = torch.randn(32, 8)
    with pytest.raises(ValueError, match="unknown gram preconditioner"):
        precond_gram_spd(G, gram_mode="bad")

    G[:, 0] = 0.0
    with pytest.raises(ValueError, match="non-zero column norms"):
        precond_gram_spd(G, gram_mode="col-norm")


def test_precond_gram_dual_row_norm_matches_jacobi_plus_none():
    g = torch.Generator().manual_seed(124)
    G = torch.randn(96, 24, generator=g)
    A_dual = G @ G.mT

    A_jac, stats_jac = precond_spd(
        A_dual,
        mode="jacobi",
        l_target=0.05,
        ridge_rel=0.0,
    )
    A_dual_norm, stats_dual = precond_gram_dual_spd(
        G,
        gram_mode="row-norm",
        mode="none",
        l_target=0.05,
        ridge_rel=0.0,
    )

    assert torch.allclose(A_dual_norm, A_jac, atol=1e-5, rtol=2e-4)
    assert stats_dual.gersh_lo == pytest.approx(stats_jac.gersh_lo, rel=1e-4, abs=1e-6)
    assert stats_dual.rho_proxy == pytest.approx(
        stats_jac.rho_proxy, rel=1e-4, abs=1e-6
    )


def test_precond_gram_dual_spd_validation():
    G = torch.randn(32, 8)
    with pytest.raises(ValueError, match="unknown dual gram preconditioner"):
        precond_gram_dual_spd(G, gram_mode="bad")

    G[0, :] = 0.0
    with pytest.raises(ValueError, match="non-zero row norms"):
        precond_gram_dual_spd(G, gram_mode="row-norm")
