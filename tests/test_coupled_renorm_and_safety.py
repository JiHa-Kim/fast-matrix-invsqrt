import pytest
import torch

from fast_iroot.coupled import (
    inverse_proot_pe_quadratic_coupled,
    inverse_solve_pe_quadratic_coupled,
)


def test_inverse_solve_nonspd_early_metric_validation():
    A = torch.eye(4)
    B = torch.eye(4)
    with pytest.raises(ValueError, match="nonspd_safe_early_metric"):
        inverse_solve_pe_quadratic_coupled(
            A,
            B,
            abc_t=[(1.0, 0.0, 0.0)],
            p_val=1,
            nonspd_safe_fallback_tol=1e-2,
            nonspd_safe_early_y_tol=0.5,
            nonspd_safe_early_metric="bad",
        )


def test_inverse_solve_nonspd_fro_early_gate_without_adaptive_runs():
    A = torch.eye(4)
    B = torch.eye(4)
    Z_hat, _ = inverse_solve_pe_quadratic_coupled(
        A,
        B,
        abc_t=[(1.0, 0.0, 0.0)],
        p_val=1,
        nonspd_adaptive=False,
        nonspd_safe_fallback_tol=1.0,
        nonspd_safe_early_y_tol=1.0,
        nonspd_safe_early_metric="fro",
    )
    assert torch.isfinite(Z_hat).all()


def test_inverse_solve_renorm_recenters_and_scales_z():
    A = 2.0 * torch.eye(4)
    B = torch.eye(4)
    Z_hat, _ = inverse_solve_pe_quadratic_coupled(
        A,
        B,
        abc_t=[(1.0, 0.0, 0.0)],
        p_val=1,
        terminal_last_step=False,
        terminal_tail_steps=0,
        renorm_every=1,
    )
    assert torch.allclose(Z_hat, 0.5 * torch.eye(4), atol=1e-6)


def test_inverse_proot_renorm_recenters_and_scales_x():
    A = 4.0 * torch.eye(4)
    X_hat, _ = inverse_proot_pe_quadratic_coupled(
        A,
        abc_t=[(1.0, 0.0, 0.0)],
        p_val=2,
        terminal_last_step=False,
        terminal_tail_steps=0,
        renorm_every=1,
    )
    assert torch.allclose(X_hat, 0.5 * torch.eye(4), atol=1e-6)
