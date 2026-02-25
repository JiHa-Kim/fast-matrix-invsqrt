import pytest
import torch

from fast_iroot import (
    apply_inverse_proot_chebyshev,
    inverse_proot_pe_quadratic_uncoupled,
    _quad_coeffs,
    build_pe_schedules,
    precond_spd,
)
from matrix_iroot import _spd_from_eigs


def get_test_matrix(n: int = 128, case: str = "gaussian") -> torch.Tensor:
    torch.manual_seed(42)
    generator = torch.Generator().manual_seed(42)
    device = torch.device("cpu")

    if case == "gaussian":
        X = torch.randn(n, n, generator=generator)
        A = (X @ X.mT) / n
        A.diagonal().add_(1e-2)
    elif case == "illcond":
        e = torch.logspace(0, -6, steps=n)
        A = _spd_from_eigs(e, device, torch.float32, generator)
    else:
        raise ValueError(f"Unknown case: {case}")
    return A


@pytest.mark.parametrize("p", [1, 2, 4])
@pytest.mark.parametrize("n,k", [(128, 4), (256, 16)])
@pytest.mark.parametrize("case", ["gaussian", "illcond"])
def test_apply_inverse_proot_chebyshev(p, n, k, case):
    A = get_test_matrix(n, case=case)

    # Precond
    l_target = 0.05
    A_norm, _ = precond_spd(A, mode="frob", l_target=l_target)

    # Random RHS
    torch.manual_seed(123)
    B = torch.randn(n, k)

    # Baseline: Inverse PE Uncoupled then matmul
    # use auto PE to match spectral conditioning
    pe_quad_t, _ = build_pe_schedules(
        l_target=l_target,
        device=A_norm.device,
        coeff_mode="auto",
        coeff_seed=0,
        coeff_safety=1.0,
        coeff_no_final_safety=False,
        p_val=p,
    )
    pe_quad_coeffs = _quad_coeffs(pe_quad_t)
    X_pe, _ = inverse_proot_pe_quadratic_uncoupled(
        A_norm, abc_t=pe_quad_coeffs, p_val=p
    )
    Z_expected = X_pe @ B

    # Test: Chebyshev Apply
    # Note: For strict alignment we might need high degree
    degree = 48
    Z_cheb, _ = apply_inverse_proot_chebyshev(
        A=A_norm,
        B=B,
        p_val=p,
        degree=degree,
        l_min=l_target,  # Since precond_spd maps eigs to [l_target, 1]
    )

    # They shouldn't match *exactly* because they are two different polynomial approximations
    # (one is PE quad composed, one is Chebyshev minimax)
    # But they both approximate A^{-1/p} B.
    # Check relative difference is small
    rel_diff = torch.linalg.matrix_norm(Z_cheb - Z_expected) / torch.linalg.matrix_norm(
        Z_expected
    )

    # The error depends on the conditioning and the degree.
    # For testing, we just check they are roughly similar (within 5%)
    assert rel_diff < 0.05, (
        f"Chebyshev solve differs too much from PE solve! Rel diff: {rel_diff}"
    )


def test_apply_inverse_proot_chebyshev_validation():
    A = torch.randn(10, 10)
    B = torch.randn(10, 2)
    # Require p > 0
    with pytest.raises(ValueError):
        apply_inverse_proot_chebyshev(A, B, p_val=0, degree=10, l_min=0.1)

    # Rectangular A
    A_rect = torch.randn(10, 8)
    with pytest.raises(ValueError):
        apply_inverse_proot_chebyshev(A_rect, B, p_val=2, degree=10, l_min=0.1)

    # Shape mismatch A and B
    B_bad = torch.randn(8, 2)
    with pytest.raises(ValueError):
        apply_inverse_proot_chebyshev(A, B_bad, p_val=2, degree=10, l_min=0.1)
