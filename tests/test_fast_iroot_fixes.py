import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
from fast_iroot.utils import _addmm_into
from fast_iroot.coupled import (
    inverse_solve_pe_quadratic_coupled,
    inverse_proot_pe_quadratic_coupled,
)
from fast_iroot import build_pe_schedules, precond_spd, apply_inverse_root
from fast_iroot.metrics import isqrt_relative_error, exact_inverse_proot


def test_addmm_into_multibatch_shape():
    torch.manual_seed(42)
    b, k, n, m = 2, 3, 4, 4
    bias = torch.randn(b, k, n, m)
    mat1 = torch.randn(b, k, n, m)
    mat2 = torch.randn(b, k, n, m)
    out = torch.empty_like(bias)

    _addmm_into(bias, mat1, mat2, beta=0.5, alpha=1.2, out=out)
    expected = 0.5 * bias + 1.2 * (mat1 @ mat2)
    assert torch.allclose(out, expected, atol=1e-5)


def test_inverse_solve_dtype_mismatch_raises():
    A_norm = torch.randn(4, 4, dtype=torch.float32)
    M_norm = torch.randn(4, 4, dtype=torch.float64)
    abc_t = [(0.1, 0.2, 0.3)]
    with pytest.raises(ValueError, match="same dtype"):
        inverse_solve_pe_quadratic_coupled(A_norm, M_norm, abc_t)


def test_inverse_solve_device_mismatch_raises():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    A_norm = torch.randn(4, 4, device="cpu")
    M_norm = torch.randn(4, 4, device="cuda")
    abc_t = [(0.1, 0.2, 0.3)]
    with pytest.raises(ValueError, match="same device"):
        inverse_solve_pe_quadratic_coupled(A_norm, M_norm, abc_t)


def test_metrics_shape_mismatch_raises():
    Xhat = torch.randn(4, 4)
    A = torch.randn(5, 5)
    with pytest.raises(ValueError, match="compatible shapes"):
        isqrt_relative_error(Xhat, A)


def test_addmm_into_non_contiguous():
    torch.manual_seed(42)
    b, k, n, m = 2, 3, 4, 4
    bias = torch.randn(b, k, n, m)
    mat1 = torch.randn(b, k, m, n).transpose(-1, -2)
    mat2 = torch.randn(b, k, m, n).transpose(-1, -2)
    out = torch.empty_like(bias)

    assert not mat1.is_contiguous()
    assert not mat2.is_contiguous()

    _addmm_into(bias, mat1, mat2, beta=0.5, alpha=1.2, out=out)
    expected = 0.5 * bias + 1.2 * (mat1 @ mat2)
    assert torch.allclose(out, expected, atol=1e-5)


def test_workspace_reuse_sanity():
    A_norm = torch.randn(4, 4, dtype=torch.float32)
    A_norm = (A_norm @ A_norm.mT) + torch.eye(4) * 0.1
    M_norm = torch.randn(4, 4, dtype=torch.float32)
    abc_t = [(0.1, 0.2, 0.3)]

    Z1, ws = inverse_solve_pe_quadratic_coupled(A_norm, M_norm, abc_t, ws=None)
    Z2, ws_reused = inverse_solve_pe_quadratic_coupled(A_norm, M_norm, abc_t, ws=ws)
    assert ws is ws_reused

    # Test reallocation on different shape
    A_large = torch.randn(5, 5)
    A_large = (A_large @ A_large.mT) + torch.eye(5) * 0.1
    M_large = torch.randn(5, 5)
    Z3, ws_new = inverse_solve_pe_quadratic_coupled(A_large, M_large, abc_t, ws=ws)
    assert ws_new.tmp.shape[-1] == 5


def test_coupled_pe_vs_exact():
    n = 10
    torch.manual_seed(42)
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1

    l_target = 0.1
    A_norm, _ = precond_spd(A, mode="frob", l_target=l_target)

    M = torch.eye(n)

    abc_t, _ = build_pe_schedules(
        l_target=l_target,
        device=A_norm.device,
        p_val=2,
        coeff_mode="auto",
        coeff_seed=0,
        coeff_safety=1.0,
        coeff_no_final_safety=False,
    )
    abc_coeffs = [
        (a, b, c)
        for a, b, c in zip(
            abc_t[:, 0].tolist(), abc_t[:, 1].tolist(), abc_t[:, 2].tolist()
        )
    ]

    Z, _ = inverse_solve_pe_quadratic_coupled(A_norm, M, abc_coeffs, p_val=2)
    Z_exact = exact_inverse_proot(A_norm, p_val=2)

    rel_diff = torch.linalg.matrix_norm(Z - Z_exact) / torch.linalg.matrix_norm(Z_exact)
    assert rel_diff < 0.05


def test_symmetrize_every_validation():
    A = torch.randn(4, 4)
    A = (A @ A.mT) + torch.eye(4) * 0.1
    M = torch.eye(4)
    abc_t = [(1.2, -0.2, 0.0)]

    with pytest.raises(ValueError, match="symmetrize_every"):
        inverse_proot_pe_quadratic_coupled(
            A, abc_t=abc_t, p_val=3, symmetrize_every=0
        )

    with pytest.raises(ValueError, match="symmetrize_every"):
        inverse_solve_pe_quadratic_coupled(
            A, M, abc_t=abc_t, p_val=3, symmetrize_every=0
        )


def test_apply_inverse_root_symmetrize_every_passthrough():
    n = 8
    torch.manual_seed(7)
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1
    A_norm, _ = precond_spd(A, mode="frob", l_target=0.1)
    M = torch.eye(n)

    abc_t, _ = build_pe_schedules(
        l_target=0.1,
        device=A_norm.device,
        p_val=3,
        coeff_mode="tuned",
        coeff_seed=0,
        coeff_safety=1.0,
        coeff_no_final_safety=False,
    )
    abc_coeffs = [
        (a, b, c)
        for a, b, c in zip(
            abc_t[:, 0].tolist(), abc_t[:, 1].tolist(), abc_t[:, 2].tolist()
        )
    ]

    Z1, _ = apply_inverse_root(
        A_norm, M, abc_coeffs, p_val=3, symmetrize_Y=True, symmetrize_every=1
    )
    Z2, _ = apply_inverse_root(
        A_norm, M, abc_coeffs, p_val=3, symmetrize_Y=True, symmetrize_every=2
    )

    assert torch.isfinite(Z1).all()
    assert torch.isfinite(Z2).all()
    assert Z1.shape == Z2.shape == M.shape


def test_inverse_solve_online_early_stop_matches_single_step():
    n = 8
    torch.manual_seed(123)
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1
    M = torch.randn(n, n // 2)

    # Use multiple coefficients but force online stop after first allowed step.
    abc_t = [
        (1.5, -0.5, 0.0),
        (1.4, -0.4, 0.0),
        (1.3, -0.3, 0.0),
    ]

    Z_online, _ = inverse_solve_pe_quadratic_coupled(
        A,
        M,
        abc_t=abc_t,
        p_val=2,
        online_stop_tol=1e9,
        online_min_steps=1,
    )
    Z_single, _ = inverse_solve_pe_quadratic_coupled(
        A,
        M,
        abc_t=[abc_t[0]],
        p_val=2,
    )

    assert torch.allclose(Z_online, Z_single, atol=1e-5, rtol=1e-5)


def test_inverse_solve_online_stop_validation():
    A = torch.randn(6, 6)
    A = (A @ A.mT) / 6 + torch.eye(6) * 0.1
    M = torch.randn(6, 3)
    abc_t = [(1.5, -0.5, 0.0)]

    with pytest.raises(ValueError, match="online_stop_tol"):
        inverse_solve_pe_quadratic_coupled(
            A,
            M,
            abc_t=abc_t,
            p_val=2,
            online_stop_tol=0.0,
        )

    with pytest.raises(ValueError, match="online_min_steps"):
        inverse_solve_pe_quadratic_coupled(
            A,
            M,
            abc_t=abc_t,
            p_val=2,
            online_stop_tol=1e-3,
            online_min_steps=0,
        )
