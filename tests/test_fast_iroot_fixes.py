import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pytest
from fast_iroot.utils import _addmm_into
from fast_iroot.apply import (
    DualGramInverseApplyWorkspace,
    GramInverseApplyWorkspace,
    apply_inverse_root,
    apply_inverse_root_auto,
    apply_inverse_root_gram_rhs_spd,
    apply_inverse_root_gram_spd,
    apply_inverse_sqrt_gram_spd,
    apply_inverse_sqrt_non_spd,
    apply_inverse_sqrt_spd,
)
from fast_iroot.coeffs import build_pe_schedules
from fast_iroot.coupled import (
    inverse_proot_pe_quadratic_coupled,
    inverse_sqrt_pe_quadratic,
    inverse_solve_pe_quadratic_coupled,
)
from fast_iroot.metrics import (
    compute_quality_stats,
    exact_inverse_proot,
    iroot_relative_error,
    isqrt_relative_error,
)
from fast_iroot.precond import precond_gram_dual_spd, precond_gram_spd, precond_spd
from fast_iroot.uncoupled import inverse_proot_pe_quadratic_uncoupled


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


def test_inverse_solve_online_early_stop_fro_matches_single_step():
    n = 8
    torch.manual_seed(124)
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1
    M = torch.randn(n, n // 2)

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
        online_stop_metric="fro",
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

    with pytest.raises(ValueError, match="online_stop_metric"):
        inverse_solve_pe_quadratic_coupled(
            A,
            M,
            abc_t=abc_t,
            p_val=2,
            online_stop_metric="bad",
        )

    with pytest.raises(ValueError, match="online_stop_check_every"):
        inverse_solve_pe_quadratic_coupled(
            A,
            M,
            abc_t=abc_t,
            p_val=2,
            online_stop_check_every=0,
        )


def test_inverse_solve_terminal_tail_steps_validation():
    A = torch.randn(6, 6)
    A = (A @ A.mT) / 6 + torch.eye(6) * 0.1
    M = torch.randn(6, 3)
    abc_t = [(1.5, -0.5, 0.0)]

    with pytest.raises(ValueError, match="terminal_tail_steps"):
        inverse_solve_pe_quadratic_coupled(
            A,
            M,
            abc_t=abc_t,
            p_val=2,
            terminal_tail_steps=-1,
        )


def test_post_correction_validation():
    n = 6
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1
    M = torch.randn(n, 3)
    abc_t = [(1.5, -0.5, 0.0)]

    with pytest.raises(ValueError, match="post_correction_steps"):
        inverse_solve_pe_quadratic_coupled(
            A,
            M,
            abc_t=abc_t,
            p_val=2,
            post_correction_steps=-1,
        )

    with pytest.raises(ValueError, match="post_correction_order"):
        inverse_solve_pe_quadratic_coupled(
            A,
            M,
            abc_t=abc_t,
            p_val=2,
            post_correction_order=3,
        )

    with pytest.raises(ValueError, match="supports p_val in \\{2,4\\}"):
        inverse_solve_pe_quadratic_coupled(
            A,
            M,
            abc_t=abc_t,
            p_val=3,
            post_correction_steps=1,
        )

    with pytest.raises(ValueError, match="assume_spd=True"):
        inverse_solve_pe_quadratic_coupled(
            A,
            M,
            abc_t=abc_t,
            p_val=2,
            assume_spd=False,
            symmetrize_Y=False,
            post_correction_steps=1,
        )


def test_post_correction_tail_improves_p4_apply():
    torch.manual_seed(1)
    n = 10
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.2
    M = torch.randn(n, 4)
    abc_t = [(1.25, -0.25, 0.0)]

    Z_base, _ = inverse_solve_pe_quadratic_coupled(
        A,
        M,
        abc_t=abc_t,
        p_val=4,
        post_correction_steps=0,
    )
    Z_tail, _ = inverse_solve_pe_quadratic_coupled(
        A,
        M,
        abc_t=abc_t,
        p_val=4,
        post_correction_steps=1,
        post_correction_order=2,
    )
    X_ref = exact_inverse_proot(A, p_val=4)
    Z_ref = X_ref @ M

    err_base = torch.linalg.matrix_norm(Z_base - Z_ref) / torch.linalg.matrix_norm(Z_ref)
    err_tail = torch.linalg.matrix_norm(Z_tail - Z_ref) / torch.linalg.matrix_norm(Z_ref)
    assert float(err_tail) < float(err_base)


def test_inverse_solve_affine_step_matches_manual_apply():
    n = 7
    torch.manual_seed(321)
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1
    M = torch.randn(n, 3)
    a, b, c = (1.5, -0.5, 0.0)

    Z, _ = inverse_solve_pe_quadratic_coupled(
        A,
        M,
        abc_t=[(a, b, c)],
        p_val=4,
        terminal_last_step=True,
    )

    B = b * A.clone()
    B.diagonal().add_(a)
    Z_ref = B @ M

    assert torch.allclose(Z, Z_ref, atol=1e-6, rtol=1e-6)


def test_inverse_solve_quadratic_terminal_step_matches_manual_apply():
    n = 7
    torch.manual_seed(322)
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1
    M = torch.randn(n, 3)
    a, b, c = (1.1, -0.2, 0.05)

    Z, _ = inverse_solve_pe_quadratic_coupled(
        A,
        M,
        abc_t=[(a, b, c)],
        p_val=2,
        terminal_last_step=True,
    )

    A2 = A @ A
    B = c * A2
    B.add_(A, alpha=b)
    B.diagonal().add_(a)
    Z_ref = B @ M

    assert torch.allclose(Z, Z_ref, atol=1e-6, rtol=1e-6)


def test_inverse_solve_terminal_tail_steps_two_freezes_last_two_steps():
    n = 7
    torch.manual_seed(323)
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1
    M = torch.randn(n, 3)
    abc_t = [(1.4, -0.4, 0.0), (1.3, -0.3, 0.0)]

    Z, _ = inverse_solve_pe_quadratic_coupled(
        A,
        M,
        abc_t=abc_t,
        p_val=2,
        terminal_last_step=True,
        terminal_tail_steps=2,
    )

    def _affine_step(a: float, b: float) -> torch.Tensor:
        B = b * A.clone()
        B.diagonal().add_(a)
        return B

    B0 = _affine_step(*abc_t[0][:2])
    B1 = _affine_step(*abc_t[1][:2])
    Z_ref = B1 @ (B0 @ M)
    assert torch.allclose(Z, Z_ref, atol=1e-6, rtol=1e-6)


def test_inverse_solve_terminal_tail_steps_zero_matches_no_terminal_shortcut():
    n = 8
    torch.manual_seed(324)
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1
    M = torch.randn(n, 4)
    abc_t = [(1.5, -0.5, 0.0), (1.3, -0.3, 0.0)]

    Z_zero, _ = inverse_solve_pe_quadratic_coupled(
        A,
        M,
        abc_t=abc_t,
        p_val=2,
        terminal_last_step=True,
        terminal_tail_steps=0,
    )
    Z_full, _ = inverse_solve_pe_quadratic_coupled(
        A,
        M,
        abc_t=abc_t,
        p_val=2,
        terminal_last_step=False,
    )
    assert torch.allclose(Z_zero, Z_full, atol=1e-5, rtol=1e-5)


def test_apply_inverse_root_auto_direct_matches_apply_inverse_root():
    n = 10
    torch.manual_seed(999)
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1
    M = torch.randn(n, 4)
    abc_t = [(1.5, -0.5, 0.0), (1.25, -0.25, 0.0)]

    Z_ref, _ = apply_inverse_root(A, M, abc_t=abc_t, p_val=2)
    Z_auto, _ = apply_inverse_root_auto(A, M, abc_t=abc_t, p_val=2, strategy="auto")

    assert torch.allclose(Z_auto, Z_ref, atol=1e-5, rtol=1e-5)


def test_apply_inverse_root_auto_materialize_matches_manual():
    n = 9
    torch.manual_seed(111)
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1
    M = torch.randn(n, n)
    abc_t = [(1.5, -0.5, 0.0), (1.25, -0.25, 0.0)]

    X_ref, _ = inverse_proot_pe_quadratic_coupled(A, abc_t=abc_t, p_val=2)
    Z_ref = X_ref @ M

    Z_auto, _ = apply_inverse_root_auto(
        A,
        M,
        abc_t=abc_t,
        p_val=2,
        strategy="materialize-root",
        expected_reuse=4,
    )
    assert torch.allclose(Z_auto, Z_ref, atol=1e-5, rtol=1e-5)


def test_apply_inverse_root_auto_strategy_validation():
    A = torch.eye(4)
    M = torch.eye(4)
    abc_t = [(1.5, -0.5, 0.0)]

    with pytest.raises(ValueError, match="Unknown strategy"):
        apply_inverse_root_auto(A, M, abc_t=abc_t, p_val=2, strategy="bad")

    with pytest.raises(ValueError, match="expected_reuse"):
        apply_inverse_root_auto(A, M, abc_t=abc_t, p_val=2, expected_reuse=0)


def test_coupled_non_spd_uses_general_y_update():
    torch.manual_seed(2026)
    n = 5
    A = torch.randn(n, n, dtype=torch.float32)
    A = A + 0.25 * torch.eye(n, dtype=A.dtype)
    abc_t = [(1.1, -0.2, 0.05)]

    _, ws = inverse_proot_pe_quadratic_coupled(
        A,
        abc_t=abc_t,
        p_val=3,
        symmetrize_Y=False,
        terminal_last_step=False,
        assume_spd=False,
    )

    a, b, c = abc_t[0]
    A2 = A @ A
    B = b * A + c * A2
    B.diagonal().add_(a)
    Y_ref = (B @ B @ B) @ A
    assert torch.allclose(ws.Y, Y_ref, atol=1e-5, rtol=1e-5)


def test_non_spd_requires_symmetrize_off():
    A = torch.randn(4, 4)
    M = torch.randn(4, 3)
    abc_t = [(1.2, -0.2, 0.0)]

    with pytest.raises(ValueError, match="assume_spd"):
        inverse_proot_pe_quadratic_coupled(
            A,
            abc_t=abc_t,
            p_val=2,
            symmetrize_Y=True,
            assume_spd=False,
        )

    with pytest.raises(ValueError, match="assume_spd"):
        inverse_solve_pe_quadratic_coupled(
            A,
            M,
            abc_t=abc_t,
            p_val=2,
            symmetrize_Y=True,
            assume_spd=False,
        )

    with pytest.raises(ValueError, match="assume_spd"):
        inverse_proot_pe_quadratic_uncoupled(
            A,
            abc_t=abc_t,
            p_val=2,
            symmetrize_X=True,
            assume_spd=False,
        )

    with pytest.raises(ValueError, match="SPD-only"):
        inverse_sqrt_pe_quadratic(
            A,
            abc_t=abc_t,
            symmetrize_Y=False,
            assume_spd=False,
        )


def test_p1_auto_disables_spd_assumptions():
    torch.manual_seed(1234)
    A = torch.randn(5, 5) + 0.2 * torch.eye(5)
    M = torch.randn(5, 2)
    abc_t = [(1.3, -0.3, 0.0)]

    Z_auto, _ = inverse_solve_pe_quadratic_coupled(
        A,
        M,
        abc_t=abc_t,
        p_val=1,
        symmetrize_Y=True,   # should be ignored for p=1
        assume_spd=True,     # should be ignored for p=1
    )
    Z_ref, _ = inverse_solve_pe_quadratic_coupled(
        A,
        M,
        abc_t=abc_t,
        p_val=1,
        symmetrize_Y=False,
        assume_spd=False,
    )
    assert torch.allclose(Z_auto, Z_ref, atol=1e-6, rtol=1e-6)


def test_nonspd_adaptive_p1_stable_case_matches_baseline():
    torch.manual_seed(5678)
    n = 10
    A = torch.eye(n) + 0.08 * torch.randn(n, n)
    B = torch.randn(n, 3)
    abc_t = [(1.5, -0.5, 0.0), (1.25, -0.25, 0.0)]

    Z_base, _ = inverse_solve_pe_quadratic_coupled(
        A,
        B,
        abc_t=abc_t,
        p_val=1,
        symmetrize_Y=False,
        assume_spd=False,
        nonspd_adaptive=False,
    )
    Z_adapt, _ = inverse_solve_pe_quadratic_coupled(
        A,
        B,
        abc_t=abc_t,
        p_val=1,
        symmetrize_Y=False,
        assume_spd=False,
        nonspd_adaptive=True,
    )
    base_res = torch.linalg.matrix_norm(A @ Z_base - B) / torch.linalg.matrix_norm(B)
    adapt_res = torch.linalg.matrix_norm(A @ Z_adapt - B) / torch.linalg.matrix_norm(B)
    assert torch.isfinite(base_res)
    assert torch.isfinite(adapt_res)
    assert float(adapt_res) <= 1.5 * float(base_res)


def test_nonspd_adaptive_validation():
    A = torch.eye(4)
    B = torch.randn(4, 2)
    abc_t = [(1.5, -0.5, 0.0)]

    with pytest.raises(ValueError, match="nonspd_adaptive_resid_tol"):
        inverse_solve_pe_quadratic_coupled(
            A,
            B,
            abc_t=abc_t,
            p_val=1,
            assume_spd=False,
            symmetrize_Y=False,
            nonspd_adaptive=True,
            nonspd_adaptive_resid_tol=0.0,
        )

    with pytest.raises(ValueError, match="nonspd_adaptive_growth_tol"):
        inverse_solve_pe_quadratic_coupled(
            A,
            B,
            abc_t=abc_t,
            p_val=1,
            assume_spd=False,
            symmetrize_Y=False,
            nonspd_adaptive=True,
            nonspd_adaptive_growth_tol=0.99,
        )

    with pytest.raises(ValueError, match="nonspd_adaptive_check_every"):
        inverse_solve_pe_quadratic_coupled(
            A,
            B,
            abc_t=abc_t,
            p_val=1,
            assume_spd=False,
            symmetrize_Y=False,
            nonspd_adaptive=True,
            nonspd_adaptive_check_every=0,
        )

    with pytest.raises(ValueError, match="nonspd_safe_fallback_tol"):
        inverse_solve_pe_quadratic_coupled(
            A,
            B,
            abc_t=abc_t,
            p_val=1,
            assume_spd=False,
            symmetrize_Y=False,
            nonspd_adaptive=True,
            nonspd_safe_fallback_tol=0.0,
        )

    with pytest.raises(ValueError, match="nonspd_safe_early_y_tol"):
        inverse_solve_pe_quadratic_coupled(
            A,
            B,
            abc_t=abc_t,
            p_val=1,
            assume_spd=False,
            symmetrize_Y=False,
            nonspd_safe_early_y_tol=0.0,
        )


def test_nonspd_safe_fallback_matches_solve_when_triggered():
    torch.manual_seed(2027)
    n = 9
    A = torch.eye(n) + 0.2 * torch.randn(n, n)
    B = torch.randn(n, 3)
    abc_t = [(1.25, -0.25, 0.0)]  # intentionally short schedule

    Z_fast, _ = inverse_solve_pe_quadratic_coupled(
        A,
        B,
        abc_t=abc_t,
        p_val=1,
        assume_spd=False,
        symmetrize_Y=False,
        nonspd_safe_fallback_tol=None,
    )
    Z_safe, _ = inverse_solve_pe_quadratic_coupled(
        A,
        B,
        abc_t=abc_t,
        p_val=1,
        assume_spd=False,
        symmetrize_Y=False,
        nonspd_safe_fallback_tol=1e-12,
    )
    Z_ref = torch.linalg.solve(A, B)

    err_fast = torch.linalg.matrix_norm(Z_fast - Z_ref) / torch.linalg.matrix_norm(Z_ref)
    err_safe = torch.linalg.matrix_norm(Z_safe - Z_ref) / torch.linalg.matrix_norm(Z_ref)
    assert float(err_safe) < 1e-6
    assert float(err_safe) <= float(err_fast)


def test_nonspd_safe_early_guard_triggers_fallback():
    torch.manual_seed(2028)
    n = 8
    A = torch.eye(n) + 0.15 * torch.randn(n, n)
    B = torch.randn(n, 2)
    abc_t = [(1.25, -0.25, 0.0), (1.10, -0.10, 0.0)]

    Z_fast, _ = inverse_solve_pe_quadratic_coupled(
        A,
        B,
        abc_t=abc_t,
        p_val=1,
        assume_spd=False,
        symmetrize_Y=False,
        nonspd_safe_fallback_tol=None,
    )
    Z_early, _ = inverse_solve_pe_quadratic_coupled(
        A,
        B,
        abc_t=abc_t,
        p_val=1,
        assume_spd=False,
        symmetrize_Y=False,
        nonspd_safe_fallback_tol=1e6,
        nonspd_safe_early_y_tol=1e-8,
    )
    Z_ref = torch.linalg.solve(A, B)

    err_fast = torch.linalg.matrix_norm(Z_fast - Z_ref) / torch.linalg.matrix_norm(Z_ref)
    err_early = torch.linalg.matrix_norm(Z_early - Z_ref) / torch.linalg.matrix_norm(
        Z_ref
    )
    assert float(err_early) < 1e-6
    assert float(err_early) <= float(err_fast)


def test_non_spd_metrics_and_exact_inverse_for_p1():
    torch.manual_seed(77)
    A = torch.randn(6, 6)
    A = A + 0.5 * torch.eye(6)  # Keep reasonably well-conditioned/invertible
    X = exact_inverse_proot(A, p_val=1, assume_spd=False)
    X_ref = torch.linalg.inv(A)

    assert torch.allclose(X, X_ref, atol=1e-6, rtol=1e-5)

    rel = iroot_relative_error(X, A, p_val=1, assume_spd=False)
    assert float(rel.max()) < 1e-6

    q = compute_quality_stats(
        X,
        A,
        power_iters=2,
        mv_samples=2,
        hard_probe_iters=2,
        p_val=1,
        assume_spd=False,
    )
    assert q.residual_fro < 1e-5
    assert math.isnan(q.hard_dir_err)


def test_apply_inverse_sqrt_spd_wrapper_matches_general_api():
    torch.manual_seed(88)
    n = 9
    A = torch.randn(n, n)
    A = (A @ A.mT) / n + torch.eye(n) * 0.1
    M = torch.randn(n, 3)
    abc_t = [(1.4, -0.4, 0.0), (1.2, -0.2, 0.0)]

    Z_wrapped, _ = apply_inverse_sqrt_spd(A, M, abc_t=abc_t)
    Z_generic, _ = apply_inverse_root(
        A, M, abc_t=abc_t, p_val=2, assume_spd=True, symmetrize_Y=True
    )
    assert torch.allclose(Z_wrapped, Z_generic, atol=1e-6, rtol=1e-6)


def test_apply_inverse_sqrt_non_spd_wrapper_matches_general_api():
    torch.manual_seed(89)
    n = 8
    A = torch.randn(n, n)
    A = A + 0.3 * torch.eye(n)
    M = torch.randn(n, 4)
    abc_t = [(1.4, -0.4, 0.0), (1.2, -0.2, 0.0)]

    Z_wrapped, _ = apply_inverse_sqrt_non_spd(A, M, abc_t=abc_t)
    Z_generic, _ = apply_inverse_root(
        A, M, abc_t=abc_t, p_val=2, assume_spd=False, symmetrize_Y=False
    )
    assert torch.allclose(Z_wrapped, Z_generic, atol=1e-6, rtol=1e-6)


def test_apply_inverse_sqrt_gram_spd_wrapper_matches_manual_path():
    torch.manual_seed(90)
    m, n, k = 14, 7, 3
    G = torch.randn(m, n)
    M = torch.randn(n, k)
    abc_t = [(1.4, -0.4, 0.0)]

    Z_wrap, _, stats_wrap = apply_inverse_sqrt_gram_spd(
        G,
        M,
        abc_t=abc_t,
        gram_mode="col-norm",
        precond_mode="none",
        l_target=0.05,
    )

    A_norm, stats_ref = precond_gram_spd(
        G,
        gram_mode="col-norm",
        mode="none",
        l_target=0.05,
    )
    Z_ref, _ = apply_inverse_sqrt_spd(A_norm, M, abc_t=abc_t)

    assert torch.allclose(Z_wrap, Z_ref, atol=1e-6, rtol=1e-6)
    assert stats_wrap.rho_proxy == pytest.approx(stats_ref.rho_proxy, rel=1e-6)
    assert stats_wrap.gersh_lo == pytest.approx(stats_ref.gersh_lo, rel=1e-6)
    assert stats_wrap.kappa_proxy == pytest.approx(stats_ref.kappa_proxy, rel=1e-6)


def test_apply_inverse_root_gram_spd_wrapper_matches_manual_path_p4():
    torch.manual_seed(91)
    m, n, k = 18, 9, 4
    G = torch.randn(m, n)
    M = torch.randn(n, k)
    abc_t = [(1.25, -0.25, 0.0), (1.1, -0.1, 0.0)]

    Z_wrap, _, stats_wrap = apply_inverse_root_gram_spd(
        G,
        M,
        abc_t=abc_t,
        p_val=4,
        strategy="direct-solve",
        gram_mode="col-norm",
        precond_mode="none",
        l_target=0.05,
    )

    A_norm, stats_ref = precond_gram_spd(
        G,
        gram_mode="col-norm",
        mode="none",
        l_target=0.05,
    )
    Z_ref, _ = apply_inverse_root(A_norm, M, abc_t=abc_t, p_val=4, assume_spd=True)

    assert torch.allclose(Z_wrap, Z_ref, atol=1e-6, rtol=1e-6)
    assert stats_wrap.rho_proxy == pytest.approx(stats_ref.rho_proxy, rel=1e-6)
    assert stats_wrap.gersh_lo == pytest.approx(stats_ref.gersh_lo, rel=1e-6)
    assert stats_wrap.kappa_proxy == pytest.approx(stats_ref.kappa_proxy, rel=1e-6)


def test_apply_inverse_root_gram_spd_cache_reuses_and_invalidates():
    torch.manual_seed(92)
    m, n, k = 16, 8, 3
    G = torch.randn(m, n)
    M = torch.randn(n, k)
    abc_t = [(1.4, -0.4, 0.0)]
    ws = GramInverseApplyWorkspace()

    Z1, ws, _ = apply_inverse_root_gram_spd(
        G,
        M,
        abc_t=abc_t,
        p_val=2,
        ws=ws,
        strategy="direct-solve",
        reuse_precond=True,
    )
    Z1_ref = Z1.clone()
    assert ws.A_norm is not None
    cache_ptr_before = int(ws.A_norm.data_ptr())
    cache_ver_before = ws.cache_g_version

    Z2, ws, _ = apply_inverse_root_gram_spd(
        G,
        M,
        abc_t=abc_t,
        p_val=2,
        ws=ws,
        strategy="direct-solve",
        reuse_precond=True,
    )
    assert int(ws.A_norm.data_ptr()) == cache_ptr_before
    assert torch.allclose(Z2, Z1_ref, atol=1e-6, rtol=1e-6)

    G.add_(0.01 * torch.randn_like(G))
    Z3, ws, _ = apply_inverse_root_gram_spd(
        G,
        M,
        abc_t=abc_t,
        p_val=2,
        ws=ws,
        strategy="direct-solve",
        reuse_precond=True,
    )
    assert ws.cache_g_version != cache_ver_before
    assert ws.cache_g_version == int(getattr(G, "_version", -1))
    assert torch.isfinite(Z3).all()


def test_apply_inverse_root_gram_rhs_spd_wrapper_matches_manual_path_p4():
    torch.manual_seed(93)
    m, n, k = 15, 9, 4
    G = torch.randn(m, n)
    B = torch.randn(m, k)
    abc_t = [(1.25, -0.25, 0.0), (1.1, -0.1, 0.0)]

    Z_wrap, _, stats_wrap = apply_inverse_root_gram_rhs_spd(
        G,
        B,
        abc_t=abc_t,
        p_val=4,
        strategy="direct-solve",
        gram_mode="row-norm",
        precond_mode="none",
        l_target=0.05,
    )

    A_dual, stats_ref = precond_gram_dual_spd(
        G,
        gram_mode="row-norm",
        mode="none",
        l_target=0.05,
    )
    U_ref, _ = apply_inverse_root(A_dual, B, abc_t=abc_t, p_val=4, assume_spd=True)
    Z_ref = G.mT @ U_ref

    assert torch.allclose(Z_wrap, Z_ref, atol=1e-6, rtol=1e-6)
    assert stats_wrap.rho_proxy == pytest.approx(stats_ref.rho_proxy, rel=1e-6)
    assert stats_wrap.gersh_lo == pytest.approx(stats_ref.gersh_lo, rel=1e-6)
    assert stats_wrap.kappa_proxy == pytest.approx(stats_ref.kappa_proxy, rel=1e-6)


def test_apply_inverse_root_gram_rhs_spd_cache_reuses_and_invalidates():
    torch.manual_seed(94)
    m, n, k = 14, 7, 3
    G = torch.randn(m, n)
    B = torch.randn(m, k)
    abc_t = [(1.4, -0.4, 0.0)]
    ws = DualGramInverseApplyWorkspace()

    Z1, ws, _ = apply_inverse_root_gram_rhs_spd(
        G,
        B,
        abc_t=abc_t,
        p_val=2,
        ws=ws,
        strategy="direct-solve",
        reuse_precond=True,
    )
    Z1_ref = Z1.clone()
    assert ws.A_norm is not None
    cache_ptr_before = int(ws.A_norm.data_ptr())
    cache_ver_before = ws.cache_g_version

    Z2, ws, _ = apply_inverse_root_gram_rhs_spd(
        G,
        B,
        abc_t=abc_t,
        p_val=2,
        ws=ws,
        strategy="direct-solve",
        reuse_precond=True,
    )
    assert int(ws.A_norm.data_ptr()) == cache_ptr_before
    assert torch.allclose(Z2, Z1_ref, atol=1e-6, rtol=1e-6)

    G.add_(0.01 * torch.randn_like(G))
    Z3, ws, _ = apply_inverse_root_gram_rhs_spd(
        G,
        B,
        abc_t=abc_t,
        p_val=2,
        ws=ws,
        strategy="direct-solve",
        reuse_precond=True,
    )
    assert ws.cache_g_version != cache_ver_before
    assert ws.cache_g_version == int(getattr(G, "_version", -1))
    assert torch.isfinite(Z3).all()
