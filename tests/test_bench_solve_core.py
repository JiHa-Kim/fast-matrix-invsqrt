import torch

from benchmarks.solve.bench_solve_core import (
    _build_solve_runner,
    _can_use_cuda_graph_for_method,
    _naive_newton_preprocess,
    matrix_solve_methods,
)


def test_matrix_solve_methods_p1_excludes_chebyshev_and_keeps_torch_baselines():
    methods = matrix_solve_methods(1)
    assert "Chebyshev-Apply" not in methods
    assert "Torch-Solve" in methods
    assert "Torch-Cholesky-Solve" in methods
    assert "Torch-Cholesky-Solve-ReuseFactor" in methods


def test_matrix_solve_methods_p2_includes_chebyshev():
    methods = matrix_solve_methods(2)
    assert "Chebyshev-Apply" in methods


def test_matrix_solve_methods_p4_includes_chebyshev():
    methods = matrix_solve_methods(4)
    assert "Chebyshev-Apply" in methods


def test_can_use_cuda_graph_for_method_rules():
    dev_cuda = torch.device("cuda")
    dev_cpu = torch.device("cpu")

    assert _can_use_cuda_graph_for_method(
        "PE-Quad-Coupled-Apply",
        use_cuda_graph=True,
        device=dev_cuda,
        online_stop_tol=None,
        cheb_cuda_graph=True,
    )
    assert not _can_use_cuda_graph_for_method(
        "PE-Quad-Coupled-Apply",
        use_cuda_graph=True,
        device=dev_cuda,
        online_stop_tol=1e-3,
        cheb_cuda_graph=True,
    )
    assert _can_use_cuda_graph_for_method(
        "Chebyshev-Apply",
        use_cuda_graph=True,
        device=dev_cuda,
        online_stop_tol=1e-3,
        cheb_cuda_graph=True,
    )
    assert _can_use_cuda_graph_for_method(
        "Chebyshev-Apply",
        use_cuda_graph=False,
        device=dev_cuda,
        online_stop_tol=None,
        cheb_cuda_graph=True,
    )
    assert not _can_use_cuda_graph_for_method(
        "Chebyshev-Apply",
        use_cuda_graph=True,
        device=dev_cuda,
        online_stop_tol=None,
        cheb_cuda_graph=False,
    )
    assert not _can_use_cuda_graph_for_method(
        "Chebyshev-Apply",
        use_cuda_graph=True,
        device=dev_cpu,
        online_stop_tol=None,
        cheb_cuda_graph=True,
    )
    assert not _can_use_cuda_graph_for_method(
        "Inverse-Newton-Coupled-Apply",
        use_cuda_graph=True,
        device=dev_cuda,
        online_stop_tol=None,
        cheb_cuda_graph=True,
    )


def test_naive_newton_preprocess_scales_fro_norm_and_output():
    A = torch.eye(4, dtype=torch.float32) * 2.0
    A_scaled, out_scale = _naive_newton_preprocess(A, p_val=2)
    fro = float(torch.linalg.matrix_norm(A_scaled, ord="fro").item())
    assert abs(fro - 1.0) < 1e-6
    assert abs(out_scale - 0.5) < 1e-6


def test_inverse_newton_coupled_runner_uses_reference_settings():
    captured: dict[str, object] = {}

    def dummy_uncoupled(*args, **kwargs):
        raise AssertionError("uncoupled path should not be used")

    def dummy_coupled(A_ref, B, **kwargs):
        captured["A_ref"] = A_ref.clone()
        captured["kwargs"] = dict(kwargs)
        return B.clone(), object()

    def dummy_cheb(*args, **kwargs):
        raise AssertionError("chebyshev path should not be used")

    runner = _build_solve_runner(
        method="Inverse-Newton-Coupled-Apply",
        pe_step_coeffs=[(1.5, -0.5, 0.0)] * 2,
        cheb_degree=8,
        cheb_coeffs=None,
        p_val=2,
        l_min=0.05,
        symmetrize_every=3,
        online_stop_tol=1e-3,
        online_min_steps=4,
        online_stop_metric="fro",
        online_stop_check_every=5,
        post_correction_steps=2,
        post_correction_order=1,
        uncoupled_fn=dummy_uncoupled,
        coupled_solve_fn=dummy_coupled,
        cheb_apply_fn=dummy_cheb,
    )

    A = torch.eye(4, dtype=torch.float32) * 2.0
    B = torch.randn(4, 3, dtype=torch.float32)
    Z = runner(A, B)

    assert torch.allclose(Z, B * 0.5)
    A_ref = captured["A_ref"]
    kwargs = captured["kwargs"]
    assert isinstance(A_ref, torch.Tensor)
    assert isinstance(kwargs, dict)
    assert float(torch.linalg.matrix_norm(A_ref, ord="fro").item()) <= 1.0 + 1e-6
    assert kwargs["symmetrize_Y"] is False
    assert kwargs["symmetrize_every"] == 1
    assert kwargs["terminal_last_step"] is False
    assert kwargs["online_stop_tol"] is None
    assert kwargs["online_min_steps"] == 1
    assert kwargs["online_stop_metric"] == "diag"
    assert kwargs["online_stop_check_every"] == 1
    assert kwargs["post_correction_steps"] == 0
    assert kwargs["post_correction_order"] == 2


def test_inverse_newton_uncoupled_runner_uses_reference_settings():
    captured: dict[str, object] = {}

    def dummy_uncoupled(A_ref, abc_t, p_val, ws, symmetrize_X):
        captured["A_ref"] = A_ref.clone()
        captured["symmetrize_X"] = bool(symmetrize_X)
        n = int(A_ref.shape[-1])
        X = torch.eye(n, device=A_ref.device, dtype=A_ref.dtype)
        return X, object()

    def dummy_coupled(*args, **kwargs):
        raise AssertionError("coupled path should not be used")

    def dummy_cheb(*args, **kwargs):
        raise AssertionError("chebyshev path should not be used")

    runner = _build_solve_runner(
        method="Inverse-Newton-Inverse-Multiply",
        pe_step_coeffs=[(1.5, -0.5, 0.0)] * 2,
        cheb_degree=8,
        cheb_coeffs=None,
        p_val=2,
        l_min=0.05,
        symmetrize_every=1,
        online_stop_tol=None,
        online_min_steps=1,
        online_stop_metric="diag",
        online_stop_check_every=1,
        post_correction_steps=0,
        post_correction_order=2,
        uncoupled_fn=dummy_uncoupled,
        coupled_solve_fn=dummy_coupled,
        cheb_apply_fn=dummy_cheb,
    )

    A = torch.eye(4, dtype=torch.float32) * 2.0
    B = torch.randn(4, 3, dtype=torch.float32)
    Z = runner(A, B)

    assert torch.allclose(Z, B * 0.5)
    A_ref = captured["A_ref"]
    assert isinstance(A_ref, torch.Tensor)
    assert float(torch.linalg.matrix_norm(A_ref, ord="fro").item()) <= 1.0 + 1e-6
    assert captured["symmetrize_X"] is False
