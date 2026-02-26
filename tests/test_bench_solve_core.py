import torch

from benchmarks.solve.bench_solve_core import (
    _can_use_cuda_graph_for_method,
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
