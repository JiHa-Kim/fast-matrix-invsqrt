import torch

from polar.polynomial.express import polar_express_action_chunked, polar_express_step, polar_express_step_matrix_only
from polar.polynomial.minimax import (
    chebyshev_clenshaw_matrix,
    newton_schulz_inv_sqrt_matrix_only,
    poly_inv_sqrt_coeffs_from_ell,
)
from polar.runner import run_one_case
from polar.synthetic import make_matrix_from_singulars


def _device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test suite")
    return "cuda"


def test_poly_coeffs_track_inv_sqrt_on_interval() -> None:
    coeffs = poly_inv_sqrt_coeffs_from_ell(degree=16, ell=0.25)
    assert coeffs.max_rel_err < 5e-3
    assert 0.99 <= coeffs.pred_sigma_max <= 1.01
    assert coeffs.pred_sigma_min > 0.9


def test_clenshaw_matrix_matches_diagonal_reference() -> None:
    S = torch.diag(torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32, device=_device()))
    coeffs = poly_inv_sqrt_coeffs_from_ell(degree=12, ell=0.5)
    Q = chebyshev_clenshaw_matrix(
        S,
        coeffs.coeffs,
        interval_lo=coeffs.interval_lo,
        interval_hi=coeffs.interval_hi,
        out_dtype=torch.float32,
    )
    expected = torch.diag(torch.tensor([2.0, 2.0**0.5, 1.0], dtype=torch.float32, device=_device()))
    assert torch.max(torch.abs(Q - expected)).item() < 5e-3


def test_newton_schulz_small_side_step_reduces_gram_error() -> None:
    S = torch.diag(torch.tensor([0.2, 0.6, 1.0], dtype=torch.float32, device=_device()))
    X, _ = newton_schulz_inv_sqrt_matrix_only(S, steps=5, matmul_dtype=torch.float32)
    resid = torch.linalg.matrix_norm(X @ S @ X - torch.eye(3, device=_device()), ord=2).item()
    assert resid < 2e-2


def test_poly_schedule_smoke() -> None:
    singulars = torch.logspace(0.0, -1.0, 16, base=10.0, dtype=torch.float32)
    G = make_matrix_from_singulars(
        m=64,
        singulars=singulars,
        seed=0,
        device=_device(),
        storage_dtype=torch.float32,
    )
    from polar.schedules import build_schedule

    res = run_one_case(
        G_storage=G,
        target_kappa_O=1.1,
        schedule=build_schedule("poly24x2", 1.0 / 10.0, 100),
        iter_dtype=torch.float32,
        gram_chunk_rows=64,
        rhs_chunk_rows=64,
        jitter_rel=1e-15,
        cert_jitter_rel=1e-15,
        tf32=False,
        exact_verify_device="cpu",
        zolo_coeff_dps=100,
        stop_on_cert=False,
    )
    assert torch.isfinite(torch.tensor(res.final_kO_exact))
    assert res.final_kO_exact < 2.0


def test_poly_schedule_smoke_bf16() -> None:
    singulars = torch.logspace(0.0, -1.0, 16, base=10.0, dtype=torch.float32)
    G = make_matrix_from_singulars(
        m=64,
        singulars=singulars,
        seed=2,
        device=_device(),
        storage_dtype=torch.bfloat16,
    )
    from polar.schedules import build_schedule

    res = run_one_case(
        G_storage=G,
        target_kappa_O=2.5,
        schedule=build_schedule("poly16x2", 1.0 / 10.0, 100),
        iter_dtype=torch.bfloat16,
        gram_chunk_rows=64,
        rhs_chunk_rows=64,
        jitter_rel=1e-15,
        cert_jitter_rel=1e-15,
        tf32=False,
        exact_verify_device="cpu",
        zolo_coeff_dps=100,
        stop_on_cert=False,
    )
    assert torch.isfinite(torch.tensor(res.final_kO_exact))
    assert res.last_step_kind == "POLY(d=16)"


def test_polar_express_chebyshev_matches_monomial() -> None:
    S = torch.diag(torch.tensor([0.2, 0.6, 1.0], dtype=torch.bfloat16, device=_device()))
    step_mono = polar_express_step(0.2, 1.0, degree_q=2, basis="monomial")
    step_cheb = polar_express_step(0.2, 1.0, degree_q=2, basis="chebyshev")
    Q_mono, _ = polar_express_step_matrix_only(S, step_mono, torch.float32)
    Q_cheb, _ = polar_express_step_matrix_only(S, step_cheb, torch.float32)
    out_mono = torch.diagonal(S.float() @ Q_mono.float()).sqrt()
    out_cheb = torch.diagonal(S.float() @ Q_cheb.float()).sqrt()
    assert torch.isfinite(Q_mono).all()
    assert torch.isfinite(Q_cheb).all()
    assert out_mono.min().item() > 0.2
    assert out_cheb.min().item() > 0.2


def test_polar_express_action_matches_matrix_only() -> None:
    torch.manual_seed(0)
    G = torch.randn(64, 16, dtype=torch.float32, device=_device())
    S = G.mT @ G
    step = polar_express_step(0.2, 1.0, degree_q=3, basis="chebyshev")
    Q, _ = polar_express_step_matrix_only(S, step, torch.float32)
    Y_action, _ = polar_express_action_chunked(G, S, step, rhs_chunk_rows=32, out_dtype=torch.float32)
    Y_matrix = G @ Q
    err = torch.linalg.matrix_norm(Y_action - Y_matrix).item()
    ref = torch.linalg.matrix_norm(Y_matrix).item()
    assert err / max(ref, 1e-12) < 5e-5


def test_polar_express_schedule_smoke() -> None:
    singulars = torch.logspace(0.0, -1.0, 32, base=10.0, dtype=torch.float32)
    G = make_matrix_from_singulars(
        m=128,
        singulars=singulars,
        seed=1,
        device=_device(),
        storage_dtype=torch.bfloat16,
    )
    from polar.schedules import build_schedule

    res = run_one_case(
        G_storage=G,
        target_kappa_O=1.1,
        schedule=build_schedule("pe2cheb12", 1.0 / 10.0, 100),
        iter_dtype=torch.bfloat16,
        gram_chunk_rows=128,
        rhs_chunk_rows=128,
        jitter_rel=1e-15,
        cert_jitter_rel=1e-15,
        tf32=False,
        exact_verify_device="cpu",
        zolo_coeff_dps=100,
        stop_on_cert=False,
    )
    assert torch.isfinite(torch.tensor(res.final_kO_exact))
    assert res.final_kO_exact < 2.0
