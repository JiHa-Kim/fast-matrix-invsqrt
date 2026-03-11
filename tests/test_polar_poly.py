import torch

from polar.polynomial.express import PaperPolarExpressStep, polar_express_paper5_step_matrix_only
from polar.runner import run_one_case
from polar.synthetic import make_matrix_from_singulars


def _device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test suite")
    return "cuda"


def test_polar_express_paper_matrix_only_smoke() -> None:
    S = torch.diag(torch.tensor([0.1, 0.5, 0.9], dtype=torch.bfloat16, device=_device()))
    coeffs = PaperPolarExpressStep(1.875, -1.25, 0.375)
    Q, shift = polar_express_paper5_step_matrix_only(S, coeffs, torch.bfloat16)
    out = torch.diagonal(S.float() @ Q.float()).sqrt()
    assert shift == 0.0
    assert torch.isfinite(Q).all()
    assert out.min().item() > 0.1


def test_polar_express_paper_schedule_smoke() -> None:
    singulars = torch.logspace(0.0, -1.0, 32, base=10.0, dtype=torch.float32)
    G = make_matrix_from_singulars(
        m=128,
        singulars=singulars,
        seed=3,
        device=_device(),
        storage_dtype=torch.bfloat16,
    )
    from polar.schedules import build_schedule

    res = run_one_case(
        G_storage=G,
        target_kappa_O=4.0,
        schedule=build_schedule("pe5paper", 1.0 / 10.0),
        iter_dtype=torch.bfloat16,
        jitter_rel=1e-15,
        tf32=False,
        exact_verify_device="cpu",
    )
    assert torch.isfinite(torch.tensor(res.final_kO_exact))
    assert res.last_step_kind == "PEPAPER5"


def test_polar_express_additive_schedule_smoke() -> None:
    singulars = torch.logspace(0.0, -1.0, 32, base=10.0, dtype=torch.float32)
    G = make_matrix_from_singulars(
        m=128,
        singulars=singulars,
        seed=4,
        device=_device(),
        storage_dtype=torch.bfloat16,
    )
    from polar.schedules import build_schedule

    res = run_one_case(
        G_storage=G,
        target_kappa_O=4.0,
        schedule=build_schedule("pe5add", 1.0 / 10.0),
        iter_dtype=torch.bfloat16,
        jitter_rel=1e-15,
        tf32=False,
        exact_verify_device="cpu",
    )
    assert torch.isfinite(torch.tensor(res.final_kO_exact))
    assert res.last_step_kind == "PEADD5"
