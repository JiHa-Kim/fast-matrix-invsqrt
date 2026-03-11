import torch

from polar.rational.runner_fast import run_one_case_fast
from polar.rational.runner_tf32 import run_one_case_tf32_rational
from polar.runner import run_one_case
from polar.schedule_spec import StepSpec


def test_fast_runner_matches_baseline_on_dwh_schedule() -> None:
    torch.manual_seed(0)
    m, n = 64, 16
    G = torch.randn(m, n, dtype=torch.float32)
    schedule = [
        StepSpec(kind="DWH_STABLE_SOLVE", ell_in=0.1, ell_out=0.2, pred_kappa_after=5.0, r=1),
        StepSpec(kind="DWH_STABLE_SOLVE", ell_in=0.2, ell_out=0.8, pred_kappa_after=1.25, r=1),
    ]
    kwargs = dict(
        G_storage=G,
        target_kappa_O=10.0,
        schedule=schedule,
        iter_dtype=torch.float32,
        jitter_rel=1e-12,
        tf32=False,
        exact_verify_device="cpu",
    )

    baseline = run_one_case(**kwargs)
    fast = run_one_case_fast(**kwargs)

    assert fast.zolo_steps == 0
    assert fast.dwh_steps == 2
    assert fast.last_step_kind == "DWH_STABLE_SOLVE"
    assert abs(fast.final_kO_exact - baseline.final_kO_exact) < 5e-4


def test_tf32_rational_runner_matches_baseline_on_dwh_schedule() -> None:
    torch.manual_seed(0)
    m, n = 64, 16
    G = torch.randn(m, n, dtype=torch.float32)
    schedule = [
        StepSpec(kind="DWH_STABLE_SOLVE", ell_in=0.1, ell_out=0.2, pred_kappa_after=5.0, r=1),
        StepSpec(kind="DWH_STABLE_SOLVE", ell_in=0.2, ell_out=0.6, pred_kappa_after=1.7, r=1),
        StepSpec(kind="DWH_STABLE_SOLVE", ell_in=0.6, ell_out=0.98, pred_kappa_after=1.05, r=1),
    ]
    kwargs = dict(
        G_storage=G,
        target_kappa_O=10.0,
        schedule=schedule,
        iter_dtype=torch.float32,
        jitter_rel=1e-12,
        tf32=False,
        exact_verify_device="cpu",
    )

    baseline = run_one_case(**kwargs)
    tf32_rational = run_one_case_tf32_rational(**kwargs)

    assert tf32_rational.zolo_steps == 0
    assert tf32_rational.dwh_steps == 3
    assert tf32_rational.fallbacks == 0
    assert tf32_rational.last_step_kind == "DWH_STABLE_SOLVE"
    assert abs(tf32_rational.final_kO_exact - baseline.final_kO_exact) < 2e-3
