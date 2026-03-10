import pytest
import torch

mpmath = pytest.importorskip("mpmath")

from polar.rational.runner_fast import run_one_case_fast
from polar.runner import run_one_case
from polar.schedule_spec import StepSpec


def test_fast_runner_matches_baseline_on_mixed_dwh_zolo_schedule() -> None:
    torch.manual_seed(0)
    m, n = 64, 16
    G = torch.randn(m, n, dtype=torch.float32)
    schedule = [
        StepSpec(kind="DWH", ell_in=0.1, ell_out=0.2, pred_kappa_after=5.0, r=1),
        StepSpec(kind="ZOLO", ell_in=0.2, ell_out=0.8, pred_kappa_after=1.25, r=2),
    ]
    kwargs = dict(
        G_storage=G,
        target_kappa_O=10.0,
        schedule=schedule,
        iter_dtype=torch.float32,
        gram_chunk_rows=64,
        rhs_chunk_rows=64,
        jitter_rel=1e-12,
        tf32=False,
        exact_verify_device="cpu",
        zolo_coeff_dps=50,
    )

    baseline = run_one_case(**kwargs)
    fast = run_one_case_fast(**kwargs)

    assert fast.zolo_steps == 1
    assert fast.dwh_steps == 1
    assert fast.last_step_kind == "ZOLO_MIXED(r=2)"
    assert abs(fast.final_kO_exact - baseline.final_kO_exact) < 5e-4
