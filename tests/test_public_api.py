import torch

from fast_iroot import (
    PrecondConfig,
    ScheduleConfig,
    build_schedule,
    solve_gram_spd,
    solve_nonspd,
    solve_spd,
)


def _make_spd(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, n, generator=g)
    return x.mT @ x + 1e-2 * torch.eye(n)


def test_build_schedule_shape_and_desc():
    coeffs, desc = build_schedule(torch.device("cpu"), p_val=2)
    assert coeffs.ndim == 2
    assert coeffs.shape[1] == 3
    assert isinstance(desc, str) and len(desc) > 0


def test_solve_spd_smoke():
    n, k = 16, 4
    A = _make_spd(n, seed=1)
    B = torch.randn(n, k)

    Z, ws, stats, schedule_desc = solve_spd(
        A,
        B,
        p_val=2,
        precond_config=PrecondConfig(mode="jacobi", l_target=0.05),
        schedule_config=ScheduleConfig(coeff_mode="auto"),
    )
    assert Z.shape == (n, k)
    assert torch.isfinite(Z).all()
    assert ws is not None
    assert stats.kappa_proxy > 0.0
    assert isinstance(schedule_desc, str)


def test_solve_nonspd_smoke():
    n, k = 16, 3
    g = torch.Generator().manual_seed(2)
    A = torch.randn(n, n, generator=g)
    B = torch.randn(n, k, generator=g)

    Z, ws, schedule_desc = solve_nonspd(
        A,
        B,
        p_val=1,
        schedule_config=ScheduleConfig(coeff_mode="tuned", coeff_seed=1),
    )
    assert Z.shape == (n, k)
    assert torch.isfinite(Z).all()
    assert ws is not None
    assert isinstance(schedule_desc, str)


def test_solve_gram_spd_smoke():
    m, n, k = 24, 12, 5
    g = torch.Generator().manual_seed(3)
    G = torch.randn(m, n, generator=g)
    B = torch.randn(n, k, generator=g)

    Z, ws, stats, schedule_desc = solve_gram_spd(
        G,
        B,
        p_val=2,
        precond_mode="none",
    )
    assert Z.shape == (n, k)
    assert torch.isfinite(Z).all()
    assert ws is not None
    assert stats.kappa_proxy > 0.0
    assert isinstance(schedule_desc, str)
