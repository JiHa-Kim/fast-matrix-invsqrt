from __future__ import annotations

import pytest
import torch

from fast_iroot.coeffs import _quad_coeffs, build_pe_schedules
from fast_iroot.coupled import (
    inverse_proot_pe_quadratic_coupled,
    inverse_solve_pe_quadratic_coupled,
)
from fast_iroot.metrics import compute_quality_stats, iroot_relative_error
from fast_iroot.precond import precond_spd
from fast_iroot.uncoupled import inverse_proot_pe_quadratic_uncoupled


def _make_spd(
    n: int,
    case: str,
    device: torch.device,
    dtype: torch.dtype,
    g: torch.Generator,
) -> torch.Tensor:
    if case == "gaussian":
        X = torch.randn(n, n, device=device, dtype=dtype, generator=g)
        A = (X @ X.mT) / n
        A.diagonal().add_(1e-3)
        return A
    if case == "illcond_1e6":
        e = torch.logspace(0.0, -6.0, steps=n, device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(
            torch.randn(n, n, device=device, dtype=torch.float32, generator=g)
        )
        return ((Q * e.unsqueeze(0)) @ Q.mT).to(dtype=dtype)
    raise ValueError(case)


@pytest.mark.parametrize("p_val", [1, 2, 4])
@pytest.mark.parametrize("case", ["gaussian", "illcond_1e6"])
def test_verify_iroot_core_methods(p_val: int, case: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    g = torch.Generator(device=device)
    g.manual_seed(42 + 31 * p_val)

    pe_quad, _ = build_pe_schedules(
        l_target=0.05,
        device=device,
        coeff_mode="tuned",
        coeff_seed=0,
        coeff_safety=1.0,
        coeff_no_final_safety=False,
        p_val=p_val,
    )
    quad_coeffs = _quad_coeffs(pe_quad)

    A = _make_spd(64, case, device, dtype, g)
    A_norm, _ = precond_spd(A, mode="aol", ridge_rel=1e-4, l_target=0.05)

    methods = (
        inverse_proot_pe_quadratic_uncoupled(
            A_norm, abc_t=quad_coeffs, p_val=p_val, symmetrize_X=True
        )[0],
        inverse_proot_pe_quadratic_coupled(
            A_norm,
            abc_t=quad_coeffs,
            p_val=p_val,
            symmetrize_Y=True,
            terminal_last_step=True,
        )[0],
    )
    for Xn in methods:
        assert torch.isfinite(Xn).all()
        q = compute_quality_stats(
            Xn,
            A_norm,
            power_iters=0,
            mv_samples=0,
            p_val=p_val,
        )
        relerr = float(iroot_relative_error(Xn.float(), A_norm.float(), p_val=p_val).mean())
        assert q.residual_fro < 0.12
        assert relerr < 0.12


@pytest.mark.parametrize("case", ["gaussian", "illcond_1e6"])
def test_verify_iroot_p1_solve(case: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    g = torch.Generator(device=device)
    g.manual_seed(777)

    pe_quad, _ = build_pe_schedules(
        l_target=0.05,
        device=device,
        coeff_mode="tuned",
        coeff_seed=0,
        coeff_safety=1.0,
        coeff_no_final_safety=False,
        p_val=1,
    )
    quad_coeffs = _quad_coeffs(pe_quad)

    A = _make_spd(64, case, device, dtype, g)
    A_norm, _ = precond_spd(A, mode="aol", ridge_rel=1e-4, l_target=0.05)
    B = torch.randn(64, 16, device=device, dtype=dtype, generator=g)

    Z_hat, _ = inverse_solve_pe_quadratic_coupled(
        A_norm,
        B,
        abc_t=quad_coeffs,
        p_val=1,
        symmetrize_Y=True,
        terminal_last_step=True,
    )
    Z_true = torch.linalg.solve(A_norm.float(), B.float()).to(dtype)
    relerr = float(torch.linalg.norm(Z_hat - Z_true) / torch.linalg.norm(Z_true))
    assert relerr < 0.12
