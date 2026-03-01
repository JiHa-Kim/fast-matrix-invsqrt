from __future__ import annotations

import pytest
import torch

from fast_iroot.coeffs import _quad_coeffs, build_pe_schedules
from fast_iroot.coupled import (
    inverse_proot_pe_quadratic_coupled,
    inverse_solve_pe_quadratic_coupled,
)
from fast_iroot.precond import precond_spd


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


def _relative_error(Xhat: torch.Tensor, A: torch.Tensor, p_val: int) -> float:
    """Simple relative error check against eigendecomposition."""
    Af = A.double()
    eigvals, V = torch.linalg.eigh(Af)
    eigvals = eigvals.clamp_min(1e-20)
    D = torch.diag_embed(eigvals ** (-1.0 / p_val))
    Xref = (V @ D @ V.mH).to(A.dtype)

    num = torch.linalg.matrix_norm(Xhat - Xref, ord="fro")
    den = torch.linalg.matrix_norm(Xref, ord="fro").clamp_min(1e-12)
    return float((num / den).max().item())


@pytest.mark.parametrize("p_val", [1, 2, 4])
@pytest.mark.parametrize("case", ["gaussian", "illcond_1e6"])
def test_verify_iroot_coupled_production(p_val: int, case: str):
    """Verify the production coupled iterative root kernels."""
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

    Xn, _ = inverse_proot_pe_quadratic_coupled(
        A_norm,
        abc_t=quad_coeffs,
        p_val=p_val,
        symmetrize_Y=True,
        terminal_last_step=True,
    )

    assert torch.isfinite(Xn).all()
    relerr = _relative_error(Xn, A_norm, p_val)
    assert relerr < 0.15


@pytest.mark.parametrize("case", ["gaussian", "illcond_1e6"])
def test_verify_iroot_solve_production(case: str):
    """Verify the production coupled solve kernels."""
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
    Z_true, _ = torch.linalg.solve_ex(A_norm.float(), B.float())
    Z_true = Z_true.to(dtype)
    relerr = float(torch.linalg.norm(Z_hat - Z_true) / torch.linalg.norm(Z_true))
    assert relerr < 0.15
