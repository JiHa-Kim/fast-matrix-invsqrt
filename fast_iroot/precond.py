import math
from dataclasses import dataclass
from typing import Tuple

import torch

from .utils import _check_square

SPD_PRECOND_MODES: tuple[str, ...] = ("none", "frob", "aol", "jacobi", "ruiz")
GRAM_PRECOND_MODES: tuple[str, ...] = ("none", "col-norm")


@dataclass
class PrecondStats:
    # Conservative batch aggregates:
    # - rho_proxy: max over batch (bigger = worse)
    # - gersh_lo: min over batch (smaller = worse)
    # - kappa_proxy: derived from gersh_lo
    rho_proxy: float
    gersh_lo: float
    kappa_proxy: float


def _scale_spd(
    A: torch.Tensor,
    mode: str,
    eps: float,
    ruiz_iters: int,
) -> torch.Tensor:
    if mode == "none":
        return A
    if mode == "frob":
        n = A.shape[-1]
        s = torch.linalg.matrix_norm(A, ord="fro")
        s = torch.clamp(s / math.sqrt(n), min=eps)
        return A / s.unsqueeze(-1).unsqueeze(-1)
    if mode == "aol":
        d = torch.rsqrt(A.abs().sum(dim=-1).clamp_min(eps))
        return (d.unsqueeze(-1) * A) * d.unsqueeze(-2)
    if mode == "jacobi":
        d = torch.rsqrt(A.diagonal(dim1=-2, dim2=-1).clamp_min(eps))
        return (d.unsqueeze(-1) * A) * d.unsqueeze(-2)
    if mode == "ruiz":
        if int(ruiz_iters) < 1:
            raise ValueError(f"ruiz_iters must be >= 1, got {ruiz_iters}")
        A_scaled = A
        for _ in range(int(ruiz_iters)):
            # Symmetric equilibration rounds keep SPD structure while shrinking
            # anisotropy with only reductions + elementwise scaling.
            row_l2 = torch.linalg.vector_norm(A_scaled.float(), dim=-1).clamp_min(
                float(eps)
            )
            d = torch.rsqrt(row_l2).to(dtype=A_scaled.dtype)
            A_scaled = (d.unsqueeze(-1) * A_scaled) * d.unsqueeze(-2)
        return A_scaled
    raise ValueError(
        f"unknown preconditioner: {mode}. Supported modes are {SPD_PRECOND_MODES}."
    )


@torch.no_grad()
def precond_spd(
    A: torch.Tensor,
    mode: str,
    eps: float = 1e-12,
    ruiz_iters: int = 2,
    ridge_rel: float = 0.0,
    l_target: float = 0.05,
    lambda_max_est: str = "row_sum",
    lambda_max_power_iters: int = 8,
    lambda_max_safety: float = 1.02,
    compute_rho_proxy: bool = True,
) -> Tuple[torch.Tensor, PrecondStats]:
    """
    Preconditions a symmetric positive definite (SPD) matrix.

    Note: The diagonal shift logic assumes that the absolute value of the diagonal
    increases exactly by the shift, i.e., `|a_ii + s| = |a_ii| + s`. This holds
    true for matrices with non-negative diagonals, such as SPD matrices.

    For `mode="aol"`, note that it uses symmetric scaling `D A D`. While this
    preserves definiteness, if `A` has unusual sign patterns, the strict Gershgorin
    diagonal dominance bounds may be less tight.
    """
    if A.is_complex():
        raise ValueError("precond_spd does not support complex tensors")
    _check_square(A)
    if (A.diagonal(dim1=-2, dim2=-1) <= 0).any():
        raise ValueError(
            "precond_spd requires SPD matrices with strictly positive diagonals"
        )

    # -------- precondition (scale) --------
    A_pre = _scale_spd(A=A, mode=mode, eps=float(eps), ruiz_iters=int(ruiz_iters))

    # -------- ridge if requested --------
    if ridge_rel > 0.0:
        diag_mean = A_pre.diagonal(dim1=-2, dim2=-1).mean(dim=-1).abs()
        lam = torch.clamp(ridge_rel * diag_mean, min=eps)
        A_pre = A_pre.clone()
        A_pre.diagonal(dim1=-2, dim2=-1).add_(lam.unsqueeze(-1))

    # -------- estimate lambda_max for normalization --------
    if lambda_max_est == "power" and int(lambda_max_power_iters) > 0:
        # FIX: correct per-batch normalization and use a per-batch random vector.
        batch = A_pre.shape[:-2]
        n = A_pre.shape[-1]
        A32 = A_pre.float()

        v = torch.randn(*batch, n, 1, device=A_pre.device, dtype=A32.dtype)

        def _bnorm(x: torch.Tensor) -> torch.Tensor:
            # norm over the last two dims, keep dims for broadcasting
            return torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp_min(
                1e-12
            )

        v = v / _bnorm(v)
        for _ in range(int(lambda_max_power_iters)):
            v = A32 @ v
            v = v / _bnorm(v)

        Av = A32 @ v
        # Rayleigh quotient per batch: (v^T Av) / (v^T v) but v is normalized.
        u = (v.mT @ Av).abs().squeeze(-1).squeeze(-1).clamp_min(eps)
        u = (u * float(lambda_max_safety)).to(dtype=A_pre.dtype)
    else:
        abs_row_sum = A_pre.abs().sum(dim=-1)
        u = abs_row_sum.max(dim=-1)[0].clamp_min(eps)

    A_norm = A_pre / u.unsqueeze(-1).unsqueeze(-1)

    # -------- optional diagonal shift to improve Gershgorin lower bound --------
    if l_target > 0.0:
        if l_target >= 1.0:
            raise ValueError(f"l_target must be < 1.0, got {l_target}")
        abs_row_sum2 = A_norm.abs().sum(dim=-1)
        diag = A_norm.diagonal(dim1=-2, dim2=-1)
        off = abs_row_sum2 - diag.abs()
        g_lo = (diag - off).min(dim=-1)[0]  # per batch element

        r = abs_row_sum2.max(dim=-1)[0].clamp_min(eps)
        den = max(1.0 - float(l_target), 1e-6)
        shift = (float(l_target) * r - g_lo) / den
        shift = shift.clamp_min(0.0)

        if torch.any(shift > 0):
            A_norm = A_norm.clone()
            A_norm.diagonal(dim1=-2, dim2=-1).add_(shift.unsqueeze(-1))
            A_norm = A_norm / (r + shift).unsqueeze(-1).unsqueeze(-1)

    # -------- final Gershgorin lower bound (per batch) --------
    abs_row_sum4 = A_norm.abs().sum(dim=-1)
    diag4 = A_norm.diagonal(dim1=-2, dim2=-1)
    off4 = abs_row_sum4 - diag4.abs()
    g_lo_final = (diag4 - off4).min(dim=-1)[0]  # shape: batch

    # -------- conservative batch aggregation for auto-policy safety --------
    # If you pick ONE method for the whole batch, use worst-case stats.
    g_lo_scalar = float(g_lo_final.float().min().item())
    if bool(compute_rho_proxy):
        diag_mean = (
            A_norm.float().diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
        )
        max_row = A_norm.float().abs().sum(dim=-1).max(dim=-1)[0].clamp_min(1e-12)
        rho = max_row / diag_mean  # shape: batch
        rho_proxy = float(rho.float().max().item())
    else:
        rho_proxy = float("nan")
    # Note: kappa_proxy is a heuristic for method selection, not a true \kappa(A).
    kappa_proxy = 1.0 / max(g_lo_scalar, 1e-6)

    return A_norm, PrecondStats(
        rho_proxy=rho_proxy,
        gersh_lo=g_lo_scalar,
        kappa_proxy=float(kappa_proxy),
    )


@torch.no_grad()
def precond_gram_spd(
    G: torch.Tensor,
    gram_mode: str = "col-norm",
    mode: str = "none",
    eps: float = 1e-12,
    ruiz_iters: int = 2,
    ridge_rel: float = 0.0,
    l_target: float = 0.05,
    lambda_max_est: str = "row_sum",
    lambda_max_power_iters: int = 8,
    lambda_max_safety: float = 1.02,
    compute_rho_proxy: bool = True,
) -> Tuple[torch.Tensor, PrecondStats]:
    """
    Preconditions an SPD Gram matrix formed from feature matrix `G`, where
    `A = G^T G`.

    `gram_mode="col-norm"` applies the same transform as scaling columns of `G`
    by inverse column-norms, but computes it through `A = G^T G` followed by
    Jacobi scaling on `A`. This avoids expensive explicit scaling of large `G`
    tensors on GPU while preserving identical algebra.
    """
    if G.is_complex():
        raise ValueError("precond_gram_spd does not support complex tensors")
    if G.dim() < 2:
        raise ValueError(
            f"precond_gram_spd expects tensor with dim >= 2, got shape {tuple(G.shape)}"
        )

    A = G.mT @ G

    if gram_mode == "none":
        A_gram = A
    elif gram_mode == "col-norm":
        diag = A.diagonal(dim1=-2, dim2=-1)
        if (diag <= float(eps)).any():
            raise ValueError(
                "precond_gram_spd requires non-zero column norms for gram_mode='col-norm'"
            )
        d = torch.rsqrt(diag.clamp_min(float(eps)))
        A_gram = (d.unsqueeze(-1) * A) * d.unsqueeze(-2)
    else:
        raise ValueError(
            f"unknown gram preconditioner: {gram_mode}. Supported modes are {GRAM_PRECOND_MODES}."
        )

    return precond_spd(
        A_gram,
        mode=mode,
        eps=eps,
        ruiz_iters=ruiz_iters,
        ridge_rel=ridge_rel,
        l_target=l_target,
        lambda_max_est=lambda_max_est,
        lambda_max_power_iters=lambda_max_power_iters,
        lambda_max_safety=lambda_max_safety,
        compute_rho_proxy=compute_rho_proxy,
    )
