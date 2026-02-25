import math
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class PrecondStats:
    # Conservative batch aggregates:
    # - rho_proxy: max over batch (bigger = worse)
    # - gersh_lo: min over batch (smaller = worse)
    # - kappa_proxy: derived from gersh_lo
    rho_proxy: float
    gersh_lo: float
    kappa_proxy: float


@torch.no_grad()
def precond_spd(
    A: torch.Tensor,
    mode: str,
    eps: float = 1e-12,
    ridge_rel: float = 0.0,
    l_target: float = 0.05,
    lambda_max_est: str = "row_sum",
    lambda_max_power_iters: int = 8,
    lambda_max_safety: float = 1.02,
) -> Tuple[torch.Tensor, PrecondStats]:
    # -------- precondition (scale) --------
    if mode == "none":
        A_pre = A
    elif mode == "frob":
        n = A.shape[-1]
        s = torch.linalg.matrix_norm(A, ord="fro")
        s = torch.clamp(s / math.sqrt(n), min=eps)
        A_pre = A / s.unsqueeze(-1).unsqueeze(-1)
    elif mode == "aol":
        d = torch.rsqrt(A.abs().sum(dim=-1).clamp_min(eps))
        A_pre = (d.unsqueeze(-1) * A) * d.unsqueeze(-2)
    else:
        raise ValueError(f"unknown preconditioner: {mode}")

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
    # Note: a final row-sum normalization may scale the lower bound back down, so
    # this is an improvement heuristic, not a strict post-normalization guarantee.
    if l_target > 0.0:
        abs_row_sum2 = A_norm.abs().sum(dim=-1)
        diag = A_norm.diagonal(dim1=-2, dim2=-1)
        off = abs_row_sum2 - diag.abs()
        g_lo = (diag - off).min(dim=-1)[0]  # per batch element
        shift = (float(l_target) - g_lo).clamp_min(0.0)

        if torch.any(shift > 0):
            A_norm = A_norm.clone()
            A_norm.diagonal(dim1=-2, dim2=-1).add_(shift.unsqueeze(-1))
            abs_row_sum3 = A_norm.abs().sum(dim=-1)
            u2 = abs_row_sum3.max(dim=-1)[0].clamp_min(eps)
            A_norm = A_norm / u2.unsqueeze(-1).unsqueeze(-1)

    # -------- final Gershgorin lower bound (per batch) --------
    abs_row_sum4 = A_norm.abs().sum(dim=-1)
    diag4 = A_norm.diagonal(dim1=-2, dim2=-1)
    off4 = abs_row_sum4 - diag4.abs()
    g_lo_final = (diag4 - off4).min(dim=-1)[0]  # shape: batch

    # -------- rho proxy (per batch) --------
    diag_mean = A_norm.float().diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
    max_row = A_norm.float().abs().sum(dim=-1).max(dim=-1)[0].clamp_min(1e-12)
    rho = max_row / diag_mean  # shape: batch

    # -------- FIX: conservative batch aggregation for auto-policy safety --------
    # If you pick ONE method for the whole batch, use worst-case stats.
    g_lo_scalar = float(g_lo_final.float().min().item())
    rho_proxy = float(rho.float().max().item())
    # Note: kappa_proxy is a heuristic for method selection, not a true \kappa(A).
    kappa_proxy = 1.0 / max(g_lo_scalar, 1e-6)

    return A_norm, PrecondStats(
        rho_proxy=rho_proxy,
        gersh_lo=g_lo_scalar,
        kappa_proxy=float(kappa_proxy),
    )
