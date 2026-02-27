from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .apply import (
    GramInverseApplyWorkspace,
    InverseApplyAutoWorkspace,
    apply_inverse_root_auto,
    apply_inverse_root_gram_spd,
)
from .coeffs import build_pe_schedules
from .nonspd import precond_nonspd
from .precond import PrecondStats, precond_spd


@dataclass(frozen=True)
class ScheduleConfig:
    """Configuration for building a quadratic PE schedule."""

    l_target: float = 0.05
    coeff_mode: str = "auto"
    coeff_seed: int = 0
    coeff_safety: float = 1.0
    coeff_no_final_safety: bool = False


@dataclass(frozen=True)
class PrecondConfig:
    """Configuration for SPD preconditioning."""

    mode: str = "none"
    eps: float = 1e-12
    ruiz_iters: int = 2
    ridge_rel: float = 0.0
    l_target: float = 0.05
    lambda_max_est: str = "row_sum"
    lambda_max_power_iters: int = 8
    lambda_max_safety: float = 1.02


def build_schedule(
    device: torch.device,
    *,
    p_val: int = 2,
    config: Optional[ScheduleConfig] = None,
) -> Tuple[torch.Tensor, str]:
    """Build a PE-Quadratic coefficient schedule for inverse p-th-root kernels."""
    cfg = config if config is not None else ScheduleConfig()
    return build_pe_schedules(
        l_target=cfg.l_target,
        device=device,
        coeff_mode=cfg.coeff_mode,
        coeff_seed=cfg.coeff_seed,
        coeff_safety=cfg.coeff_safety,
        coeff_no_final_safety=cfg.coeff_no_final_safety,
        p_val=p_val,
    )


@torch.no_grad()
def solve_spd(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    p_val: int = 2,
    abc_t: Optional[Sequence[Tuple[float, float, float]] | torch.Tensor] = None,
    schedule_config: Optional[ScheduleConfig] = None,
    precond_config: Optional[PrecondConfig] = None,
    workspace: Optional[InverseApplyAutoWorkspace] = None,
    strategy: str = "auto",
    expected_reuse: int = 1,
    symmetrize_Y: bool = True,
    symmetrize_every: int = 1,
    terminal_last_step: bool = True,
    terminal_tail_steps: int = 1,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
    online_stop_metric: str = "diag",
    online_stop_check_every: int = 1,
    post_correction_steps: int = 0,
    post_correction_order: int = 2,
    k_threshold: float = 0.1,
) -> Tuple[torch.Tensor, InverseApplyAutoWorkspace, PrecondStats, str]:
    """Solve/apply `Z ~= A^(-1/p) B` for SPD `A` using preconditioning + PE kernels."""
    pcfg = precond_config if precond_config is not None else PrecondConfig()

    A_norm, stats = precond_spd(
        A,
        mode=pcfg.mode,
        eps=pcfg.eps,
        ruiz_iters=pcfg.ruiz_iters,
        ridge_rel=pcfg.ridge_rel,
        l_target=pcfg.l_target,
        lambda_max_est=pcfg.lambda_max_est,
        lambda_max_power_iters=pcfg.lambda_max_power_iters,
        lambda_max_safety=pcfg.lambda_max_safety,
    )

    if abc_t is None:
        abc_t, schedule_desc = build_schedule(
            A.device,
            p_val=p_val,
            config=schedule_config,
        )
    else:
        schedule_desc = "user-provided"

    Z, ws = apply_inverse_root_auto(
        A_norm=A_norm,
        M_norm=B,
        abc_t=abc_t,
        p_val=p_val,
        ws=workspace,
        strategy=strategy,
        expected_reuse=expected_reuse,
        symmetrize_Y=symmetrize_Y,
        symmetrize_every=symmetrize_every,
        terminal_last_step=terminal_last_step,
        terminal_tail_steps=terminal_tail_steps,
        online_stop_tol=online_stop_tol,
        online_min_steps=online_min_steps,
        online_stop_metric=online_stop_metric,
        online_stop_check_every=online_stop_check_every,
        post_correction_steps=post_correction_steps,
        post_correction_order=post_correction_order,
        assume_spd=True,
        k_threshold=k_threshold,
    )
    return Z, ws, stats, schedule_desc


@torch.no_grad()
def solve_nonspd(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    p_val: int = 1,
    abc_t: Optional[Sequence[Tuple[float, float, float]] | torch.Tensor] = None,
    schedule_config: Optional[ScheduleConfig] = None,
    nonspd_precond_mode: str = "row-norm",
    nonspd_precond_ruiz_iters: int = 2,
    nonspd_precond_eps: float = 1e-12,
    workspace: Optional[InverseApplyAutoWorkspace] = None,
    strategy: str = "auto",
    expected_reuse: int = 1,
    terminal_last_step: bool = True,
    terminal_tail_steps: int = 1,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
    online_stop_metric: str = "diag",
    online_stop_check_every: int = 1,
    post_correction_steps: int = 0,
    post_correction_order: int = 2,
    nonspd_adaptive: bool = False,
    nonspd_adaptive_resid_tol: float = 1.0,
    nonspd_adaptive_growth_tol: float = 1.02,
    nonspd_adaptive_check_every: int = 1,
    nonspd_safe_fallback_tol: Optional[float] = None,
    nonspd_safe_early_y_tol: Optional[float] = None,
    k_threshold: float = 0.1,
) -> Tuple[torch.Tensor, InverseApplyAutoWorkspace, str]:
    """Solve/apply `Z ~= A^(-1) B` for non-SPD `A` with generic scaling.

    Non-SPD support is intentionally restricted to p=1 in the high-level API.
    """
    if int(p_val) != 1:
        raise ValueError(
            f"solve_nonspd currently supports p_val=1 only, got {p_val}"
        )
    A_norm = precond_nonspd(
        A,
        mode=nonspd_precond_mode,
        ruiz_iters=nonspd_precond_ruiz_iters,
        eps=nonspd_precond_eps,
    )

    if abc_t is None:
        abc_t, schedule_desc = build_schedule(
            A.device,
            p_val=p_val,
            config=schedule_config,
        )
    else:
        schedule_desc = "user-provided"

    Z, ws = apply_inverse_root_auto(
        A_norm=A_norm,
        M_norm=B,
        abc_t=abc_t,
        p_val=p_val,
        ws=workspace,
        strategy=strategy,
        expected_reuse=expected_reuse,
        symmetrize_Y=False,
        symmetrize_every=1,
        terminal_last_step=terminal_last_step,
        terminal_tail_steps=terminal_tail_steps,
        online_stop_tol=online_stop_tol,
        online_min_steps=online_min_steps,
        online_stop_metric=online_stop_metric,
        online_stop_check_every=online_stop_check_every,
        post_correction_steps=post_correction_steps,
        post_correction_order=post_correction_order,
        assume_spd=False,
        nonspd_adaptive=nonspd_adaptive,
        nonspd_adaptive_resid_tol=nonspd_adaptive_resid_tol,
        nonspd_adaptive_growth_tol=nonspd_adaptive_growth_tol,
        nonspd_adaptive_check_every=nonspd_adaptive_check_every,
        nonspd_safe_fallback_tol=nonspd_safe_fallback_tol,
        nonspd_safe_early_y_tol=nonspd_safe_early_y_tol,
        k_threshold=k_threshold,
    )
    return Z, ws, schedule_desc


@torch.no_grad()
def solve_gram_spd(
    G: torch.Tensor,
    B: torch.Tensor,
    *,
    p_val: int = 2,
    abc_t: Optional[Sequence[Tuple[float, float, float]] | torch.Tensor] = None,
    schedule_config: Optional[ScheduleConfig] = None,
    workspace: Optional[GramInverseApplyWorkspace] = None,
    strategy: str = "auto",
    expected_reuse: int = 1,
    reuse_precond: bool = True,
    gram_mode: str = "col-norm",
    precond_mode: str = "none",
    eps: float = 1e-12,
    precond_ruiz_iters: int = 2,
    ridge_rel: float = 0.0,
    l_target: float = 0.05,
    lambda_max_est: str = "row_sum",
    lambda_max_power_iters: int = 8,
    lambda_max_safety: float = 1.02,
    symmetrize_Y: bool = True,
    symmetrize_every: int = 1,
    terminal_last_step: bool = True,
    terminal_tail_steps: int = 1,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
    online_stop_metric: str = "diag",
    online_stop_check_every: int = 1,
    post_correction_steps: int = 0,
    post_correction_order: int = 2,
    k_threshold: float = 0.1,
) -> Tuple[torch.Tensor, GramInverseApplyWorkspace, PrecondStats, str]:
    """Solve/apply `Z ~= (G^T G)^(-1/p) B` with cached Gram preconditioning."""
    if abc_t is None:
        abc_t, schedule_desc = build_schedule(
            G.device,
            p_val=p_val,
            config=schedule_config,
        )
    else:
        schedule_desc = "user-provided"

    Z, ws, stats = apply_inverse_root_gram_spd(
        G=G,
        M_norm=B,
        abc_t=abc_t,
        p_val=p_val,
        ws=workspace,
        strategy=strategy,
        expected_reuse=expected_reuse,
        reuse_precond=reuse_precond,
        gram_mode=gram_mode,
        precond_mode=precond_mode,
        eps=eps,
        precond_ruiz_iters=precond_ruiz_iters,
        ridge_rel=ridge_rel,
        l_target=l_target,
        lambda_max_est=lambda_max_est,
        lambda_max_power_iters=lambda_max_power_iters,
        lambda_max_safety=lambda_max_safety,
        symmetrize_Y=symmetrize_Y,
        symmetrize_every=symmetrize_every,
        terminal_last_step=terminal_last_step,
        terminal_tail_steps=terminal_tail_steps,
        online_stop_tol=online_stop_tol,
        online_min_steps=online_min_steps,
        online_stop_metric=online_stop_metric,
        online_stop_check_every=online_stop_check_every,
        post_correction_steps=post_correction_steps,
        post_correction_order=post_correction_order,
        k_threshold=k_threshold,
    )
    return Z, ws, stats, schedule_desc
