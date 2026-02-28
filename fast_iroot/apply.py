from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import torch

from .coupled import (
    IrootWorkspaceCoupled,
    InverseSolveWorkspaceCoupled,
    inverse_proot_pe_quadratic_coupled,
    inverse_solve_pe_quadratic_coupled,
)
from .precond import PrecondStats, precond_gram_dual_spd, precond_gram_spd


@dataclass
class InverseApplyAutoWorkspace:
    solve_ws: Optional[InverseSolveWorkspaceCoupled] = None
    root_ws: Optional[IrootWorkspaceCoupled] = None


@dataclass
class GramInverseApplyWorkspace:
    auto_ws: Optional[InverseApplyAutoWorkspace] = None
    A_norm: Optional[torch.Tensor] = None
    stats: Optional[PrecondStats] = None
    cache_g_data_ptr: int = 0
    cache_g_version: int = -1
    cache_g_shape: Tuple[int, ...] = ()
    cache_gram_mode: str = ""
    cache_precond_mode: str = ""
    cache_eps: float = 0.0
    cache_precond_ruiz_iters: int = 0
    cache_ridge_rel: float = 0.0
    cache_l_target: float = 0.0
    cache_lambda_max_est: str = ""
    cache_lambda_max_power_iters: int = 0
    cache_lambda_max_safety: float = 0.0


@dataclass
class DualGramInverseApplyWorkspace:
    auto_ws: Optional[InverseApplyAutoWorkspace] = None
    A_norm: Optional[torch.Tensor] = None
    stats: Optional[PrecondStats] = None
    cache_g_data_ptr: int = 0
    cache_g_version: int = -1
    cache_g_shape: Tuple[int, ...] = ()
    cache_gram_mode: str = ""
    cache_precond_mode: str = ""
    cache_eps: float = 0.0
    cache_precond_ruiz_iters: int = 0
    cache_ridge_rel: float = 0.0
    cache_l_target: float = 0.0
    cache_lambda_max_est: str = ""
    cache_lambda_max_power_iters: int = 0
    cache_lambda_max_safety: float = 0.0


def _gram_cache_ok(
    ws: GramInverseApplyWorkspace,
    G: torch.Tensor,
    *,
    gram_mode: str,
    precond_mode: str,
    eps: float,
    precond_ruiz_iters: int,
    ridge_rel: float,
    l_target: float,
    lambda_max_est: str,
    lambda_max_power_iters: int,
    lambda_max_safety: float,
) -> bool:
    if ws.A_norm is None or ws.stats is None:
        return False
    if ws.A_norm.device != G.device or ws.A_norm.dtype != G.dtype:
        return False
    if ws.cache_g_shape != tuple(G.shape):
        return False
    if ws.cache_g_data_ptr != int(G.data_ptr()):
        return False
    g_version = int(getattr(G, "_version", -1))
    if ws.cache_g_version != g_version:
        return False
    return (
        ws.cache_gram_mode == str(gram_mode)
        and ws.cache_precond_mode == str(precond_mode)
        and ws.cache_eps == float(eps)
        and ws.cache_precond_ruiz_iters == int(precond_ruiz_iters)
        and ws.cache_ridge_rel == float(ridge_rel)
        and ws.cache_l_target == float(l_target)
        and ws.cache_lambda_max_est == str(lambda_max_est)
        and ws.cache_lambda_max_power_iters == int(lambda_max_power_iters)
        and ws.cache_lambda_max_safety == float(lambda_max_safety)
    )


def _dual_gram_cache_ok(
    ws: DualGramInverseApplyWorkspace,
    G: torch.Tensor,
    *,
    gram_mode: str,
    precond_mode: str,
    eps: float,
    precond_ruiz_iters: int,
    ridge_rel: float,
    l_target: float,
    lambda_max_est: str,
    lambda_max_power_iters: int,
    lambda_max_safety: float,
) -> bool:
    if ws.A_norm is None or ws.stats is None:
        return False
    if ws.A_norm.device != G.device or ws.A_norm.dtype != G.dtype:
        return False
    if ws.cache_g_shape != tuple(G.shape):
        return False
    if ws.cache_g_data_ptr != int(G.data_ptr()):
        return False
    g_version = int(getattr(G, "_version", -1))
    if ws.cache_g_version != g_version:
        return False
    return (
        ws.cache_gram_mode == str(gram_mode)
        and ws.cache_precond_mode == str(precond_mode)
        and ws.cache_eps == float(eps)
        and ws.cache_precond_ruiz_iters == int(precond_ruiz_iters)
        and ws.cache_ridge_rel == float(ridge_rel)
        and ws.cache_l_target == float(l_target)
        and ws.cache_lambda_max_est == str(lambda_max_est)
        and ws.cache_lambda_max_power_iters == int(lambda_max_power_iters)
        and ws.cache_lambda_max_safety == float(lambda_max_safety)
    )


@torch.no_grad()
def apply_inverse(
    A_norm: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[InverseSolveWorkspaceCoupled] = None,
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
    assume_spd: bool = True,
    nonspd_adaptive: bool = False,
    nonspd_adaptive_resid_tol: float = 1.0,
    nonspd_adaptive_growth_tol: float = 1.02,
    nonspd_adaptive_check_every: int = 1,
    nonspd_safe_fallback_tol: Optional[float] = None,
    nonspd_safe_early_y_tol: Optional[float] = None,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """
    Apply an iterative inverse to M_norm using a coupled quadratic PE scheme.
    This effectively computes Z ≈ A_norm^{-1} M_norm by evolving an operator.
    Set assume_spd=False for general (non-symmetric) matrices.
    For non-SPD p=1 solves, set nonspd_adaptive=True to enable runtime
    inverse-Newton fallback when Y residual appears unstable.
    Optionally set nonspd_safe_fallback_tol to trigger exact solve fallback
    when final ||A Z - M||/||M|| remains above tolerance.

    Note: When terminal_last_step=True, ws.Y is not advanced on the final step.
    """
    return inverse_solve_pe_quadratic_coupled(
        A_norm=A_norm,
        M_norm=M_norm,
        abc_t=abc_t,
        p_val=1,
        ws=ws,
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
        assume_spd=assume_spd,
        nonspd_adaptive=nonspd_adaptive,
        nonspd_adaptive_resid_tol=nonspd_adaptive_resid_tol,
        nonspd_adaptive_growth_tol=nonspd_adaptive_growth_tol,
        nonspd_adaptive_check_every=nonspd_adaptive_check_every,
        nonspd_safe_fallback_tol=nonspd_safe_fallback_tol,
        nonspd_safe_early_y_tol=nonspd_safe_early_y_tol,
    )


@torch.no_grad()
def apply_inverse_root(
    A_norm: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    p_val: int = 2,
    ws: Optional[InverseSolveWorkspaceCoupled] = None,
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
    assume_spd: bool = True,
    nonspd_adaptive: bool = False,
    nonspd_adaptive_resid_tol: float = 1.0,
    nonspd_adaptive_growth_tol: float = 1.02,
    nonspd_adaptive_check_every: int = 1,
    nonspd_safe_fallback_tol: Optional[float] = None,
    nonspd_safe_early_y_tol: Optional[float] = None,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """
    Apply an iterative inverse p-th root to M_norm using a coupled quadratic PE scheme.
    This effectively computes Z ≈ A_norm^{-1/p} M_norm.
    Set assume_spd=False for general (non-symmetric) matrices.

    Note: When terminal_last_step=True, ws.Y is not advanced on the final step.
    """
    return inverse_solve_pe_quadratic_coupled(
        A_norm=A_norm,
        M_norm=M_norm,
        abc_t=abc_t,
        p_val=p_val,
        ws=ws,
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
        assume_spd=assume_spd,
        nonspd_adaptive=nonspd_adaptive,
        nonspd_adaptive_resid_tol=nonspd_adaptive_resid_tol,
        nonspd_adaptive_growth_tol=nonspd_adaptive_growth_tol,
        nonspd_adaptive_check_every=nonspd_adaptive_check_every,
        nonspd_safe_fallback_tol=nonspd_safe_fallback_tol,
        nonspd_safe_early_y_tol=nonspd_safe_early_y_tol,
    )


@torch.no_grad()
def apply_inverse_root_auto(
    A_norm: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    p_val: int = 2,
    ws: Optional[InverseApplyAutoWorkspace] = None,
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
    assume_spd: bool = True,
    nonspd_adaptive: bool = False,
    nonspd_adaptive_resid_tol: float = 1.0,
    nonspd_adaptive_growth_tol: float = 1.02,
    nonspd_adaptive_check_every: int = 1,
    nonspd_safe_fallback_tol: Optional[float] = None,
    nonspd_safe_early_y_tol: Optional[float] = None,
) -> Tuple[torch.Tensor, InverseApplyAutoWorkspace]:
    """Apply inverse p-th root with strategy selection for single-shot vs reuse.

    Strategies:
      - `auto` (default): branching based on k/n and expected_reuse.
      - `direct-solve`: always run coupled solve on RHS block.
      - `materialize-root`: compute root operator then multiply (`X @ M_norm`).
    """
    if strategy not in ("auto", "direct-solve", "materialize-root"):
        raise ValueError(
            "Unknown strategy: "
            f"'{strategy}'. Supported strategies: 'auto', 'direct-solve', 'materialize-root'."
        )
    reuse = int(expected_reuse)
    if reuse < 1:
        raise ValueError(f"expected_reuse must be >= 1, got {expected_reuse}")

    if ws is None:
        ws = InverseApplyAutoWorkspace()

    # Reuse Case: materialize root once
    if strategy == "materialize-root" or (strategy == "auto" and reuse > 1):
        Xn, ws.root_ws = inverse_proot_pe_quadratic_coupled(
            A_norm,
            abc_t=abc_t,
            p_val=p_val,
            ws=ws.root_ws,
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
            assume_spd=assume_spd,
        )
        return Xn @ M_norm, ws

    # Solve Case (reuse=1 or explicit direct)
    Zn, ws.solve_ws = inverse_solve_pe_quadratic_coupled(
        A_norm=A_norm,
        M_norm=M_norm,
        abc_t=abc_t,
        p_val=p_val,
        ws=ws.solve_ws,
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
        assume_spd=assume_spd,
        nonspd_adaptive=nonspd_adaptive,
        nonspd_adaptive_resid_tol=nonspd_adaptive_resid_tol,
        nonspd_adaptive_growth_tol=nonspd_adaptive_growth_tol,
        nonspd_adaptive_check_every=nonspd_adaptive_check_every,
        nonspd_safe_fallback_tol=nonspd_safe_fallback_tol,
        nonspd_safe_early_y_tol=nonspd_safe_early_y_tol,
    )
    return Zn, ws


@torch.no_grad()
def apply_inverse_root_gram_spd(
    G: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    p_val: int = 2,
    ws: Optional[GramInverseApplyWorkspace] = None,
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
) -> Tuple[torch.Tensor, GramInverseApplyWorkspace, PrecondStats]:
    """Gram-matrix SPD apply path with optional cached preconditioning.

    This path targets ML workloads where `G` is reused across multiple right-hand
    sides. When `reuse_precond=True`, it caches preconditioned `A_norm = f(G^T G)`
    in `ws` and invalidates when `G` storage/version or precondition settings
    change.
    """
    if G.dim() < 2:
        raise ValueError(
            f"G must have dim >= 2 for Gram apply, got shape {tuple(G.shape)}"
        )
    n = int(G.shape[-1])
    if M_norm.shape[-2] != n:
        raise ValueError(
            f"M_norm must have shape[..., {n}, :], got {tuple(M_norm.shape)}"
        )
    if M_norm.device != G.device:
        raise ValueError("G and M_norm must be on the same device")
    if M_norm.dtype != G.dtype:
        raise ValueError("G and M_norm must have the same dtype")

    if ws is None:
        ws = GramInverseApplyWorkspace()

    if reuse_precond and _gram_cache_ok(
        ws,
        G,
        gram_mode=gram_mode,
        precond_mode=precond_mode,
        eps=eps,
        precond_ruiz_iters=precond_ruiz_iters,
        ridge_rel=ridge_rel,
        l_target=l_target,
        lambda_max_est=lambda_max_est,
        lambda_max_power_iters=lambda_max_power_iters,
        lambda_max_safety=lambda_max_safety,
    ):
        assert ws.A_norm is not None and ws.stats is not None
        A_norm = ws.A_norm
        stats = ws.stats
    else:
        A_norm, stats = precond_gram_spd(
            G,
            gram_mode=gram_mode,
            mode=precond_mode,
            eps=eps,
            ruiz_iters=precond_ruiz_iters,
            ridge_rel=ridge_rel,
            l_target=l_target,
            lambda_max_est=lambda_max_est,
            lambda_max_power_iters=lambda_max_power_iters,
            lambda_max_safety=lambda_max_safety,
        )
        if reuse_precond:
            ws.A_norm = A_norm
            ws.stats = stats
            ws.cache_g_data_ptr = int(G.data_ptr())
            ws.cache_g_version = int(getattr(G, "_version", -1))
            ws.cache_g_shape = tuple(G.shape)
            ws.cache_gram_mode = str(gram_mode)
            ws.cache_precond_mode = str(precond_mode)
            ws.cache_eps = float(eps)
            ws.cache_precond_ruiz_iters = int(precond_ruiz_iters)
            ws.cache_ridge_rel = float(ridge_rel)
            ws.cache_l_target = float(l_target)
            ws.cache_lambda_max_est = str(lambda_max_est)
            ws.cache_lambda_max_power_iters = int(lambda_max_power_iters)
            ws.cache_lambda_max_safety = float(lambda_max_safety)

    Z, ws.auto_ws = apply_inverse_root_auto(
        A_norm=A_norm,
        M_norm=M_norm,
        abc_t=abc_t,
        p_val=p_val,
        ws=ws.auto_ws,
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
    )
    return Z, ws, stats


@torch.no_grad()
def apply_inverse_root_gram_rhs_spd(
    G: torch.Tensor,
    B: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    p_val: int = 2,
    ws: Optional[DualGramInverseApplyWorkspace] = None,
    strategy: str = "auto",
    expected_reuse: int = 1,
    reuse_precond: bool = True,
    gram_mode: str = "row-norm",
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
) -> Tuple[torch.Tensor, DualGramInverseApplyWorkspace, PrecondStats]:
    """Dual Gram SPD apply path for RHS in range(G^T): M = G^T B.

    Computes `Z ~= (G^T G)^(-1/p) G^T B` via the exact identity
    `Z = G^T (G G^T)^(-1/p) B`, with optional cached preconditioning.
    """
    if G.dim() < 2:
        raise ValueError(
            f"G must have dim >= 2 for dual Gram apply, got shape {tuple(G.shape)}"
        )
    m = int(G.shape[-2])
    if B.shape[-2] != m:
        raise ValueError(f"B must have shape[..., {m}, :], got {tuple(B.shape)}")
    if B.device != G.device:
        raise ValueError("G and B must be on the same device")
    if B.dtype != G.dtype:
        raise ValueError("G and B must have the same dtype")

    if ws is None:
        ws = DualGramInverseApplyWorkspace()

    if reuse_precond and _dual_gram_cache_ok(
        ws,
        G,
        gram_mode=gram_mode,
        precond_mode=precond_mode,
        eps=eps,
        precond_ruiz_iters=precond_ruiz_iters,
        ridge_rel=ridge_rel,
        l_target=l_target,
        lambda_max_est=lambda_max_est,
        lambda_max_power_iters=lambda_max_power_iters,
        lambda_max_safety=lambda_max_safety,
    ):
        assert ws.A_norm is not None and ws.stats is not None
        A_norm = ws.A_norm
        stats = ws.stats
    else:
        A_norm, stats = precond_gram_dual_spd(
            G,
            gram_mode=gram_mode,
            mode=precond_mode,
            eps=eps,
            ruiz_iters=precond_ruiz_iters,
            ridge_rel=ridge_rel,
            l_target=l_target,
            lambda_max_est=lambda_max_est,
            lambda_max_power_iters=lambda_max_power_iters,
            lambda_max_safety=lambda_max_safety,
        )
        if reuse_precond:
            ws.A_norm = A_norm
            ws.stats = stats
            ws.cache_g_data_ptr = int(G.data_ptr())
            ws.cache_g_version = int(getattr(G, "_version", -1))
            ws.cache_g_shape = tuple(G.shape)
            ws.cache_gram_mode = str(gram_mode)
            ws.cache_precond_mode = str(precond_mode)
            ws.cache_eps = float(eps)
            ws.cache_precond_ruiz_iters = int(precond_ruiz_iters)
            ws.cache_ridge_rel = float(ridge_rel)
            ws.cache_l_target = float(l_target)
            ws.cache_lambda_max_est = str(lambda_max_est)
            ws.cache_lambda_max_power_iters = int(lambda_max_power_iters)
            ws.cache_lambda_max_safety = float(lambda_max_safety)

    U, ws.auto_ws = apply_inverse_root_auto(
        A_norm=A_norm,
        M_norm=B,
        abc_t=abc_t,
        p_val=p_val,
        ws=ws.auto_ws,
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
    )
    Z = G.mT @ U
    return Z, ws, stats


@torch.no_grad()
def apply_inverse_sqrt_spd(
    A_norm: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[InverseSolveWorkspaceCoupled] = None,
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
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """Dedicated SPD p=2 apply path."""
    return apply_inverse_root(
        A_norm,
        M_norm,
        abc_t=abc_t,
        p_val=2,
        ws=ws,
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
    )


@torch.no_grad()
def apply_inverse_sqrt_non_spd(
    A: torch.Tensor,
    M: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[InverseSolveWorkspaceCoupled] = None,
    terminal_last_step: bool = True,
    terminal_tail_steps: int = 1,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
    online_stop_metric: str = "diag",
    online_stop_check_every: int = 1,
    post_correction_steps: int = 0,
    post_correction_order: int = 2,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """Dedicated non-SPD p=2 apply path (no symmetry assumptions)."""
    return apply_inverse_root(
        A,
        M,
        abc_t=abc_t,
        p_val=2,
        ws=ws,
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
    )


@torch.no_grad()
def apply_inverse_sqrt_gram_spd(
    G: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[InverseSolveWorkspaceCoupled] = None,
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
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled, PrecondStats]:
    """Gram-matrix SPD p=2 apply path: precondition A=G^T G, then apply inverse sqrt."""
    A_norm, stats = precond_gram_spd(
        G,
        gram_mode=gram_mode,
        mode=precond_mode,
        eps=eps,
        ruiz_iters=precond_ruiz_iters,
        ridge_rel=ridge_rel,
        l_target=l_target,
        lambda_max_est=lambda_max_est,
        lambda_max_power_iters=lambda_max_power_iters,
        lambda_max_safety=lambda_max_safety,
    )
    Z, ws = apply_inverse_sqrt_spd(
        A_norm,
        M_norm,
        abc_t=abc_t,
        ws=ws,
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
    )
    return Z, ws, stats
