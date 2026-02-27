from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import torch

from .coupled import (
    IrootWorkspaceCoupled,
    InverseSolveWorkspaceCoupled,
    inverse_proot_pe_quadratic_coupled,
    inverse_solve_pe_quadratic_coupled,
)
from .precond import PrecondStats, precond_gram_spd
from .nsrc import NSRCWorkspace, hybrid_pe_nsrc_solve


@dataclass
class InverseApplyAutoWorkspace:
    solve_ws: Optional[InverseSolveWorkspaceCoupled] = None
    root_ws: Optional[IrootWorkspaceCoupled] = None
    nsrc_ws: Optional[NSRCWorkspace] = None


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
    k_threshold: float = 0.1,  # use hybrid-NSRC if k/n <= threshold
) -> Tuple[torch.Tensor, InverseApplyAutoWorkspace]:
    """Apply inverse p-th root with strategy selection for single-shot vs reuse.

    Strategies:
      - `auto` (default): branching based on k/n and expected_reuse.
      - `direct-solve`: always run coupled solve on RHS block.
      - `hybrid-pe-nsrc`: use PE preconditioner + NSRC refinement (only for p=1).
      - `materialize-root`: compute root operator then multiply (`X @ M_norm`).
    """
    if strategy not in ("auto", "direct-solve", "materialize-root", "hybrid-pe-nsrc"):
        raise ValueError(
            "Unknown strategy: "
            f"'{strategy}'. Supported strategies: 'auto', 'direct-solve', 'materialize-root', 'hybrid-pe-nsrc'."
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
    n, k = M_norm.shape[-2:]
    use_hybrid = strategy == "hybrid-pe-nsrc" or (
        strategy == "auto" and p_val == 1 and (k / n <= k_threshold)
    )

    if use_hybrid:
        # p=1 hybrid solve: faster for small k
        Zn, ws.nsrc_ws = hybrid_pe_nsrc_solve(
            A_norm,
            M_norm,
            abc_t=abc_t,
            pe_steps=2,
            ref_steps=3,
            tol=online_stop_tol,
            ws=ws.nsrc_ws,
        )
        return Zn, ws

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
