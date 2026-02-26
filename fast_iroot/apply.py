from typing import Optional, Sequence, Tuple
import torch

from .coupled import inverse_solve_pe_quadratic_coupled, InverseSolveWorkspaceCoupled


@torch.no_grad()
def apply_inverse(
    A_norm: torch.Tensor,
    M_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[InverseSolveWorkspaceCoupled] = None,
    symmetrize_Y: bool = True,
    symmetrize_every: int = 1,
    terminal_last_step: bool = True,
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """
    Apply an iterative inverse to M_norm using a coupled quadratic PE scheme.
    This effectively computes Z ≈ A_norm^{-1} M_norm by evolving an operator.

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
        online_stop_tol=online_stop_tol,
        online_min_steps=online_min_steps,
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
    online_stop_tol: Optional[float] = None,
    online_min_steps: int = 2,
) -> Tuple[torch.Tensor, InverseSolveWorkspaceCoupled]:
    """
    Apply an iterative inverse p-th root to M_norm using a coupled quadratic PE scheme.
    This effectively computes Z ≈ A_norm^{-1/p} M_norm.

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
        online_stop_tol=online_stop_tol,
        online_min_steps=online_min_steps,
    )
