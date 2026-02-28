from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class SpectralStepStats:
    step: int
    min_eig: float
    max_eig: float
    mean_eig: float
    std_eig: float
    rho_residual: float  # max|1 - lambda|
    log_width: float  # log(max_eig / min_eig)
    error_to_identity: float  # max(1 - min_eig, max_eig - 1)
    clustering_90: float  # fraction of eigenvalues in [0.9, 1.1]
    clustering_99: float  # fraction of eigenvalues in [0.99, 1.01]


@torch.no_grad()
def analyze_spectral_convergence(Y: torch.Tensor, step: int) -> SpectralStepStats:
    """Analyze the eigenvalues of the iteration matrix Y.
    
    Uses the symmetric part of Y to ensure real eigenvalues for SPD diagnostics.
    """
    # Move to CPU and double for accurate eigenvalue decomposition
    Y_f64 = Y.detach().cpu().double()
    
    # Use symmetric part for robust SPD diagnostic: Y_s = 0.5 * (Y + Y^T)
    # This captures the 'actual' SPD quality even if there is slight drift.
    Y_s = 0.5 * (Y_f64 + Y_f64.mT)

    try:
        # eigvalsh is faster and safer for symmetric matrices
        eigs = torch.linalg.eigvalsh(Y_s)
    except RuntimeError:
        # Fallback for corner cases
        return SpectralStepStats(
            step=step,
            min_eig=0.0,
            max_eig=0.0,
            mean_eig=0.0,
            std_eig=0.0,
            rho_residual=1.0,
            log_width=float("inf"),
            error_to_identity=1.0,
            clustering_90=0.0,
            clustering_99=0.0,
        )

    abs_diff = torch.abs(1.0 - eigs)
    rho = float(abs_diff.max().item())
    
    m_k = float(eigs.min().item())
    M_k = float(eigs.max().item())
    
    # Avoid log of zero or negative for malformed iterations
    if m_k > 0 and M_k > 0:
        log_width = float(torch.log(torch.tensor(M_k / m_k)).item())
    else:
        log_width = float("inf")
        
    err_id = max(1.0 - m_k, M_k - 1.0)

    c90 = float((abs_diff <= 0.1).float().mean().item())
    c99 = float((abs_diff <= 0.01).float().mean().item())

    return SpectralStepStats(
        step=step,
        min_eig=m_k,
        max_eig=M_k,
        mean_eig=float(eigs.mean().item()),
        std_eig=float(eigs.std().item()),
        rho_residual=rho,
        log_width=log_width,
        error_to_identity=err_id,
        clustering_90=c90,
        clustering_99=c99,
    )


def format_spectral_report(stats_list: List[SpectralStepStats]) -> str:
    """Format spectral stats into a markdown table."""
    lines = [
        "| Step | Min λ | Max λ | ρ(I-Y) | log(M/m) | C90% | C99% |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stats_list:
        lines.append(
            f"| {s.step} | {s.min_eig:.4f} | {s.max_eig:.4f} | "
            f"{s.rho_residual:.2e} | {s.log_width:.3f} | "
            f"{s.clustering_90:.1%} | {s.clustering_99:.1%} |"
        )
    return "\n".join(lines)
