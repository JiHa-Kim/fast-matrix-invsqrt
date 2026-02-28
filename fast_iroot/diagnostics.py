from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class SpectralStepStats:
    step: int
    min_eig: float
    max_eig: float
    mean_eig: float
    std_eig: float
    rho_residual: float  # max|1 - lambda|
    clustering_90: float  # fraction of eigenvalues in [0.9, 1.1]
    clustering_99: float  # fraction of eigenvalues in [0.99, 1.01]


@torch.no_grad()
def analyze_spectral_convergence(
    Y: torch.Tensor, step: int
) -> SpectralStepStats:
    """Analyze the eigenvalues of the iteration matrix Y."""
    # Move to CPU and double for accurate eigenvalue decomposition
    Y_f64 = Y.detach().cpu().double()
    
    # We assume Y is diagonalizable and ideally has real eigenvalues (for SPD A)
    # If Y is not symmetric (due to drift), we use eigvals()
    try:
        eigs = torch.linalg.eigvals(Y_f64)
        # Iterative inverse roots of SPD matrices should have real positive eigenvalues
        eigs = eigs.real
    except RuntimeError:
        # Fallback for corner cases
        return SpectralStepStats(
            step=step, min_eig=0, max_eig=0, mean_eig=0, std_eig=0,
            rho_residual=1.0, clustering_90=0, clustering_99=0
        )

    abs_diff = torch.abs(1.0 - eigs)
    rho = float(abs_diff.max().item())
    
    c90 = float((abs_diff <= 0.1).float().mean().item())
    c99 = float((abs_diff <= 0.01).float().mean().item())
    
    return SpectralStepStats(
        step=step,
        min_eig=float(eigs.min().item()),
        max_eig=float(eigs.max().item()),
        mean_eig=float(eigs.mean().item()),
        std_eig=float(eigs.std().item()),
        rho_residual=rho,
        clustering_90=c90,
        clustering_99=c99,
    )


def format_spectral_report(stats_list: List[SpectralStepStats]) -> str:
    """Format spectral stats into a markdown table."""
    lines = [
        "| Step | Min λ | Max λ | Mean λ | ρ(I-Y) | Cluster 90% | Cluster 99% |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stats_list:
        lines.append(
            f"| {s.step} | {s.min_eig:.4f} | {s.max_eig:.4f} | {s.mean_eig:.4f} | "
            f"{s.rho_residual:.2e} | {s.clustering_90:.1%} | {s.clustering_99:.1%} |"
        )
    return "\n".join(lines)
