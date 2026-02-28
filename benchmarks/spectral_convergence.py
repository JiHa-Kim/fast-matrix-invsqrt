import torch
import math
from typing import List, Tuple
from fast_iroot.coeffs import build_pe_schedules, _quad_coeffs
from fast_iroot.precond import precond_spd
from fast_iroot.diagnostics import analyze_spectral_convergence, format_spectral_report, SpectralStepStats

def run_diagnostic_iteration(
    A_norm: torch.Tensor,
    abc_t: List[Tuple[float, float, float]],
    p_val: int,
    method_name: str
) -> List[SpectralStepStats]:
    Y = A_norm.clone()
    stats = []
    
    # Step 0: Initial spectrum
    stats.append(analyze_spectral_convergence(Y, 0))
    
    for t, (a, b, c) in enumerate(abc_t):
        # B = aI + bY + cY^2
        B = a * torch.eye(Y.shape[-1], device=Y.device, dtype=Y.dtype) + b * Y
        if abs(c) > 1e-9:
            B = B + c * (Y @ Y)
        
        # Update Y based on coupled PE rules
        # For SPD, the commuting model update is Y <- B^p Y
        # We simulate the production update logic
        if p_val == 1:
            Y = B @ Y
        elif p_val == 2:
            Y = B @ Y @ B
        else:
            Bp = torch.matrix_power(B, p_val)
            Y = Bp @ Y
            
        # Symmetrize to maintain real eigenvalues in simulation
        Y = 0.5 * (Y + Y.mT)
        stats.append(analyze_spectral_convergence(Y, t + 1))
        
    return stats

def main():
    n = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 # Use double for diagnostic accuracy
    
    # Create ill-conditioned SPD matrix
    e = torch.logspace(0.0, -4.0, steps=n, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(torch.randn(n, n, device=device, dtype=dtype))
    A = (Q * e.unsqueeze(0)) @ Q.mT
    
    # Precondition
    A_norm, _ = precond_spd(A, mode="jacobi", l_target=0.05)
    
    print(f"# Spectral Convergence Analysis (n={n}, p=2)")
    print(f"Condition Number (preconditioned): {float((1.0 / A_norm.diagonal().min()).item()):.2e}")
    print()

    # 1. PE-Quad (Tuned)
    pe_quad_t, _ = build_pe_schedules(
        l_target=0.05,
        device=device,
        coeff_mode="tuned",
        coeff_seed=0,
        coeff_safety=1.0,
        coeff_no_final_safety=False,
        p_val=2
    )
    abc_pe = _quad_coeffs(pe_quad_t)
    
    # 2. Newton-Schulz
    abc_ns = [(1.5, -0.5, 0.0)] * len(abc_pe)
    
    print("## Method: PE-Quad (Ours)")
    stats_pe = run_diagnostic_iteration(A_norm, abc_pe, 2, "PE-Quad")
    print(format_spectral_report(stats_pe))
    print()
    
    print("## Method: Newton-Schulz (Baseline)")
    stats_ns = run_diagnostic_iteration(A_norm, abc_ns, 2, "Newton-Schulz")
    print(format_spectral_report(stats_ns))

if __name__ == "__main__":
    main()
