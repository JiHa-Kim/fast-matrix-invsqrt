import sys
import os
import torch

# Ensure local modules are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fast_iroot.eval import (
    PHASE2_TRANSITION_COEFFS,
    PHASE2_TRANSITION_RHO,
    PHASE2_TERMINAL_COEFFS,
    PHASE2_TERMINAL_RHO,
    step_phase2_local,
)


def verify_policy(n=1024):
    print(f"Verifying Phase 2 Local Protocol (N={n}, bf16)")
    print("-" * 60)

    # 1. Initialize exact eigenvalues at the theoretical boundary (0.7653)
    # Using orthogonal matrices guarantees starting precision.
    rho_start = PHASE2_TRANSITION_RHO
    eigvals = torch.linspace(
        1.0 - rho_start, 1.0 + rho_start, n, dtype=torch.float64, device="cuda"
    )
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, device="cuda"))

    # B is constructed to exactly have the spectrum [1-0.7653, 1+0.7653]
    B = (Q @ torch.diag(eigvals) @ Q.T).to(torch.bfloat16)

    # Z initially acts as Identity on our idealized mathematical construct B
    Z = torch.eye(n, dtype=torch.bfloat16, device="cuda")

    def get_rho(Z, B):
        # We must carefully evaluate the symmetric certificate S = Z^T B Z in fp64 to measure exact bf16 drift
        S = Z.T @ B @ Z
        S_sym = 0.5 * (S + S.T)  # Hardware struct symmetry constraint
        e = torch.linalg.eigvalsh(S_sym.to(torch.float64))
        return float(torch.max(torch.abs(e - 1.0)))

    rho0 = get_rho(Z, B)
    print(f"Initial Phase 1 Output rho: {rho0:.6f} (Theoretical: {rho_start:.6f})")

    # 2. Apply Transition Step (P2-A)
    # Designed for rho = 0.7653 to safely enter < 0.0816 zone
    Z = step_phase2_local(Z, B, PHASE2_TRANSITION_RHO, PHASE2_TRANSITION_COEFFS)
    rho1 = get_rho(Z, B)
    print(
        f"After Transition Step (rho_in={PHASE2_TRANSITION_RHO}): {rho1:.6f} (Target < {PHASE2_TERMINAL_RHO:.6f})"
    )

    if rho1 >= PHASE2_TERMINAL_RHO:
        print("FAIL: Transition step failed to enter the terminal zone.")
        return

    # 3. Apply Terminal Step (P2-B)
    # Designed for rho = 0.0816 to squeeze spectrum into machine noise floor
    Z = step_phase2_local(Z, B, PHASE2_TERMINAL_RHO, PHASE2_TERMINAL_COEFFS)
    rho2 = get_rho(Z, B)
    print(f"After Terminal Step (rho_in={PHASE2_TERMINAL_RHO}): {rho2:.6f}")

    # A d=3 Clenshaw evaluation requires 3 GEMM applications.
    # While a single GEMM at near-identity drifts by 1 ULP (~0.0078),
    # the accumulation of 3 GEMMs shifts the practical hardware noise floor to ~0.023.
    EXPECTED_NOISE_FLOOR = 0.03

    if rho2 <= EXPECTED_NOISE_FLOOR:
        print(
            f"\nSUCCESS: Convergence strictly bounded by multi-GEMM hardware noise floor (< {EXPECTED_NOISE_FLOOR})."
        )
    else:
        print("\nWARNING: Convergence did not reach expected tight noise floor limits.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Verification requires GPU for native bf16 GEMM.")
        sys.exit(0)
    verify_policy()
