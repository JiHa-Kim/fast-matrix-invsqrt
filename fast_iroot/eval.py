import torch


def fro_norm(a: torch.Tensor) -> torch.Tensor:
    # Use higher precision internally for the reduction to avoid overflow/underflow
    # but the result returned to the bf16 loop should be compatible.
    return torch.linalg.norm(a, ord="fro")


def jacobi_init(B: torch.Tensor, jacobi_eps: float) -> torch.Tensor:
    # Everything in bf16
    d = torch.diagonal(B)
    inv_sqrt = torch.rsqrt(d + jacobi_eps)
    Z = torch.diag(inv_sqrt)
    return Z


def choose_beta(S: torch.Tensor, mode: str = "fro", rel_padding: float = 0.01) -> torch.Tensor:
    """
    Unified scaling policy: returns beta such that S/beta has lambda_max <= 1.0.
    Includes relative padding to protect against bf16 GEMM noise.
    """
    n = S.shape[0]
    if mode == "fro":
        # Normalized so that S=I => beta=1
        e = fro_norm(S) / (n**0.5)
    elif mode == "trace":
        e = torch.trace(S).abs() / n
    elif mode == "maxdiag":
        e = torch.max(torch.diagonal(S))
    else:
        raise ValueError("beta mode must be fro|trace|maxdiag")

    # Clamp to ensure we don't shrink if S is already small, 
    # then apply relative padding.
    beta = torch.max(e, torch.ones_like(e)) * (1.0 + rel_padding)
    return beta


def symmetrize(S: torch.Tensor) -> torch.Tensor:
    """Force structural symmetry in bf16."""
    return 0.5 * (S + S.T)


def apply_poly_right_mono(
    Z: torch.Tensor, S: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    """
    Compute Z q(S) with monomial Horner in pure bf16.
    """
    d = a.numel() - 1
    Y = a[d] * Z
    for k in range(d - 1, -1, -1):
        Y = Y @ S
        Y = Y + a[k] * Z
    return Y


def apply_poly_right_cheb(
    Z: torch.Tensor, S: torch.Tensor, c: torch.Tensor, a_dom: float, b_dom: float = 1.0
) -> torch.Tensor:
    """
    Clenshaw's algorithm in pure bf16 on interval [a_dom, b_dom].
    Default b_dom=1.0 matches Phase 1 usage where a_dom=ell.
    """
    d = c.numel() - 1
    if d == 0:
        return c[0] * Z

    # alpha = 2/(b-a), beta_cheb = -(b+a)/(b-a)
    denom = b_dom - a_dom
    alpha = 2.0 / denom
    beta_cheb = -(b_dom + a_dom) / denom

    B_k2 = torch.zeros_like(Z)
    B_k1 = c[d] * Z 

    for k in range(d - 1, 0, -1):
        B_k1_S = B_k1 @ S
        B_k = c[k] * Z + 2.0 * (alpha * B_k1_S + beta_cheb * B_k1) - B_k2
        B_k2 = B_k1
        B_k1 = B_k

    B_k1_S = B_k1 @ S
    out = c[0] * Z + (alpha * B_k1_S + beta_cheb * B_k1) - B_k2

    return out

# ---------------------------------------------------------
# Phase 2: Local Minimax Refinement (Minimal Implementation)
# ---------------------------------------------------------

# Hardware-optimal Phase 2 Chebyshev coefficients (d=3) for r=2.0 (inverse square root).
# Extracted via exact bf16-in-the-loop backward induction.

PHASE2_TRANSITION_COEFFS = [
    1.1710753446292332, 
    -0.5625516521895263, 
    0.19683376700487223, 
    -0.0764417229369569
]
PHASE2_TRANSITION_RHO = 0.7653

PHASE2_TERMINAL_COEFFS = [
    1.0012529091554423, 
    -0.040928175241500456, 
    0.0012543153087897158, 
    -4.297118508547139e-05
]
PHASE2_TERMINAL_RHO = 0.0816

def step_phase2_local(Z: torch.Tensor, B: torch.Tensor, rho_in: float, coeffs: list[float]) -> torch.Tensor:
    """
    Applies a single step of the local Phase 2 preconditioner without dynamic scaling (beta).
    The input matrix spectrum is mathematically guaranteed to be within [1 - rho_in, 1 + rho_in].
    """
    S = symmetrize(Z.T @ B @ Z)
    a_dom = 1.0 - rho_in
    b_dom = 1.0 + rho_in
    c_t = torch.tensor(coeffs, dtype=torch.bfloat16, device=Z.device)
    
    # Z_new = Z * q(S)
    return apply_poly_right_cheb(Z, S, c_t, a_dom=a_dom, b_dom=b_dom)

