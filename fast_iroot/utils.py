from typing import Optional
import torch


@torch.no_grad()
def _symmetrize_inplace(M: torch.Tensor, tmp: Optional[torch.Tensor] = None) -> None:
    if tmp is None:
        M.copy_(0.5 * (M + M.mT))
        return
    tmp.copy_(M.mT)
    M.add_(tmp).mul_(0.5)


@torch.no_grad()
def _matmul_into(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    torch.matmul(A, B, out=out)
    return out


@torch.no_grad()
def _addmm_into(
    bias: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    *,
    beta: float,
    alpha: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """out = beta * bias + alpha * (mat1 @ mat2).  Fused BLAS call."""
    if mat1.dim() == 2:
        torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha, out=out)
    else:
        torch.baddbmm(bias, mat1, mat2, beta=beta, alpha=alpha, out=out)
    return out


@torch.no_grad()
def _bpow_times_y(
    B: torch.Tensor,
    Y: torch.Tensor,
    p: int,
    out: torch.Tensor,
    tmp1: torch.Tensor,
    tmp2: torch.Tensor,
) -> None:
    """Compute B^p * Y into `out` using O(log p) matmuls (binary exponentiation).

    B, Y are inputs. tmp1, tmp2 are scratch buffers (same shape).
    `out`, `tmp1`, and `tmp2` must not alias `B` or `Y`.
    """
    if p <= 0:
        out.copy_(Y)
        return
    if p == 1:
        torch.matmul(B, Y, out=out)
        return
    if p == 2:
        torch.matmul(B, Y, out=tmp1)
        torch.matmul(B, tmp1, out=out)
        return
    if p == 4:
        # B^2 -> tmp1, B^2*Y -> tmp2, B^2*(B^2*Y) -> out  (3 matmuls)
        torch.matmul(B, B, out=tmp1)
        torch.matmul(tmp1, Y, out=tmp2)
        torch.matmul(tmp1, tmp2, out=out)
        return

    # General binary exponentiation for p >= 3
    # Collect binary digits (LSB first)
    bits = []
    pp = p
    while pp > 0:
        bits.append(pp & 1)
        pp >>= 1

    # cur_base cycles through B <-> tmp1
    # cur_result cycles through Y <-> out <-> tmp2
    cur_base = B
    next_base = tmp1
    cur_result = Y
    next_result = out

    for i, bit in enumerate(bits):
        if bit:
            torch.matmul(cur_base, cur_result, out=next_result)
            cur_result, next_result = next_result, cur_result
            # Ensure next_result is a *different* buffer from cur_result
            if next_result is Y:
                next_result = tmp2
            elif next_result is out:
                next_result = tmp2
            elif next_result is tmp2:
                next_result = out if cur_result is not out else tmp2

        if i < len(bits) - 1:
            torch.matmul(cur_base, cur_base, out=next_base)
            cur_base, next_base = next_base, cur_base

    # Copy to out if result is not already there
    if cur_result is not out:
        out.copy_(cur_result)
