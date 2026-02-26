from __future__ import annotations

import math

import torch

NONSPD_PRECOND_MODES: tuple[str, ...] = ("row-norm", "frob", "ruiz")


@torch.no_grad()
def precond_nonspd(
    A: torch.Tensor,
    mode: str = "row-norm",
    ruiz_iters: int = 2,
    eps: float = 1e-12,
) -> torch.Tensor:
    mode_s = str(mode)
    eps_f = float(eps)
    if eps_f <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")
    if mode_s == "row-norm":
        # Generic non-SPD scaling: normalize by max absolute row-sum.
        u = A.abs().sum(dim=-1).max(dim=-1).values.clamp_min(eps_f)
        return A / u.unsqueeze(-1).unsqueeze(-1)
    if mode_s == "frob":
        # Frobenius scaling to keep typical singular values around O(1).
        n = float(A.shape[-1])
        scale = (torch.linalg.matrix_norm(A, ord="fro") / math.sqrt(n)).clamp_min(eps_f)
        return A / scale.unsqueeze(-1).unsqueeze(-1)
    if mode_s == "ruiz":
        iters = int(ruiz_iters)
        if iters < 1:
            raise ValueError(f"ruiz_iters must be >= 1, got {ruiz_iters}")
        X = A.clone()
        for _ in range(iters):
            row = X.abs().sum(dim=-1).clamp_min(eps_f)
            X = X / row.unsqueeze(-1)
            col = X.abs().sum(dim=-2).clamp_min(eps_f)
            X = X / col.unsqueeze(-2)
        return X
    raise ValueError(
        "Unknown non-SPD preconditioner mode: "
        f"'{mode}'. Supported modes are {list(NONSPD_PRECOND_MODES)}."
    )
