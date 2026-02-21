from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class QualityStats:
    residual_fro: float
    residual_spec: float
    sym_x: float
    sym_w: float
    mv_err: float


@torch.no_grad()
def exact_inverse_sqrt(A: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    eigvals, V = torch.linalg.eigh(A.double())
    eigvals = eigvals.clamp_min(eps)
    D = torch.diag_embed(torch.rsqrt(eigvals))
    X = V @ D @ V.mT
    return X.to(dtype=A.dtype)


@torch.no_grad()
def isqrt_relative_error(Xhat: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    Xref = exact_inverse_sqrt(A)
    denom = torch.linalg.matrix_norm(Xref, ord="fro").clamp_min(1e-12)
    num = torch.linalg.matrix_norm(Xhat - Xref, ord="fro")
    return num / denom


@torch.no_grad()
def compute_quality_stats(
    X: torch.Tensor,
    A: torch.Tensor,
    power_iters: int,
    mv_samples: int,
    eye_mat: Optional[torch.Tensor] = None,
) -> QualityStats:
    n = A.shape[-1]
    Xf = X.float()
    Af = A.float()
    eye = eye_mat
    if (
        eye is None
        or eye.shape != Af.shape
        or eye.device != Af.device
        or eye.dtype != Af.dtype
    ):
        eye = torch.eye(n, device=Af.device, dtype=Af.dtype)
    W = Xf @ Af @ Xf
    R = eye - W

    residual_fro = torch.linalg.matrix_norm(R, ord="fro").item() / math.sqrt(float(n))

    x_denom = torch.linalg.matrix_norm(Xf, ord="fro").clamp_min(1e-12)
    w_denom = torch.linalg.matrix_norm(W, ord="fro").clamp_min(1e-12)
    sym_x = torch.linalg.matrix_norm(Xf - Xf.mT, ord="fro") / x_denom
    sym_w = torch.linalg.matrix_norm(W - W.mT, ord="fro") / w_denom

    if mv_samples > 0:
        V = torch.randn(n, int(mv_samples), device=Af.device, dtype=Af.dtype)
        RV = R @ V
        mv_num = torch.linalg.vector_norm(RV, dim=0)
        mv_den = torch.linalg.vector_norm(V, dim=0).clamp_min(1e-12)
        mv_err = float((mv_num / mv_den).median().item())
    else:
        mv_err = float("nan")

    if power_iters > 0:
        v = torch.randn(n, 1, device=Af.device, dtype=Af.dtype)
        v = v / torch.linalg.vector_norm(v).clamp_min(1e-12)
        for _ in range(int(power_iters)):
            v = R @ v
            v = v / torch.linalg.vector_norm(v).clamp_min(1e-12)
        residual_spec = float(torch.linalg.vector_norm(R @ v).item())
    else:
        residual_spec = float("nan")

    return QualityStats(
        residual_fro=float(residual_fro),
        residual_spec=residual_spec,
        sym_x=float(sym_x.item()),
        sym_w=float(sym_w.item()),
        mv_err=mv_err,
    )
