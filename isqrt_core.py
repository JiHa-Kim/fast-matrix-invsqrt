from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

try:
    from .coeff_tuner import make_schedule
except ImportError:
    try:
        from coeff_tuner import make_schedule
    except ImportError:  # Optional: only needed for tuned PE schedule generation.
        make_schedule = None


@dataclass
class IsqrtWorkspace:
    X: torch.Tensor
    Xbuf: torch.Tensor
    Y: torch.Tensor
    Ybuf: torch.Tensor
    Y2: torch.Tensor
    B: torch.Tensor
    B2: torch.Tensor
    eye_mat: torch.Tensor


@dataclass
class PrecondStats:
    # Conservative batch aggregates:
    # - rho_proxy: max over batch (bigger = worse)
    # - gersh_lo: min over batch (smaller = worse)
    # - kappa_proxy: derived from gersh_lo
    rho_proxy: float
    gersh_lo: float
    kappa_proxy: float


@dataclass
class AutoPolicyConfig:
    policy: str
    n_switch: int
    rho_switch: float
    kappa_ns3_max: float
    kappa_pe2_min: float


def _alloc_ws(A: torch.Tensor) -> IsqrtWorkspace:
    shape = A.shape
    n = shape[-1]
    # IMPORTANT: do NOT .contiguous() an expanded identity; that materializes a full batch of I.
    eye = torch.eye(n, device=A.device, dtype=A.dtype).expand_as(A)
    return IsqrtWorkspace(
        X=A.new_empty(shape),
        Xbuf=A.new_empty(shape),
        Y=A.new_empty(shape),
        Ybuf=A.new_empty(shape),
        Y2=A.new_empty(shape),
        B=A.new_empty(shape),
        B2=A.new_empty(shape),
        eye_mat=eye,
    )


def _ws_ok(ws: Optional[IsqrtWorkspace], A: torch.Tensor) -> bool:
    if ws is None:
        return False
    return (
        ws.X.device == A.device
        and ws.X.dtype == A.dtype
        and ws.X.shape == A.shape
        and ws.eye_mat.device == A.device
        and ws.eye_mat.dtype == A.dtype
        and ws.eye_mat.shape == A.shape
    )


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


def _affine_coeffs(
    ab_t: Sequence[Tuple[float, float]] | torch.Tensor,
) -> List[Tuple[float, float]]:
    if isinstance(ab_t, torch.Tensor):
        ab_cpu = ab_t.detach().to(device="cpu", dtype=torch.float32)
        return [
            (float(ab_cpu[t, 0]), float(ab_cpu[t, 1]))
            for t in range(int(ab_cpu.shape[0]))
        ]
    return [(float(a), float(b)) for (a, b) in ab_t]


def _quad_coeffs(
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
) -> List[Tuple[float, float, float]]:
    if isinstance(abc_t, torch.Tensor):
        abc_cpu = abc_t.detach().to(device="cpu", dtype=torch.float32)
        return [
            (float(abc_cpu[t, 0]), float(abc_cpu[t, 1]), float(abc_cpu[t, 2]))
            for t in range(int(abc_cpu.shape[0]))
        ]
    return [(float(a), float(b), float(c)) for (a, b, c) in abc_t]


@torch.no_grad()
def precond_spd(
    A: torch.Tensor,
    mode: str,
    eps: float = 1e-12,
    ridge_rel: float = 0.0,
    l_target: float = 0.05,
    lambda_max_est: str = "row_sum",
    lambda_max_power_iters: int = 8,
    lambda_max_safety: float = 1.02,
) -> Tuple[torch.Tensor, PrecondStats]:
    # -------- precondition (scale) --------
    if mode == "none":
        A_pre = A
    elif mode == "frob":
        n = A.shape[-1]
        s = torch.linalg.matrix_norm(A, ord="fro")
        s = torch.clamp(s / math.sqrt(n), min=eps)
        A_pre = A / s.unsqueeze(-1).unsqueeze(-1)
    elif mode == "aol":
        d = torch.rsqrt(A.abs().sum(dim=-1).clamp_min(eps))
        A_pre = (d.unsqueeze(-1) * A) * d.unsqueeze(-2)
    else:
        raise ValueError(f"unknown preconditioner: {mode}")

    # -------- ridge if requested --------
    if ridge_rel > 0.0:
        diag_mean = A_pre.diagonal(dim1=-2, dim2=-1).mean(dim=-1).abs()
        lam = torch.clamp(ridge_rel * diag_mean, min=eps)
        A_pre = A_pre.clone()
        A_pre.diagonal(dim1=-2, dim2=-1).add_(lam.unsqueeze(-1))

    # -------- estimate lambda_max for normalization --------
    if lambda_max_est == "power" and int(lambda_max_power_iters) > 0:
        # FIX: correct per-batch normalization and use a per-batch random vector.
        batch = A_pre.shape[:-2]
        n = A_pre.shape[-1]
        A32 = A_pre.float()

        v = torch.randn(*batch, n, 1, device=A_pre.device, dtype=A32.dtype)

        def _bnorm(x: torch.Tensor) -> torch.Tensor:
            # norm over the last two dims, keep dims for broadcasting
            return torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True).clamp_min(
                1e-12
            )

        v = v / _bnorm(v)
        for _ in range(int(lambda_max_power_iters)):
            v = A32 @ v
            v = v / _bnorm(v)

        Av = A32 @ v
        # Rayleigh quotient per batch: (v^T Av) / (v^T v) but v is normalized.
        u = (v.mT @ Av).abs().squeeze(-1).squeeze(-1).clamp_min(eps)
        u = (u * float(lambda_max_safety)).to(dtype=A_pre.dtype)
    else:
        abs_row_sum = A_pre.abs().sum(dim=-1)
        u = abs_row_sum.max(dim=-1)[0].clamp_min(eps)

    A_norm = A_pre / u.unsqueeze(-1).unsqueeze(-1)

    # -------- optional diagonal shift to enforce Gershgorin lower bound >= l_target --------
    if l_target > 0.0:
        abs_row_sum2 = A_norm.abs().sum(dim=-1)
        diag = A_norm.diagonal(dim1=-2, dim2=-1)
        off = abs_row_sum2 - diag.abs()
        g_lo = (diag - off).min(dim=-1)[0]  # per batch element
        shift = (float(l_target) - g_lo).clamp_min(0.0)

        if torch.any(shift > 0):
            A_norm = A_norm.clone()
            A_norm.diagonal(dim1=-2, dim2=-1).add_(shift.unsqueeze(-1))
            abs_row_sum3 = A_norm.abs().sum(dim=-1)
            u2 = abs_row_sum3.max(dim=-1)[0].clamp_min(eps)
            A_norm = A_norm / u2.unsqueeze(-1).unsqueeze(-1)

    # -------- final Gershgorin lower bound (per batch) --------
    abs_row_sum4 = A_norm.abs().sum(dim=-1)
    diag4 = A_norm.diagonal(dim1=-2, dim2=-1)
    off4 = abs_row_sum4 - diag4.abs()
    g_lo_final = (diag4 - off4).min(dim=-1)[0]  # shape: batch

    # -------- rho proxy (per batch) --------
    diag_mean = A_norm.float().diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
    max_row = A_norm.float().abs().sum(dim=-1).max(dim=-1)[0].clamp_min(1e-12)
    rho = max_row / diag_mean  # shape: batch

    # -------- FIX: conservative batch aggregation for auto-policy safety --------
    # If you pick ONE method for the whole batch, use worst-case stats.
    g_lo_scalar = float(g_lo_final.float().min().item())
    rho_proxy = float(rho.float().max().item())
    kappa_proxy = 1.0 / max(g_lo_scalar, 1e-6)

    return A_norm, PrecondStats(
        rho_proxy=rho_proxy,
        gersh_lo=g_lo_scalar,
        kappa_proxy=float(kappa_proxy),
    )


def choose_auto_method(n: int, stats: PrecondStats, cfg: AutoPolicyConfig) -> str:
    # Return one of: "NS3", "PE-NS3", "PE2"
    if cfg.policy == "size_rho":
        if (n >= int(cfg.n_switch)) or (stats.rho_proxy >= float(cfg.rho_switch)):
            return "PE2"
        return "PE-NS3"

    if cfg.policy == "interval":
        if stats.kappa_proxy >= float(cfg.kappa_pe2_min):
            return "PE2"
        if stats.kappa_proxy <= float(cfg.kappa_ns3_max):
            return "NS3"
        return "PE-NS3"

    # default: combined
    if (n >= int(cfg.n_switch)) or (stats.rho_proxy >= float(cfg.rho_switch)):
        return "PE2"
    if stats.kappa_proxy <= float(cfg.kappa_ns3_max):
        return "NS3"
    if stats.kappa_proxy >= float(cfg.kappa_pe2_min):
        return "PE2"
    return "PE-NS3"


@torch.no_grad()
def inverse_sqrt_ns(
    A_norm: torch.Tensor,
    iters: int,
    ws: Optional[IsqrtWorkspace] = None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspace]:
    if not _ws_ok(ws, A_norm):
        ws = _alloc_ws(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)

    T = int(iters)
    for t in range(T):
        ws.B.copy_(ws.Y).mul_(-0.5)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(1.5)

        _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        # NOTE: if terminal_last_step is True, ws.Y is intentionally not updated on the last step.
        if terminal_last_step and (t == T - 1):
            break

        _matmul_into(ws.Y, ws.B, ws.B2)
        _matmul_into(ws.B, ws.B2, ws.Ybuf)
        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf, ws.B)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws


@torch.no_grad()
def inverse_sqrt_pe_affine(
    A_norm: torch.Tensor,
    ab_t: Sequence[Tuple[float, float]] | torch.Tensor,
    ws: Optional[IsqrtWorkspace] = None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspace]:
    if not _ws_ok(ws, A_norm):
        ws = _alloc_ws(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)
    coeffs = _affine_coeffs(ab_t)

    T = len(coeffs)
    for t, (a, b) in enumerate(coeffs):
        ws.B.copy_(ws.Y).mul_(b)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(a)

        _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        # NOTE: if terminal_last_step is True, ws.Y is intentionally not updated on the last step.
        if terminal_last_step and (t == T - 1):
            break

        _matmul_into(ws.Y, ws.B, ws.B2)
        _matmul_into(ws.B, ws.B2, ws.Ybuf)
        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf, ws.B)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws


@torch.no_grad()
def inverse_sqrt_pe_quadratic(
    A_norm: torch.Tensor,
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
    ws: Optional[IsqrtWorkspace] = None,
    symmetrize_Y: bool = True,
    terminal_last_step: bool = True,
) -> Tuple[torch.Tensor, IsqrtWorkspace]:
    if not _ws_ok(ws, A_norm):
        ws = _alloc_ws(A_norm)
    assert ws is not None

    ws.X.copy_(ws.eye_mat)
    ws.Y.copy_(A_norm)
    coeffs = _quad_coeffs(abc_t)

    T = len(coeffs)
    for t, (a, b, c) in enumerate(coeffs):
        _matmul_into(ws.Y, ws.Y, ws.Y2)
        ws.B.copy_(ws.Y2).mul_(c)
        ws.B.add_(ws.Y, alpha=b)
        ws.B.diagonal(dim1=-2, dim2=-1).add_(a)

        _matmul_into(ws.X, ws.B, ws.Xbuf)
        ws.X, ws.Xbuf = ws.Xbuf, ws.X

        # NOTE: if terminal_last_step is True, ws.Y is intentionally not updated on the last step.
        if terminal_last_step and (t == T - 1):
            break

        _matmul_into(ws.Y, ws.B, ws.B2)
        _matmul_into(ws.B, ws.B2, ws.Ybuf)
        if symmetrize_Y:
            _symmetrize_inplace(ws.Ybuf, ws.B)
        ws.Y, ws.Ybuf = ws.Ybuf, ws.Y

    return ws.X, ws


def build_pe_schedules(
    l_target: float,
    device: torch.device,
    coeff_mode: str,
    coeff_seed: int,
    coeff_safety: float,
    coeff_no_final_safety: bool,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    pe4_005 = torch.tensor(
        [
            [3.9021484662, -7.5907070592, 4.8608311100],
            [1.9377808302, -1.3492930916, 0.4109873756],
            [1.8751235181, -1.2502010546, 0.3750775341],
            [1.8749540081, -1.2499080179, 0.3749540099],
        ],
        device=device,
        dtype=torch.float32,
    )
    pe_ns3_005 = torch.tensor(
        [
            [2.8915871412, -2.2700922296],
            [1.6887519393, -0.6154098405],
            [1.5200745126, -0.5165294726],
        ],
        device=device,
        dtype=torch.float32,
    )

    use_precomputed = coeff_mode == "precomputed" or (
        coeff_mode == "auto"
        and math.isclose(float(l_target), 0.05, rel_tol=0.0, abs_tol=1e-12)
    )
    if use_precomputed:
        pe_affine = pe_ns3_005.clone()
        pe2 = pe4_005[:2].contiguous().clone()
        base_desc = "precomputed(l_target=0.05)"
    else:
        if make_schedule is None:
            raise ImportError(
                "make_schedule is unavailable. Install coeff_tuner or use "
                "coeff_mode='precomputed'/'auto' with l_target=0.05."
            )
        l0 = max(float(l_target), 1e-6)
        aff = make_schedule("affine", T=3, l0=l0, l_cushion=l0, seed=int(coeff_seed))
        quad = make_schedule("quad", T=4, l0=l0, l_cushion=l0, seed=int(coeff_seed))
        pe_affine = torch.tensor(
            [[float(row[0]), float(row[1])] for row in aff],
            device=device,
            dtype=torch.float32,
        )
        pe2 = torch.tensor(
            [[float(row[0]), float(row[1]), float(row[2])] for row in quad[:2]],
            device=device,
            dtype=torch.float32,
        )
        base_desc = f"tuned(l_target={l_target}, seed={coeff_seed})"

    s = max(float(coeff_safety), 1.0)
    if s > 1.0:
        pe_affine[:, 1].div_(s)
        pe2[:, 1].div_(s)
        pe2[:, 2].div_(s * s)
        if coeff_no_final_safety:
            pe_affine[-1, 1].mul_(s)
            pe2[-1, 1].mul_(s)
            pe2[-1, 2].mul_(s * s)

    return (
        pe_affine,
        pe2,
        f"{base_desc}, safety={s}, no_final_safety={bool(coeff_no_final_safety)}",
    )
