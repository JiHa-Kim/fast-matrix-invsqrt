import math
from typing import List, Sequence, Tuple

import torch

from .coeff_tuner import make_schedule


def _quad_coeffs(
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
) -> List[Tuple[float, float, float]]:
    if isinstance(abc_t, torch.Tensor):
        if abc_t.ndim != 2 or abc_t.shape[1] != 3:
            raise ValueError(f"abc_t tensor must have shape (T, 3), got {abc_t.shape}")
        if abc_t.shape[0] == 0:
            raise ValueError(
                "abc_t must contain at least one (a,b,c) coefficient triple"
            )
        abc_cpu = abc_t.detach().to(device="cpu", dtype=torch.float64)
        return [
            (float(abc_cpu[t, 0]), float(abc_cpu[t, 1]), float(abc_cpu[t, 2]))
            for t in range(int(abc_cpu.shape[0]))
        ]

    if len(abc_t) == 0:
        raise ValueError("abc_t must contain at least one (a,b,c) coefficient triple")
    # O(1) fast path for the hottest call pattern:
    # callers pass the already-normalized list[tuple[float,float,float]] from
    # a prior _quad_coeffs(...) conversion.
    if isinstance(abc_t, list):
        first = abc_t[0]
        if len(first) == 3 and all(isinstance(x, float) for x in first):
            return abc_t
        for item in abc_t:
            if len(item) != 3:
                raise ValueError(f"Each item in abc_t must be a triple, got {item}")
            if not all(isinstance(x, float) for x in item):
                return [(float(a), float(b), float(c)) for (a, b, c) in abc_t]
        return abc_t
    for item in abc_t:
        if len(item) != 3:
            raise ValueError(f"Each item in abc_t must be a triple, got {item}")
    return [(float(a), float(b), float(c)) for (a, b, c) in abc_t]


def _quad_coeffs_hot(
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
) -> List[Tuple[float, float, float]]:
    """Cheap per-call coefficient accessor for kernel hot loops.

    Accepts pre-normalized Python lists with minimal validation, and falls back
    to full normalization for tensor/tuple inputs.
    """
    if isinstance(abc_t, list):
        if len(abc_t) == 0:
            raise ValueError(
                "abc_t must contain at least one (a,b,c) coefficient triple"
            )
        first = abc_t[0]
        if len(first) != 3:
            raise ValueError(f"Each item in abc_t must be a triple, got {first}")
        return abc_t
    return _quad_coeffs(abc_t)


def _abc_from_alpha(alpha: float, p_val: int) -> Tuple[float, float, float]:
    inv_p = 1.0 / float(p_val)
    a = 1.0 + inv_p + float(alpha)
    b = -inv_p - 2.0 * float(alpha)
    c = float(alpha)
    return float(a), float(b), float(c)


def _project_to_local_family(
    a: float, b: float, c: float, p_val: int
) -> Tuple[float, float, float]:
    """Project an arbitrary quadratic q(y)=a+by+cy^2 onto the local family.

    We match curvature at 1 (alpha=c) and enforce q(1)=1, q'(1)=-1/p exactly.
    """
    alpha = float(c)
    return _abc_from_alpha(alpha, p_val)


def _apply_quadratic_safety_local(
    pe_quad: torch.Tensor, p_val: int, s: float, no_final: bool, project: bool
) -> None:
    """Apply safety by damping alpha in the local family.

    Operates in-place on pe_quad[:, 0:3].
    """
    T = pe_quad.shape[0]
    for t in range(T):
        if no_final and (t == T - 1):
            s_t = 1.0
        else:
            s_t = float(s)
        if s_t == 1.0 and not project:
            continue

        a = float(pe_quad[t, 0].item())
        b = float(pe_quad[t, 1].item())
        c = float(pe_quad[t, 2].item())

        if project:
            a, b, c = _project_to_local_family(a, b, c, p_val)

        alpha = float(c)
        alpha = alpha / s_t

        a2, b2, c2 = _abc_from_alpha(alpha, p_val)
        pe_quad[t, 0] = float(a2)
        pe_quad[t, 1] = float(b2)
        pe_quad[t, 2] = float(c2)


def build_pe_schedules(
    l_target: float,
    device: torch.device,
    coeff_mode: str,
    coeff_seed: int,
    coeff_safety: float,
    coeff_no_final_safety: bool,
    p_val: int = 2,
) -> Tuple[torch.Tensor, str]:
    """Build quadratic PE coefficient schedule.

    Returns (pe_quad_tensor, description_string).
    """
    if coeff_mode not in ("precomputed", "auto", "tuned"):
        raise ValueError(
            f"Unknown coeff_mode: '{coeff_mode}'. Supported modes are 'precomputed', 'auto', 'tuned'."
        )

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

    use_precomputed = (p_val == 2) and (
        coeff_mode == "precomputed"
        or (
            coeff_mode == "auto"
            and math.isclose(float(l_target), 0.05, rel_tol=0.0, abs_tol=1e-6)
        )
    )
    if use_precomputed:
        pe_quad = pe4_005.clone()
        base_desc = "precomputed(l_target=0.05)"
    else:
        l0 = max(float(l_target), 1e-6)
        quad = make_schedule(
            "quad", T=4, l0=l0, l_cushion=l0, seed=int(coeff_seed), p_val=p_val
        )
        pe_quad = torch.tensor(
            [[float(row[0]), float(row[1]), float(row[2])] for row in quad],
            device=device,
            dtype=torch.float32,
        )
        base_desc = f"tuned(l_target={l_target}, seed={coeff_seed})"

    s = float(coeff_safety)
    if s < 1.0:
        raise ValueError(f"coeff_safety must be >= 1.0, got {s}")

    if s > 1.0:
        # NEW: damp alpha in local family, preserving q(1)=1 and q'(1)=-1/p
        # We project precomputed steps; tuned steps from make_schedule are already in the family.
        project = bool(use_precomputed)
        _apply_quadratic_safety_local(
            pe_quad,
            p_val=p_val,
            s=s,
            no_final=bool(coeff_no_final_safety),
            project=project,
        )

    return (
        pe_quad,
        f"{base_desc}, safety={s}, no_final_safety={bool(coeff_no_final_safety)}",
    )
