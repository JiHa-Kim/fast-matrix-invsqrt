import math
from typing import List, Sequence, Tuple

import torch

try:
    from ..coeff_tuner import make_schedule
except ImportError:
    try:
        from coeff_tuner import make_schedule
    except ImportError:
        make_schedule = None


def _quad_coeffs(
    abc_t: Sequence[Tuple[float, float, float]] | torch.Tensor,
) -> List[Tuple[float, float, float]]:
    if isinstance(abc_t, torch.Tensor):
        abc_cpu = abc_t.detach().to(device="cpu", dtype=torch.float64)
        return [
            (float(abc_cpu[t, 0]), float(abc_cpu[t, 1]), float(abc_cpu[t, 2]))
            for t in range(int(abc_cpu.shape[0]))
        ]
    return [(float(a), float(b), float(c)) for (a, b, c) in abc_t]


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
            and math.isclose(float(l_target), 0.05, rel_tol=0.0, abs_tol=1e-12)
        )
    )
    if use_precomputed:
        pe_quad = pe4_005.clone()
        base_desc = "precomputed(l_target=0.05)"
    else:
        if make_schedule is None:
            raise ImportError(
                "make_schedule is unavailable. Install coeff_tuner or use "
                "coeff_mode='precomputed'/'auto' with l_target=0.05."
            )
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

    s = max(float(coeff_safety), 1.0)
    if s > 1.0:
        pe_quad[:, 1].div_(s)
        pe_quad[:, 2].div_(s * s)
        if coeff_no_final_safety:
            pe_quad[-1, 1].mul_(s)
            pe_quad[-1, 2].mul_(s * s)

    return (
        pe_quad,
        f"{base_desc}, safety={s}, no_final_safety={bool(coeff_no_final_safety)}",
    )
