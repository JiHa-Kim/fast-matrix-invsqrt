"""
Reference-only Paterson-Stockmeyer variant for PE2 non-terminal Y update.

This file is intentionally not wired into the benchmark harness. It exists as
an archived implementation reference for future experimentation.
"""

from __future__ import annotations

from typing import Tuple

import torch


@torch.no_grad()
def pe2_ps_y_update_reference(
    Y: torch.Tensor, a: float, b: float, c: float
) -> torch.Tensor:
    """
    Compute Y_next = B Y B with:
        B = a I + b Y + c Y^2

    Uses a Paterson-Stockmeyer-style degree-5 split:
        Y_next = U(Y) + Y^3 V(Y)
    where
        U(Y) = a^2 Y + 2ab Y^2
        V(Y) = (b^2 + 2ac) I + 2bc Y + c^2 Y^2
    """
    Y2 = Y @ Y
    Y3 = Y2 @ Y

    a2 = a * a
    b2 = b * b
    c2 = c * c
    ab2 = 2.0 * a * b
    bc2 = 2.0 * b * c
    b2_2ac = b2 + 2.0 * a * c

    U = Y.mul(a2) + Y2.mul(ab2)
    V = Y2.mul(c2) + Y.mul(bc2)
    V = V + torch.eye(Y.shape[-1], device=Y.device, dtype=Y.dtype).mul(b2_2ac)
    return U + (Y3 @ V)


def _coeff_terms(a: float, b: float, c: float) -> Tuple[float, ...]:
    """Small helper retained with the reference for symbolic sanity checks."""
    return (
        a * a,               # Y^1
        2.0 * a * b,         # Y^2
        b * b + 2.0 * a * c, # Y^3
        2.0 * b * c,         # Y^4
        c * c,               # Y^5
    )
