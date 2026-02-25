"""
verify_iroot.py — Correctness & stability verification for inverse p-th root.

Tests quadratic methods (uncoupled and coupled) across
p in {1, 2, 3, 4, 8} and n in {64, 256}.
"""

from __future__ import annotations

import sys
import torch

from fast_iroot import (
    _quad_coeffs,
    build_pe_schedules,
    inverse_proot_pe_quadratic_uncoupled,
    inverse_proot_pe_quadratic_coupled,
    precond_spd,
)
from fast_iroot.metrics import compute_quality_stats, iroot_relative_error


def make_spd(
    n: int, case: str, device: torch.device, dtype: torch.dtype, g: torch.Generator
):
    if case == "gaussian":
        X = torch.randn(n, n, device=device, dtype=dtype, generator=g)
        A = (X @ X.mT) / n
        A.diagonal().add_(1e-3)
    elif case == "illcond_1e6":
        e = torch.logspace(0.0, -6.0, steps=n, device=device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(
            torch.randn(n, n, device=device, dtype=torch.float32, generator=g)
        )
        A = (Q * e.unsqueeze(0)) @ Q.mT
        A = A.to(dtype=dtype)
    else:
        raise ValueError(case)
    return A


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    g = torch.Generator(device=device)
    g.manual_seed(42)

    p_values = [1, 2, 3, 4, 8]
    sizes = [64, 256]
    cases = ["gaussian", "illcond_1e6"]
    resid_thresh = 0.10
    relerr_thresh = 0.10

    all_pass = True
    print(f"Device: {device} | dtype: {dtype}")
    print("=" * 100)

    for p_val in p_values:
        pe_quad, desc = build_pe_schedules(
            l_target=0.05,
            device=device,
            coeff_mode="tuned",
            coeff_seed=0,
            coeff_safety=1.0,
            coeff_no_final_safety=False,
            p_val=p_val,
        )
        quad_coeffs = _quad_coeffs(pe_quad)

        with torch.inference_mode():
            for n in sizes:
                for case in cases:
                    A = make_spd(n, case, device, dtype, g)
                    A_norm, stats = precond_spd(
                        A, mode="aol", ridge_rel=1e-4, l_target=0.05
                    )

                    methods = {
                        "Uncoupled-Quad": lambda: inverse_proot_pe_quadratic_uncoupled(
                            A_norm, abc_t=quad_coeffs, p_val=p_val, symmetrize_X=True
                        ),
                        "Coupled-Quad": lambda: inverse_proot_pe_quadratic_coupled(
                            A_norm,
                            abc_t=quad_coeffs,
                            p_val=p_val,
                            symmetrize_Y=True,
                            terminal_last_step=True,
                        ),
                    }

                    header = f"p={p_val} n={n} case={case}"
                    for name, fn in methods.items():
                        Xn, _ = fn()
                        if not torch.isfinite(Xn).all():
                            print(f"  X {header:30s} {name:20s} FAIL (non-finite)")
                            all_pass = False
                            continue

                        q = compute_quality_stats(
                            Xn,
                            A_norm,
                            power_iters=0,
                            mv_samples=0,
                            p_val=p_val,
                        )
                        relerr = float(
                            iroot_relative_error(
                                Xn.float(), A_norm.float(), p_val=p_val
                            )
                            .mean()
                            .item()
                        )

                        passed = (
                            q.residual_fro < resid_thresh and relerr < relerr_thresh
                        )
                        mark = "+" if passed else "X"
                        if not passed:
                            all_pass = False
                        print(
                            f"  {mark} {header:30s} {name:20s} "
                            f"resid={q.residual_fro:.3e} relerr={relerr:.3e}"
                        )

    print("=" * 100)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
