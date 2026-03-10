#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch

from polar.ops import bf16_target
from polar.runner import RunSummary, run_one_case
from polar.rational.runner_fast import run_one_case_fast
from polar.rational.runner_ultra_fast import run_one_case_ultra_fast
from polar.rational.runner_business import run_one_case_business
from polar.schedules import StepSpec, auto_schedule_name, build_schedule
from polar.synthetic import (
    dtype_from_name,
    make_matrix_from_singulars,
    make_spectrum_bank,
    pct,
    suite_shapes_kimi_glm5,
)
from polar.rational.zolo import mp

Tensor = torch.Tensor


def print_schedule(schedule_name: str, schedule: list[StepSpec]) -> None:
    print(f"chosen schedule: {schedule_name}")
    print("theory schedule:")
    for i, st in enumerate(schedule, 1):
        if st.kind in {"DWH", "DWH_STABLE_SOLVE", "DWH_TUNED_FP32", "DWH_MIXED", "DWH_MIXED_SOLVE"}:
            print(f"  step {i}: {st.kind:<18s} ell_in={st.ell_in:.3e}  pred_kappa(O)_after={st.pred_kappa_after:.8g}")
        elif st.kind == "POLY":
            print(f"  step {i}: POLY d={st.degree:<2d}       ell_in={st.ell_in:.3e}  pred_kappa(O)_after={st.pred_kappa_after:.8g}")
        elif st.kind == "PE":
            print(
                f"  step {i}: PE qdeg={st.pe_degree:<2d} {st.basis:<9s} "
                f"sigma_in=[{st.ell_in:.3e}, {st.u_in:.3e}]  pred_kappa(O)_after={st.pred_kappa_after:.8g}"
            )
        elif st.kind == "PEPAPER5":
            a, b, c = st.paper_coeffs
            print(f"  step {i}: PEPAPER5          a={a:.6g} b={b:.6g} c={c:.6g}")
        else:
            print(f"  step {i}: ZOLO r={st.r:<2d} ell_in={st.ell_in:.3e}  pred_kappa(O)_after={st.pred_kappa_after:.8g}")


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=["demo", "bank", "suite"], default="suite")
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--kappa_G", type=float, default=1e7)
    ap.add_argument("--target_mode", choices=["aggressive", "robust", "custom"], default="robust")
    ap.add_argument("--target_kappa_O", type=float, default=0.0)
    ap.add_argument(
        "--schedule",
        choices=[
            "auto",
            "zolo22",
            "zolo23",
            "zolo32",
            "dwh3",
            "dwh3_stable_solve",
            "dwh3_mixed",
            "dwh3_mixed_solve",
            "dwh3_scaled_fp32",
            "dwh_tuned_fp32",
            "poly16x2",
            "poly24x2",
            "pe2mono12",
            "pe2cheb12",
            "pe3cheb12",
            "pe32hyb12",
            "pe5paper",
        ],
        default="auto",
    )
    ap.add_argument("--input_dtype", choices=["float32", "bfloat16", "float64"], default="float32")
    ap.add_argument("--iter_dtype", choices=["float32", "bfloat16", "float64"], default="float32")
    ap.add_argument("--jitter_rel", type=float, default=1e-15)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--ell0", type=float, default=0.0)
    ap.add_argument("--zolo_coeff_dps", type=int, default=100)
    ap.add_argument("--exact_verify_device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--fast_runner", action="store_true", default=False)
    ap.add_argument("--ultra_fast_runner", action="store_true", default=False)
    ap.add_argument("--hybrid_runner", action="store_true", default=False)
    ap.add_argument("--business_runner", action="store_true", default=False)
    ap.add_argument("--bank_size", type=int, default=12)
    ap.add_argument("--suite_cases", type=int, default=6)
    ap.add_argument("--suite_shapes", choices=["kimi_glm5"], default="kimi_glm5")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def summarize_demo(args: argparse.Namespace, res: RunSummary) -> None:
    print("")
    print(
        f"demo m={args.m} n={args.n}: success={res.success} "
        f"final_kappa(O)_exact={res.final_kO_exact:.8g} "
        f"steps={res.steps} dwh_steps={res.dwh_steps} zolo_steps={res.zolo_steps} "
        f"guards={res.guards} fallbacks={res.fallbacks} last_step={res.last_step_kind}"
    )
    print(
        f"  ms timed total={res.ms_total_timed:.3f} "
        f"(gram={res.ms_gram:.3f} solve={res.ms_solve:.3f} upd={res.ms_upd:.3f})"
    )
    print(f"  ms exact verify (excluded)={res.ms_exact_verify:.3f}")


def main() -> None:
    args = make_parser().parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if mp is None:
        raise RuntimeError("This script requires mpmath for Zolo coefficients")

    input_dtype = dtype_from_name(args.input_dtype)
    iter_dtype = dtype_from_name(args.iter_dtype)
    ell0 = float(args.ell0) if args.ell0 > 0.0 else (1.0 / float(args.kappa_G))
    target_kappa_O = (
        float(args.target_kappa_O) if args.target_mode == "custom" else bf16_target(args.target_mode)
    )

    schedule_name = args.schedule
    if schedule_name == "auto":
        schedule_name = auto_schedule_name(target_kappa_O)
    schedule = build_schedule(schedule_name, ell0, args.zolo_coeff_dps)

    print(
        f"device={args.device}  mode={args.mode}  kappa_G<={args.kappa_G:.3g}  target_kappa(O)<={target_kappa_O:.8g}"
    )
    print(
        "knobs: "
        f"input_dtype={args.input_dtype} iter_dtype={args.iter_dtype} "
        f"jitter_rel={args.jitter_rel:g} "
        f"tf32={args.tf32} exact_verify_device={args.exact_verify_device}"
    )
    print(f"control: ell0={ell0:.6g} target_mode={args.target_mode} zolo_coeff_dps={args.zolo_coeff_dps}")
    print_schedule(schedule_name, schedule)

    def make_case(m: int, n: int, case_seed: int) -> Tensor:
        spectra = make_spectrum_bank(n, args.kappa_G, bank_size=1, seed=case_seed + n)
        return make_matrix_from_singulars(
            m=m,
            singulars=spectra[0],
            seed=case_seed,
            device=args.device,
            storage_dtype=input_dtype,
        )

    def run_case(G: Tensor) -> RunSummary:
        if args.business_runner:
            return run_one_case_business(
                G_storage=G,
                target_kappa_O=target_kappa_O,
                schedule=schedule,
                iter_dtype=iter_dtype,
                jitter_rel=args.jitter_rel,
                tf32=args.tf32,
                exact_verify_device=args.exact_verify_device,
                zolo_coeff_dps=args.zolo_coeff_dps,
            )
        if args.hybrid_runner:
            from polar.rational.runner_hybrid import run_one_case_hybrid
            return run_one_case_hybrid(
                G_storage=G,
                target_kappa_O=target_kappa_O,
                schedule=schedule,
                iter_dtype=iter_dtype,
                jitter_rel=args.jitter_rel,
                tf32=args.tf32,
                exact_verify_device=args.exact_verify_device,
                zolo_coeff_dps=args.zolo_coeff_dps,
            )
        if args.ultra_fast_runner:
            return run_one_case_ultra_fast(
                G_storage=G,
                target_kappa_O=target_kappa_O,
                schedule=schedule,
                iter_dtype=iter_dtype,
                jitter_rel=args.jitter_rel,
                tf32=args.tf32,
                exact_verify_device=args.exact_verify_device,
                zolo_coeff_dps=args.zolo_coeff_dps,
            )
        if args.fast_runner:
            return run_one_case_fast(
                G_storage=G,
                target_kappa_O=target_kappa_O,
                schedule=schedule,
                iter_dtype=iter_dtype,
                jitter_rel=args.jitter_rel,
                tf32=args.tf32,
                exact_verify_device=args.exact_verify_device,
                zolo_coeff_dps=args.zolo_coeff_dps,
            )
        return run_one_case(
            G_storage=G,
            target_kappa_O=target_kappa_O,
            schedule=schedule,
            iter_dtype=iter_dtype,
            jitter_rel=args.jitter_rel,
            tf32=args.tf32,
            exact_verify_device=args.exact_verify_device,
            zolo_coeff_dps=args.zolo_coeff_dps,
        )

    if args.mode == "demo":
        # Warmup
        for _ in range(2):
            _ = run_case(make_case(args.m, args.n, args.seed))
        
        summarize_demo(args, run_case(make_case(args.m, args.n, args.seed)))
        return

    shapes = suite_shapes_kimi_glm5() if args.mode == "suite" else [(args.m, args.n)]
    num_cases = args.suite_cases if args.mode == "suite" else args.bank_size

    for m, n in shapes:
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            print(f"\nshape m={m} n={n}  (cuda mem free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB)")
        else:
            print(f"\nshape m={m} n={n}")

        finals = []
        steps_used = []
        dwh_steps_used = []
        zolo_steps_used = []
        guards_used = []
        fallbacks_used = []
        ms_total = []
        ms_gram = []
        ms_solve = []
        ms_upd = []
        ms_exact_verify = []
        successes = 0

        t0 = time.time()
        for i in range(num_cases):
            try:
                G = make_case(m, n, args.seed + 10000 + i)
                res = run_case(G)
                finals.append(res.final_kO_exact)
                steps_used.append(res.steps)
                dwh_steps_used.append(res.dwh_steps)
                zolo_steps_used.append(res.zolo_steps)
                guards_used.append(res.guards)
                fallbacks_used.append(res.fallbacks)
                successes += int(res.success)
                ms_total.append(res.ms_total_timed)
                ms_gram.append(res.ms_gram)
                ms_solve.append(res.ms_solve)
                ms_upd.append(res.ms_upd)
                ms_exact_verify.append(res.ms_exact_verify)
                del G
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  case {i:02d} OOM (skipping)")
                finals.append(float("inf"))
                steps_used.append(0)
                dwh_steps_used.append(0)
                zolo_steps_used.append(0)
                guards_used.append(0)
                fallbacks_used.append(0)
                ms_total.append(float("inf"))
                ms_gram.append(float("inf"))
                ms_solve.append(float("inf"))
                ms_upd.append(float("inf"))
                ms_exact_verify.append(float("inf"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as ex:
                print(f"  case {i:02d} FAILED: {type(ex).__name__}: {ex}")
                finals.append(float("inf"))
                steps_used.append(0)
                dwh_steps_used.append(0)
                zolo_steps_used.append(0)
                guards_used.append(0)
                fallbacks_used.append(0)
                ms_total.append(float("inf"))
                ms_gram.append(float("inf"))
                ms_solve.append(float("inf"))
                ms_upd.append(float("inf"))
                ms_exact_verify.append(float("inf"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        dt = time.time() - t0
        print(f"  ran {num_cases} cases in {dt:.2f}s")
        print(f"  success <= target: {successes}/{num_cases}")
        print(
            f"  worst kappa(O)_exact: {max(finals):.8g}  median: {pct(finals, 0.5):.8g}  p90: {pct(finals, 0.9):.8g}"
        )
        print(f"  steps median: {pct(steps_used, 0.5):.6g}  p90: {pct(steps_used, 0.9):.6g}")
        print(f"  dwh_steps median: {pct(dwh_steps_used, 0.5):.6g}  p90: {pct(dwh_steps_used, 0.9):.6g}")
        print(f"  zolo_steps median: {pct(zolo_steps_used, 0.5):.6g}  p90: {pct(zolo_steps_used, 0.9):.6g}")
        print(f"  fallbacks median: {pct(fallbacks_used, 0.5):.6g}  p90: {pct(fallbacks_used, 0.9):.6g}")
        print(f"  guards median: {pct(guards_used, 0.5):.6g}  p90: {pct(guards_used, 0.9):.6g}")
        print(f"  ms timed total median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}")
        print(f"    ms gram  median: {pct(ms_gram, 0.5):.3f}  p90: {pct(ms_gram, 0.9):.3f}")
        print(f"    ms solve median: {pct(ms_solve, 0.5):.3f}  p90: {pct(ms_solve, 0.9):.3f}")
        print(f"    ms upd   median: {pct(ms_upd, 0.5):.3f}  p90: {pct(ms_upd, 0.9):.3f}")
        print(f"    ms exact_verify median: {pct(ms_exact_verify, 0.5):.3f}  p90: {pct(ms_exact_verify, 0.9):.3f}  (excluded)")


if __name__ == "__main__":
    main()
