#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch

from polar.ops import bf16_target
from polar.runner import RunSummary, run_one_case
from polar.rational.runner_tf32 import run_one_case_tf32_rational
from polar.schedules import StepSpec, auto_schedule_name, build_schedule
from polar.synthetic import (
    dtype_from_name,
    make_matrix_from_singulars,
    make_spectrum_bank,
    mean_finite,
    pct,
    suite_shapes_kimi_glm5,
    suite_shapes_light,
)

Tensor = torch.Tensor
SCHEDULE_CHOICES = [
    "auto",
    "dwh3",
    "dwh3_stable_solve",
    "dwh_tuned_fp32",
    "dwh3_sigma3x2",
    "dwh3_sigma3x3",
    "dwh4_cubic",
    "dwh4_cubic_cheb",
    "dwh4_sigma2x2",
    "pe5add",
    "pe5paper",
]


def print_schedule(schedule_name: str, schedule: list[StepSpec]) -> None:
    print(f"chosen schedule: {schedule_name}")
    print("theory schedule:")
    for i, st in enumerate(schedule, 1):
        if st.kind in {"DWH", "DWH_STABLE_SOLVE", "DWH_TUNED_FP32", "DWH_MIXED", "DWH_MIXED_SOLVE"}:
            print(f"  step {i}: {st.kind:<18s} ell_in={st.ell_in:.3e}  pred_kappa(O)_after={st.pred_kappa_after:.8g}")
        elif st.kind == "POLY_SIGMA_MAP":
            coeffs = ", ".join(f"{v:.6g}" for v in st.coeffs)
            print(
                f"  step {i}: POLY_SIGMA_MAP "
                f"ell_in={st.ell_in:.3e} pred_kappa(O)_after={st.pred_kappa_after:.8g} "
                f"deg={st.degree} fit={st.fit_kind or 'n/a'} basis={st.basis_kind or 'n/a'} coeffs=({coeffs})"
            )
        elif st.kind == "PEADD5":
            a, b, c = st.coeffs
            print(
                f"  step {i}: PEADD5 "
                f"ell_in={st.ell_in:.3e} pred_kappa(O)_after={st.pred_kappa_after:.8g} "
                f"a={a:.6g} b={b:.6g} c={c:.6g}"
            )
        elif st.kind == "PEPAPER5":
            a, b, c = st.paper_coeffs
            print(f"  step {i}: PEPAPER5          a={a:.6g} b={b:.6g} c={c:.6g}")
        else:
            raise ValueError(f"Unsupported step kind in schedule printout: {st.kind}")


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
        choices=SCHEDULE_CHOICES,
        default="auto",
    )
    ap.add_argument(
        "--compare_schedules",
        nargs="+",
        choices=[name for name in SCHEDULE_CHOICES if name != "auto"],
        default=None,
        help="Run a paired benchmark on the same generated cases for each listed schedule.",
    )
    ap.add_argument("--input_dtype", choices=["float32", "bfloat16", "float64"], default="float32")
    ap.add_argument("--iter_dtype", choices=["float32", "bfloat16", "float64"], default="float32")
    ap.add_argument("--jitter_rel", type=float, default=1e-15)
    ap.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF32 matmuls on CUDA (default: enabled). Use --no-tf32 to disable.",
    )
    ap.add_argument("--ell0", type=float, default=0.0)
    ap.add_argument("--exact_verify_device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--tf32_rational_runner", action="store_true", default=False)
    ap.add_argument("--bank_size", type=int, default=12)
    ap.add_argument("--suite_cases", type=int, default=6)
    ap.add_argument("--suite_shapes", choices=["kimi_glm5", "light"], default="kimi_glm5")
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
    input_dtype = dtype_from_name(args.input_dtype)
    iter_dtype = dtype_from_name(args.iter_dtype)
    ell0 = float(args.ell0) if args.ell0 > 0.0 else (1.0 / float(args.kappa_G))
    target_kappa_O = (
        float(args.target_kappa_O) if args.target_mode == "custom" else bf16_target(args.target_mode)
    )

    schedule_name = args.schedule
    if schedule_name == "auto":
        schedule_name = auto_schedule_name(target_kappa_O)
    schedule = build_schedule(schedule_name, ell0)
    compare_schedule_names = list(args.compare_schedules or [])

    print(
        f"device={args.device}  mode={args.mode}  kappa_G<={args.kappa_G:.3g}  target_kappa(O)<={target_kappa_O:.8g}"
    )
    print(
        "knobs: "
        f"input_dtype={args.input_dtype} iter_dtype={args.iter_dtype} "
        f"jitter_rel={args.jitter_rel:g} "
        f"tf32={args.tf32} exact_verify_device={args.exact_verify_device}"
    )
    print(f"control: ell0={ell0:.6g} target_mode={args.target_mode}")
    print_schedule(schedule_name, schedule)
    if compare_schedule_names:
        print(f"compare schedules: {', '.join(compare_schedule_names)}")

    def make_case(m: int, n: int, case_seed: int) -> Tensor:
        spectra = make_spectrum_bank(n, args.kappa_G, bank_size=1, seed=case_seed + n)
        return make_matrix_from_singulars(
            m=m,
            singulars=spectra[0],
            seed=case_seed,
            device=args.device,
            storage_dtype=input_dtype,
        )

    def run_case_with_schedule(G: Tensor, sched: list[StepSpec]) -> RunSummary:
        if args.tf32_rational_runner:
            return run_one_case_tf32_rational(
                G_storage=G,
                target_kappa_O=target_kappa_O,
                schedule=sched,
                iter_dtype=iter_dtype,
                jitter_rel=args.jitter_rel,
                tf32=args.tf32,
                exact_verify_device=args.exact_verify_device,
            )
        return run_one_case(
            G_storage=G,
            target_kappa_O=target_kappa_O,
            schedule=sched,
            iter_dtype=iter_dtype,
            jitter_rel=args.jitter_rel,
            tf32=args.tf32,
            exact_verify_device=args.exact_verify_device,
        )

    def run_case(G: Tensor) -> RunSummary:
        return run_case_with_schedule(G, schedule)

    if args.mode == "demo":
        # Warmup
        for _ in range(2):
            _ = run_case(make_case(args.m, args.n, args.seed))
        
        summarize_demo(args, run_case(make_case(args.m, args.n, args.seed)))
        return

    if args.mode == "suite":
        if args.suite_shapes == "kimi_glm5":
            shapes = suite_shapes_kimi_glm5()
        elif args.suite_shapes == "light":
            shapes = suite_shapes_light()
        else:
            raise ValueError(f"Unsupported suite shape preset: {args.suite_shapes}")
    else:
        shapes = [(args.m, args.n)]
    num_cases = args.suite_cases if args.mode == "suite" else args.bank_size

    def summarize_stats(
        finals: list[float],
        steps_used: list[int],
        dwh_steps_used: list[int],
        zolo_steps_used: list[int],
        guards_used: list[int],
        fallbacks_used: list[int],
        ms_total: list[float],
        ms_gram: list[float],
        ms_solve: list[float],
        ms_upd: list[float],
        ms_exact_verify: list[float],
        successes: int,
        num_cases_local: int,
        elapsed_s: float,
        label: str | None = None,
    ) -> None:
        if label is not None:
            print(f"  schedule={label}")
        print(f"  ran {num_cases_local} cases in {elapsed_s:.2f}s")
        print(f"  success <= target: {successes}/{num_cases_local}")
        print(
            f"  worst kappa(O)_exact: {max(finals):.8g}  mean: {mean_finite(finals):.8g} "
            f"median: {pct(finals, 0.5):.8g}  p90: {pct(finals, 0.9):.8g}"
        )
        print(f"  steps median: {pct(steps_used, 0.5):.6g}  p90: {pct(steps_used, 0.9):.6g}")
        print(f"  dwh_steps median: {pct(dwh_steps_used, 0.5):.6g}  p90: {pct(dwh_steps_used, 0.9):.6g}")
        print(f"  zolo_steps median: {pct(zolo_steps_used, 0.5):.6g}  p90: {pct(zolo_steps_used, 0.9):.6g}")
        print(f"  fallbacks median: {pct(fallbacks_used, 0.5):.6g}  p90: {pct(fallbacks_used, 0.9):.6g}")
        print(f"  guards median: {pct(guards_used, 0.5):.6g}  p90: {pct(guards_used, 0.9):.6g}")
        print(
            f"  ms timed total mean: {mean_finite(ms_total):.3f}  "
            f"median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}"
        )
        print(
            f"    ms gram  mean: {mean_finite(ms_gram):.3f}  "
            f"median: {pct(ms_gram, 0.5):.3f}  p90: {pct(ms_gram, 0.9):.3f}"
        )
        print(
            f"    ms solve mean: {mean_finite(ms_solve):.3f}  "
            f"median: {pct(ms_solve, 0.5):.3f}  p90: {pct(ms_solve, 0.9):.3f}"
        )
        print(
            f"    ms upd   mean: {mean_finite(ms_upd):.3f}  "
            f"median: {pct(ms_upd, 0.5):.3f}  p90: {pct(ms_upd, 0.9):.3f}"
        )
        print(
            f"    ms exact_verify mean: {mean_finite(ms_exact_verify):.3f}  "
            f"median: {pct(ms_exact_verify, 0.5):.3f}  p90: {pct(ms_exact_verify, 0.9):.3f}  (excluded)"
        )

    def make_summary_bucket() -> dict[str, list[float] | int]:
        return {
            "finals": [],
            "steps_used": [],
            "dwh_steps_used": [],
            "zolo_steps_used": [],
            "guards_used": [],
            "fallbacks_used": [],
            "ms_total": [],
            "ms_gram": [],
            "ms_solve": [],
            "ms_upd": [],
            "ms_exact_verify": [],
            "successes": 0,
        }

    for m, n in shapes:
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            print(f"\nshape m={m} n={n}  (cuda mem free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB)")
        else:
            print(f"\nshape m={m} n={n}")

        if compare_schedule_names:
            compare_schedules = {
                name: build_schedule(name, ell0)
                for name in compare_schedule_names
            }
            summaries = {name: make_summary_bucket() for name in compare_schedule_names}
            win_counts = {name: 0 for name in compare_schedule_names}
            t0 = time.time()
            for i in range(num_cases):
                case_seed = args.seed + 10000 + i
                try:
                    G = make_case(m, n, case_seed)
                    case_results: dict[str, RunSummary] = {}
                    for name, sched in compare_schedules.items():
                        res = run_case_with_schedule(G, sched)
                        case_results[name] = res
                        bucket = summaries[name]
                        bucket["finals"].append(res.final_kO_exact)
                        bucket["steps_used"].append(res.steps)
                        bucket["dwh_steps_used"].append(res.dwh_steps)
                        bucket["zolo_steps_used"].append(res.zolo_steps)
                        bucket["guards_used"].append(res.guards)
                        bucket["fallbacks_used"].append(res.fallbacks)
                        bucket["ms_total"].append(res.ms_total_timed)
                        bucket["ms_gram"].append(res.ms_gram)
                        bucket["ms_solve"].append(res.ms_solve)
                        bucket["ms_upd"].append(res.ms_upd)
                        bucket["ms_exact_verify"].append(res.ms_exact_verify)
                        bucket["successes"] += int(res.success)
                    successful_case_results = [
                        (name, res.ms_total_timed) for name, res in case_results.items() if res.success
                    ]
                    if successful_case_results:
                        best_name, _ = min(successful_case_results, key=lambda item: item[1])
                        win_counts[best_name] += 1
                    del G
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    print(f"  case {i:02d} OOM (paired skip)")
                    for bucket in summaries.values():
                        bucket["finals"].append(float('inf'))
                        bucket["steps_used"].append(0)
                        bucket["dwh_steps_used"].append(0)
                        bucket["zolo_steps_used"].append(0)
                        bucket["guards_used"].append(0)
                        bucket["fallbacks_used"].append(0)
                        bucket["ms_total"].append(float('inf'))
                        bucket["ms_gram"].append(float('inf'))
                        bucket["ms_solve"].append(float('inf'))
                        bucket["ms_upd"].append(float('inf'))
                        bucket["ms_exact_verify"].append(float('inf'))
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                except Exception as ex:
                    print(f"  case {i:02d} FAILED: {type(ex).__name__}: {ex}")
                    for bucket in summaries.values():
                        bucket["finals"].append(float('inf'))
                        bucket["steps_used"].append(0)
                        bucket["dwh_steps_used"].append(0)
                        bucket["zolo_steps_used"].append(0)
                        bucket["guards_used"].append(0)
                        bucket["fallbacks_used"].append(0)
                        bucket["ms_total"].append(float('inf'))
                        bucket["ms_gram"].append(float('inf'))
                        bucket["ms_solve"].append(float('inf'))
                        bucket["ms_upd"].append(float('inf'))
                        bucket["ms_exact_verify"].append(float('inf'))
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
            dt = time.time() - t0
            for name in compare_schedule_names:
                bucket = summaries[name]
                summarize_stats(
                    finals=bucket["finals"],
                    steps_used=bucket["steps_used"],
                    dwh_steps_used=bucket["dwh_steps_used"],
                    zolo_steps_used=bucket["zolo_steps_used"],
                    guards_used=bucket["guards_used"],
                    fallbacks_used=bucket["fallbacks_used"],
                    ms_total=bucket["ms_total"],
                    ms_gram=bucket["ms_gram"],
                    ms_solve=bucket["ms_solve"],
                    ms_upd=bucket["ms_upd"],
                    ms_exact_verify=bucket["ms_exact_verify"],
                    successes=int(bucket["successes"]),
                    num_cases_local=num_cases,
                    elapsed_s=dt,
                    label=name,
                )
                print(f"    paired fastest successful cases: {win_counts[name]}/{num_cases}")
            continue

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
        summarize_stats(
            finals=finals,
            steps_used=steps_used,
            dwh_steps_used=dwh_steps_used,
            zolo_steps_used=zolo_steps_used,
            guards_used=guards_used,
            fallbacks_used=fallbacks_used,
            ms_total=ms_total,
            ms_gram=ms_gram,
            ms_solve=ms_solve,
            ms_upd=ms_upd,
            ms_exact_verify=ms_exact_verify,
            successes=successes,
            num_cases_local=num_cases,
            elapsed_s=dt,
        )


if __name__ == "__main__":
    main()
