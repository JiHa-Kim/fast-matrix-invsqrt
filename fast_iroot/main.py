#!/usr/bin/env python3
import argparse
import math
import time
from typing import Tuple

import torch

from .ops import pct
from .synthetic import make_eig_bank, make_spd_from_eigs, make_tall_random, suite_shapes_default
from .runner import run_one_case, RunSummary

Tensor = torch.Tensor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_root", type=int, choices=[2, 4], default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", choices=["demo", "bank", "suite"], default="suite")

    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--kappa_P", type=float, default=1e7)
    ap.add_argument("--target_action_rel", type=float, default=1e-3)
    ap.add_argument("--max_steps", type=int, default=6)

    ap.add_argument(
        "--input_dtype", choices=["float64", "float32", "bfloat16"], default="float32"
    )
    ap.add_argument("--iter_dtype", choices=["float32", "bfloat16"], default="float32")

    ap.add_argument("--cert_mode", choices=["auto", "exact", "bound"], default="auto")
    ap.add_argument("--exact_threshold", type=int, default=1024)
    ap.add_argument("--rhs_chunk_rows", type=int, default=2048)
    ap.add_argument("--solve_jitter_rel", type=float, default=1e-15)

    ap.add_argument("--oracle_mode", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--oracle_n_max", type=int, default=512)

    ap.add_argument("--bank_size", type=int, default=12)
    ap.add_argument("--suite_cases", type=int, default=6)
    ap.add_argument("--suite_shapes", choices=["default"], default="default")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if args.input_dtype == "float64":
        input_dtype = torch.float64
    elif args.input_dtype == "float32":
        input_dtype = torch.float32
    else:
        input_dtype = torch.bfloat16

    iter_dtype = torch.float32 if args.iter_dtype == "float32" else torch.bfloat16

    print(
        f"device={args.device}  mode={args.mode}  target=G P^(-1/{args.p_root})  "
        f"kappa_P<={args.kappa_P:.3g}  target_action_rel<={args.target_action_rel:.6g}"
    )
    print(
        "knobs: "
        f"max_steps={args.max_steps} input_dtype={args.input_dtype} iter_dtype={args.iter_dtype} "
        f"cert_mode={args.cert_mode} exact_threshold={args.exact_threshold} "
        f"rhs_chunk_rows={args.rhs_chunk_rows} oracle_mode={args.oracle_mode} oracle_n_max={args.oracle_n_max} "
        f"solve_jitter_rel={args.solve_jitter_rel:g}"
    )
    print(f"method: explicit composite type-(1,0) rational minimax iteration for p={args.p_root}")

    def make_case(
        m: int, n: int, case_seed: int, case_idx: int
    ) -> Tuple[Tensor, Tensor]:
        bank = make_eig_bank(
            n,
            args.kappa_P,
            bank_size=max(args.bank_size, args.suite_cases, 8),
            seed=case_seed + 17 * n,
        )
        eigs = bank[case_idx % len(bank)]
        P = make_spd_from_eigs(
            eigs=eigs, seed=case_seed, device=args.device, storage_dtype=input_dtype
        )
        G = make_tall_random(
            m=m, n=n, seed=case_seed + 1, device=args.device, storage_dtype=input_dtype
        )
        return G, P

    def run_case(G: Tensor, P: Tensor) -> RunSummary:
        return run_one_case(
            G_storage=G,
            P_storage=P,
            p_root=args.p_root,
            target_action_rel=args.target_action_rel,
            max_steps=args.max_steps,
            iter_dtype=iter_dtype,
            cert_mode=args.cert_mode,
            exact_threshold=args.exact_threshold,
            rhs_chunk_rows=args.rhs_chunk_rows,
            solve_jitter_rel=args.solve_jitter_rel,
            oracle_mode=args.oracle_mode,
            oracle_n_max=args.oracle_n_max,
        )

    if args.mode == "demo":
        # Warmup
        for _ in range(2):
            G, P = make_case(args.m, args.n, args.seed, 0)
            _ = run_case(G, P)
            del G, P
        
        G, P = make_case(args.m, args.n, args.seed, 0)
        res = run_case(G, P)
        print("")
        print(
            f"demo m={args.m} n={args.n}: success={res.success} "
            f"action_rel_cert={res.action_rel_cert:.6g} "
            f"resid(M)_cert={res.resid_M_cert:.6g} alpha_final={res.alpha_final:.6g} "
            f"alpha_pred_action_rel={res.alpha_pred_action_rel:.6g} "
            f"steps={res.steps} guards={res.guards}"
        )
        if math.isfinite(res.oracle_action_rel_fro):
            print(
                f"  oracle action rel fro={res.oracle_action_rel_fro:.6g} spec={res.oracle_action_rel_spec:.6g} "
                f"replay={res.oracle_replay_rel_fro:.6g} root={res.oracle_root_rel_fro:.6g} "
                f"root_resid={res.oracle_root_resid:.6g}"
            )
        print(
            f"  ms total={res.ms_total:.3f} "
            f"(scale={res.ms_scale:.3f} small={res.ms_small:.3f} apply={res.ms_apply:.3f} cert={res.ms_cert:.3f} oracle={res.ms_oracle:.3f})"
        )
        return

    if args.mode == "bank":
        finals = []
        residuals = []
        alpha_preds = []
        steps = []
        guards = []
        ms_total = []
        oracle_action_fro = []
        oracle_action_spec = []
        oracle_replay = []
        oracle_root = []
        oracle_resid = []

        for i in range(args.bank_size):
            try:
                G, P = make_case(args.m, args.n, args.seed + 1000 + i, i)
                res = run_case(G, P)
                finals.append(res.action_rel_cert)
                residuals.append(res.resid_M_cert)
                alpha_preds.append(res.alpha_pred_action_rel)
                steps.append(res.steps)
                guards.append(res.guards)
                ms_total.append(res.ms_total)
                oracle_action_fro.append(res.oracle_action_rel_fro)
                oracle_action_spec.append(res.oracle_action_rel_spec)
                oracle_replay.append(res.oracle_replay_rel_fro)
                oracle_root.append(res.oracle_root_rel_fro)
                oracle_resid.append(res.oracle_root_resid)
                del G, P
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                finals.append(float("inf"))
                residuals.append(float("inf"))
                alpha_preds.append(float("inf"))
                steps.append(0)
                guards.append(0)
                ms_total.append(float("inf"))
                oracle_action_fro.append(float("nan"))
                oracle_action_spec.append(float("nan"))
                oracle_replay.append(float("nan"))
                oracle_root.append(float("nan"))
                oracle_resid.append(float("nan"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        print("")
        print(f"bank summary (N={len(finals)}):")
        print(
            f"  success <= target action rel: {sum(1 for x in finals if x <= args.target_action_rel)}/{len(finals)}"
        )
        print(
            f"  worst action rel cert: {max(finals):.6g}  median: {pct(finals, 0.5):.6g}  p90: {pct(finals, 0.9):.6g}"
        )
        print(
            f"  resid(M)_cert median: {pct(residuals, 0.5):.6g}  p90: {pct(residuals, 0.9):.6g}"
        )
        print(
            f"  alpha-pred action rel median: {pct(alpha_preds, 0.5):.6g}  p90: {pct(alpha_preds, 0.9):.6g}"
        )
        print(f"  steps median: {pct(steps, 0.5):.6g}  p90: {pct(steps, 0.9):.6g}")
        print(f"  guards median: {pct(guards, 0.5):.6g}  p90: {pct(guards, 0.9):.6g}")
        if any(math.isfinite(x) for x in oracle_action_fro):
            print(
                f"  oracle action rel_fro median: {pct(oracle_action_fro, 0.5):.6g}  p90: {pct(oracle_action_fro, 0.9):.6g}"
            )
            print(
                f"  oracle action rel_spec median: {pct(oracle_action_spec, 0.5):.6g}  p90: {pct(oracle_action_spec, 0.9):.6g}"
            )
            print(
                f"  oracle replay rel_fro median: {pct(oracle_replay, 0.5):.6g}  p90: {pct(oracle_replay, 0.9):.6g}"
            )
            print(
                f"  oracle root rel_fro median: {pct(oracle_root, 0.5):.6g}  p90: {pct(oracle_root, 0.9):.6g}"
            )
            print(
                f"  oracle root resid median: {pct(oracle_resid, 0.5):.6g}  p90: {pct(oracle_resid, 0.9):.6g}"
            )
        print(
            f"  ms total median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}"
        )
        return

    shapes = (
        suite_shapes_default() if args.suite_shapes == "default" else [(args.m, args.n)]
    )

    for m, n in shapes:
        if args.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            print(
                f"\nshape m={m} n={n}  (cuda mem free={free / 1e9:.2f}GB total={total / 1e9:.2f}GB)"
            )
        else:
            print(f"\nshape m={m} n={n}")

        finals = []
        residuals = []
        alpha_preds = []
        steps = []
        guards = []
        ms_total = []
        ms_scale = []
        ms_small = []
        ms_apply = []
        ms_cert = []
        ms_oracle = []
        oracle_action_fro = []
        oracle_action_spec = []
        oracle_replay = []
        oracle_root = []
        oracle_resid = []
        successes = 0

        t0 = time.time()
        for i in range(args.suite_cases):
            try:
                G, P = make_case(m, n, args.seed + 10000 + i, i)
                res = run_case(G, P)
                finals.append(res.action_rel_cert)
                residuals.append(res.resid_M_cert)
                alpha_preds.append(res.alpha_pred_action_rel)
                steps.append(res.steps)
                guards.append(res.guards)
                successes += int(res.success)
                ms_total.append(res.ms_total)
                ms_scale.append(res.ms_scale)
                ms_small.append(res.ms_small)
                ms_apply.append(res.ms_apply)
                ms_cert.append(res.ms_cert)
                ms_oracle.append(res.ms_oracle)
                oracle_action_fro.append(res.oracle_action_rel_fro)
                oracle_action_spec.append(res.oracle_action_rel_spec)
                oracle_replay.append(res.oracle_replay_rel_fro)
                oracle_root.append(res.oracle_root_rel_fro)
                oracle_resid.append(res.oracle_root_resid)
                del G, P
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                print(f"  case {i:02d} OOM (skipping)")
                finals.append(float("inf"))
                residuals.append(float("inf"))
                alpha_preds.append(float("inf"))
                steps.append(0)
                guards.append(0)
                ms_total.append(float("inf"))
                ms_scale.append(float("inf"))
                ms_small.append(float("inf"))
                ms_apply.append(float("inf"))
                ms_cert.append(float("inf"))
                ms_oracle.append(float("inf"))
                oracle_action_fro.append(float("nan"))
                oracle_action_spec.append(float("nan"))
                oracle_replay.append(float("nan"))
                oracle_root.append(float("nan"))
                oracle_resid.append(float("nan"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except Exception as ex:
                print(f"  case {i:02d} FAILED: {type(ex).__name__}: {ex}")
                finals.append(float("inf"))
                residuals.append(float("inf"))
                alpha_preds.append(float("inf"))
                steps.append(0)
                guards.append(0)
                ms_total.append(float("inf"))
                ms_scale.append(float("inf"))
                ms_small.append(float("inf"))
                ms_apply.append(float("inf"))
                ms_cert.append(float("inf"))
                ms_oracle.append(float("inf"))
                oracle_action_fro.append(float("nan"))
                oracle_action_spec.append(float("nan"))
                oracle_replay.append(float("nan"))
                oracle_root.append(float("nan"))
                oracle_resid.append(float("nan"))
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        dt = time.time() - t0
        print(f"  ran {args.suite_cases} cases in {dt:.2f}s")
        print(f"  success <= target action rel: {successes}/{args.suite_cases}")
        print(
            f"  action rel cert median: {pct(finals, 0.5):.6g}  p90: {pct(finals, 0.9):.6g}"
        )
        print(
            f"  resid(M)_cert median: {pct(residuals, 0.5):.6g}  p90: {pct(residuals, 0.9):.6g}"
        )
        print(
            f"  alpha-pred action rel median: {pct(alpha_preds, 0.5):.6g}  p90: {pct(alpha_preds, 0.9):.6g}"
        )
        print(f"  steps median: {pct(steps, 0.5):.6g}  p90: {pct(steps, 0.9):.6g}")
        print(f"  guards median: {pct(guards, 0.5):.6g}  p90: {pct(guards, 0.9):.6g}")
        if any(math.isfinite(x) for x in oracle_action_fro):
            print(
                f"  oracle action rel_fro median: {pct(oracle_action_fro, 0.5):.6g}  p90: {pct(oracle_action_fro, 0.9):.6g}"
            )
            print(
                f"  oracle action rel_spec median: {pct(oracle_action_spec, 0.5):.6g}  p90: {pct(oracle_action_spec, 0.9):.6g}"
            )
            print(
                f"  oracle replay rel_fro median: {pct(oracle_replay, 0.5):.6g}  p90: {pct(oracle_replay, 0.9):.6g}"
            )
            print(
                f"  oracle root rel_fro median: {pct(oracle_root, 0.5):.6g}  p90: {pct(oracle_root, 0.9):.6g}"
            )
            print(
                f"  oracle root resid median: {pct(oracle_resid, 0.5):.6g}  p90: {pct(oracle_resid, 0.9):.6g}"
            )
        else:
            print("  oracle metrics: skipped on all cases")
        print(
            f"  ms total median: {pct(ms_total, 0.5):.3f}  p90: {pct(ms_total, 0.9):.3f}"
        )
        print(
            f"    ms scale median: {pct(ms_scale, 0.5):.3f}  p90: {pct(ms_scale, 0.9):.3f}"
        )
        print(
            f"    ms small median: {pct(ms_small, 0.5):.3f}  p90: {pct(ms_small, 0.9):.3f}"
        )
        print(
            f"    ms apply median: {pct(ms_apply, 0.5):.3f}  p90: {pct(ms_apply, 0.9):.3f}"
        )
        print(
            f"    ms cert  median: {pct(ms_cert, 0.5):.3f}  p90: {pct(ms_cert, 0.9):.3f}"
        )
        print(
            f"    ms oracle median: {pct(ms_oracle, 0.5):.3f}  p90: {pct(ms_oracle, 0.9):.3f}"
        )


if __name__ == "__main__":
    main()
